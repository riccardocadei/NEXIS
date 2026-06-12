"""
Build the analysis dataset and run NEXIS to select SAE features that modify
the treatment effect.

The NEXIS model per feature j:
  Y = β₀ + βₜT + βⱼZⱼ + γⱼ(T·Zⱼ) + ε
  H0: γⱼ = 0  (feature j does not modify the treatment effect)

Z is the (obs × features) matrix of SAE features (+ optionally W covariates
when --w-candidates is active), where each individual inherits the site-level
feature vector of their satellite image.

Pre-treatment covariates (W) can be handled in two ways:
  default (--w-candidates)  W is treated identically to SAE features: appended
                            to Z_full and tested as a 2-regressor [W_k, T*W_k]
                            candidate.  D contains only [1, T, Z_S, T*Z_S].
  --no-w-candidates         W and T*W are both partialled out as nuisance;
                            not tested (original behaviour).

Analysis can be run at individual level (default) or group level
(--group-level).  At group level all outcomes, features, and covariates
are aggregated to the group: group-level variables (lang_group,
karamojan_district, group_female) take their first value; individual-level
variables (age, female, father_educ, mother_educ) are averaged across members.

Usage
-----
    python src/analyze.py [--embed-model dinov2] [--sae-dim 3072]
                          [--alpha 0.05] [--max-steps 20]
                          [--no-w-candidates] [--group-level]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT     = Path(__file__).parent.parent.parent.parent   # repo root
DATA_DIR = ROOT / "data" / "uganda"

from method.nexis import nexis, marginal_select
from apps.uganda.data import resolve_outcome


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embed-model", default="dinov2_vitb14",
                   help="Vision backbone used to produce embeddings "
                        "(determines which results subdir to read).")
    p.add_argument("--sae-dim",     type=int, default=3072,
                   help="SAE hidden dimension (determines results subdir).")
    p.add_argument("--alpha",       type=float, default=0.05)
    p.add_argument("--max-steps",   type=int,   default=20)
    p.add_argument("--active-threshold", type=int, default=5,
                   help="Minimum number of RCT sites that must activate a SAE feature "
                        "for it to be included in Z (default 5).")
    p.add_argument("--no-spectral", dest="spectral", action="store_false",
                   help="Do not add spectral indices (NDVI, NDWI, …) as W candidates.")
    p.add_argument("--no-w-candidates", dest="w_candidates", action="store_false",
                   help="Revert to treating W as nuisance controls "
                        "(partials out T*W unconditionally).")
    p.add_argument("--w-priority", action="store_true",
                   help="Give W covariates priority: if any W candidate clears its "
                        "gate, select from W first regardless of SAE p-values.")
    p.add_argument("--district-dummies", action="store_true",
                   help="One-hot encode district and include all dummies as W candidates "
                        "(replaces lang_group dummies to avoid collinearity).")
    p.add_argument("--out-suffix", default="",
                   help="Append a suffix to the model results directory "
                        "(e.g. '_districts') to avoid overwriting the base results.")
    p.add_argument("--group-level", action="store_true",
                   help="Aggregate observations to group level before analysis.")
    p.add_argument("--outcome", default="log_skilled_hours",
                   help="Outcome to analyse. Accepts clean aliases (e.g. log_skilled_hours) "
                        "or raw CSV column names (e.g. Yobs). "
                        "See uganda.OUTCOME_ALIASES for the full list.")
    p.set_defaults(w_candidates=True, spectral=True)
    return p.parse_args()


def make_lang_dummies(series: pd.Series) -> pd.DataFrame:
    """One-hot encode lang_group (7 levels).

    All dummies are kept (including lang_1) so that T×lang_1 can be tested as
    a candidate effect modifier alongside the other language interactions.
    The resulting main-effects block is rank-deficient by one (dummies sum to
    the intercept), but the QR-based residualisation in nexis.py handles this
    gracefully: it projects onto the column space of D, which is unchanged.
    """
    return pd.get_dummies(series, prefix="lang", dtype=float)


def build_covariates(df: pd.DataFrame, district_dummies: bool = False) -> pd.DataFrame:
    """
    Return a DataFrame of pre-treatment covariates from df.
    Always includes all available columns; at group level these have already
    been aggregated appropriately by aggregate_to_groups().

    district_dummies: if True, one-hot encode district and include all dummies.
    """
    parts = []
    for col in ["age", "female", "father_educ", "mother_educ", "group_female"]:
        if col in df.columns:
            parts.append(df[[col]])
    if district_dummies and "district" in df.columns:
        # District dummies replace lang_group dummies (collinear — each lang_group
        # maps to a fixed set of districts, so including both causes rank deficiency)
        parts.append(pd.get_dummies(df["district"], prefix="district", dtype=float))
    elif "lang_group" in df.columns:
        parts.append(make_lang_dummies(df["lang_group"]))
    if not parts:
        return pd.DataFrame(index=df.index)
    return pd.concat(parts, axis=1)


# Group-level covariate columns: constant within group, take first value.
_GROUP_LEVEL_COLS = {"lang_group", "group_female", "district", "groupid"}


def aggregate_to_groups(
    df: pd.DataFrame,
    Z_all: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Aggregate individual-level data to group level.

    - Y: mean per group
    - T: first (constant within group by RCT design)
    - Group-level covariates (lang_group, karamojan_district, group_female):
      first value (constant within group)
    - Individual-level covariates (age, female, father_educ, mother_educ):
      mean across group members
    - SAE features: mean-pooled per group (same site vector for most groups;
      spatial mean for the ~31 groups that span multiple sites)
    """
    df = df.copy()
    df["_row"] = np.arange(len(df))

    # Determine aggregation function per column
    agg_dict: dict = {"Y": "mean", "T": "first", "_row": "first"}
    for col in df.columns:
        if col in ("Y", "T", "_row", "groupid"):
            continue
        if col in _GROUP_LEVEL_COLS:
            agg_dict[col] = "first"
        elif df[col].dtype.kind in ("i", "f", "u"):
            agg_dict[col] = "mean"

    grp_agg = (
        df.groupby("groupid", sort=False)
          .agg(agg_dict)
          .reset_index()
    )

    # Mean-pool SAE features per group
    group_order   = grp_agg["groupid"].values
    group_to_idx  = {g: i for i, g in enumerate(group_order)}
    row_groups    = df["groupid"].values

    Z_grp  = np.zeros((len(group_order), Z_all.shape[1]), dtype=np.float32)
    counts = np.zeros(len(group_order), dtype=np.int32)
    for row_i, gid in enumerate(row_groups):
        if gid in group_to_idx:
            gi = group_to_idx[gid]
            Z_grp[gi] += Z_all[row_i]
            counts[gi] += 1
    Z_grp /= np.maximum(counts[:, None], 1)

    grp_df = grp_agg.drop(columns=["_row"]).reset_index(drop=True)
    return grp_df, Z_grp


def main():
    args = parse_args()
    csv_col = resolve_outcome(args.outcome)

    MODEL_DIR = ROOT / "results" / "uganda" / f"{args.embed_model}_{args.sae_dim}"
    if not MODEL_DIR.exists():
        print(f"ERROR: results directory not found: {MODEL_DIR}")
        print("Run train.py first, or check --embed-model / --sae-dim.")
        sys.exit(1)
    suffix = args.out_suffix if args.out_suffix else ""
    OUT_DIR = MODEL_DIR / (args.outcome + suffix)
    OUT_DIR.mkdir(exist_ok=True)

    # ── Load RCT data ─────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "UgandaDataProcessed.csv", low_memory=False)

    # Guard: skip gracefully if outcome column is absent or entirely null
    if csv_col not in df.columns or df[csv_col].isna().all():
        print(f"SKIP: outcome '{args.outcome}' (column '{csv_col}') is "
              f"{'not in the CSV' if csv_col not in df.columns else 'entirely missing at endline'}. "
              "No analysis produced.")
        sys.exit(0)

    df = df.rename(columns={"Wobs": "T", csv_col: "Y"})

    # ── Load SAE features (individual level) ──────────────────────────────────
    feat_data = np.load(MODEL_DIR / "individual_features.npz")
    Z_all     = feat_data["features"]

    # ── Filter SAE features to those active in ≥ active_threshold RCT sites ──
    site_data       = np.load(MODEL_DIR / "site_features.npz")
    site_feats_full = site_data["site_features"]          # (N_rct, d_hidden)
    community_act   = (site_feats_full > 0).sum(axis=0)  # per feature: # active sites
    active_mask     = community_act >= args.active_threshold
    sae_orig_idx    = np.where(active_mask)[0]            # original SAE indices
    Z_all           = Z_all[:, active_mask]
    site_feats      = site_feats_full[:, active_mask]
    n_sae_features  = int(active_mask.sum())
    print(f"SAE features after active≥{args.active_threshold} filter: "
          f"{n_sae_features}/{site_feats_full.shape[1]}")

    # ── Initial individual-level filter ──────────────────────────────────────
    has_feat = np.isfinite(Z_all[:, 0])
    mask     = df["Y"].notna() & has_feat
    df_ind   = df[mask].reset_index(drop=True)
    Z_ind    = Z_all[mask]

    # ── Group-level aggregation (optional) ────────────────────────────────────
    if args.group_level:
        df_sub, Z_sub = aggregate_to_groups(df_ind, Z_ind)
    else:
        df_sub = df_ind
        Z_sub  = Z_ind

    # ── Build covariate matrix W ──────────────────────────────────────────────
    W_df   = build_covariates(df_sub, district_dummies=args.district_dummies)

    # ── Optionally add spectral indices as W candidates ───────────────────────
    spec_path = DATA_DIR / "satellite" / "rct" / "spectral_indices.csv"
    if args.spectral and spec_path.exists():
        spec_df  = pd.read_csv(spec_path).set_index("site_key")
        spec_cols = list(spec_df.columns)
        # Map each individual to their site's spectral values
        if "geo_long_lat_key" in df_sub.columns:
            spec_mat = np.full((len(df_sub), len(spec_cols)), np.nan)
            for row_i, key in enumerate(df_sub["geo_long_lat_key"].values):
                if pd.notna(key) and int(key) in spec_df.index:
                    spec_mat[row_i] = spec_df.loc[int(key)].values
            spec_part = pd.DataFrame(spec_mat, columns=spec_cols, index=df_sub.index)
            W_df = pd.concat([W_df, spec_part], axis=1)
            print(f"Spectral indices added as W candidates: {spec_cols}")

    W_vals  = W_df.values.astype(float) if not W_df.empty else None
    w_names = list(W_df.columns)

    Y = df_sub["Y"].values.astype(float)
    T = df_sub["T"].values.astype(float)

    level_label = "group" if args.group_level else "individual"
    n, p = Z_sub.shape
    print(f"Analysis dataset:  n={n} ({level_label}-level)  "
          f"SAE features={p}  W covariates={len(w_names)}")
    print(f"Treatment rate:    {T.mean():.1%}  ({int(T.sum())} treated)")
    print(f"W covariates:      {w_names}")
    w_mode_str = 'candidates' if args.w_candidates else 'controls'
    if args.w_candidates and args.w_priority:
        w_mode_str += ' (W-priority)'
    if args.w_candidates and args.district_dummies:
        w_mode_str += ' (district dummies)'
    print(f"W mode:            {w_mode_str}")
    print()

    # ── Decide how W enters the model ─────────────────────────────────────────
    if args.w_candidates and W_vals is not None:
        controls  = None
        main_ctrl = None
        Z_full    = np.hstack([Z_sub, W_vals])
        n_w_cols  = W_vals.shape[1]
    else:
        controls  = W_vals
        main_ctrl = None
        Z_full    = Z_sub
        n_w_cols  = 0

    # ── Run NEXIS (3 correction variants) ─────────────────────────────────────
    print(f"Running NEXIS exploratory  (α={args.alpha}, adjust=None)...")
    nexis_expl = nexis(Y, T, Z_full, alpha=args.alpha, max_rounds=args.max_steps,
                       adjust=None, verbose=True)
    print(f"  → {len(nexis_expl.selected)} feature(s) selected")

    print(f"\nRunning NEXIS FDR  (α={args.alpha}, adjust=FDR)...")
    nexis_fdr = nexis(Y, T, Z_full, alpha=args.alpha, max_rounds=args.max_steps,
                      adjust="FDR", verbose=True)
    print(f"  → {len(nexis_fdr.selected)} feature(s) selected")

    print(f"\nRunning NEXIS FWER  (α={args.alpha}, adjust=FWER)...")
    nexis_fwer = nexis(Y, T, Z_full, alpha=args.alpha, max_rounds=args.max_steps,
                       adjust="FWER", verbose=True)
    print(f"  → {len(nexis_fwer.selected)} feature(s) selected")

    # ── Marginal baseline (unadjusted / exploratory only) ────────────────────
    print(f"\nRunning marginal (unadjusted) baseline...")
    marg_none = marginal_select(Y, T, Z_full, alpha=args.alpha, adjust=None)
    print(f"  → {len(marg_none.selected)} feature(s) selected")

    # ── Per-feature summary helpers ───────────────────────────────────────────
    # site_feats is already loaded (filtered to active features)

    def _feature_label(idx: int) -> str:
        if idx < n_sae_features:
            return f"Z_{int(sae_orig_idx[idx])}"
        return f"W_{w_names[idx - n_sae_features]}"

    def _activation_summary(idx: int) -> str:
        if idx < n_sae_features:
            act = site_feats[:, idx]
            if (act > 0).any():
                return (f"active={(act > 0).mean():.0%} of sites  "
                        f"mean|active={act[act>0].mean():.3f}")
            return "never active"
        return "W covariate"

    def _nexis_selected(res):
        return [{"idx": i, "label": _feature_label(i),
                 "group": "SAE" if i < n_sae_features else "W",
                 "pvalue": float(res.pvalues[i])}
                for i in res.selected]

    def _marg_selected(res):
        return [{"idx": i, "label": _feature_label(i),
                 "group": "SAE" if i < n_sae_features else "W",
                 "pvalue": float(res.pvalues[i])}
                for i in res.selected]

    print()
    for tag, res in [("exploratory", nexis_expl), ("FDR", nexis_fdr), ("FWER", nexis_fwer)]:
        if res.selected:
            print(f"── NEXIS ({tag}) selected features ───────────────────────")
            for rank, feat_idx in enumerate(res.selected):
                p_val = res.pvalues[feat_idx]
                print(f"  rank={rank+1}  feature={_feature_label(feat_idx):16s}  "
                      f"p={p_val:.2e}  {_activation_summary(feat_idx)}")
        else:
            print(f"NEXIS ({tag}) selected no features at α={args.alpha}.")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "nexis_exploratory": {
            "selected": _nexis_selected(nexis_expl),
            "pvalues":  nexis_expl.pvalues.tolist(),
            "alpha":    nexis_expl.alpha,
            "metadata": nexis_expl.metadata,
        },
        "nexis_fdr": {
            "selected": _nexis_selected(nexis_fdr),
            "pvalues":  nexis_fdr.pvalues.tolist(),
            "alpha":    nexis_fdr.alpha,
            "metadata": nexis_fdr.metadata,
        },
        "nexis_fwer": {
            "selected": _nexis_selected(nexis_fwer),
            "pvalues":  nexis_fwer.pvalues.tolist(),
            "alpha":    nexis_fwer.alpha,
            "metadata": nexis_fwer.metadata,
        },
        "marginal_exploratory": {
            "selected": _marg_selected(marg_none),
            "pvalues":  marg_none.pvalues.tolist(),
        },
        "feature_meta": {
            "n_sae_features":   n_sae_features,
            "sae_active_idx":   sae_orig_idx.tolist(),
            "active_threshold": args.active_threshold,
            "n_w_features":     n_w_cols,
            "w_names":          w_names,
            "w_mode":           "candidates" if args.w_candidates else "controls",
            "w_priority":       args.w_priority if args.w_candidates else False,
            "district_dummies": args.district_dummies,
            "spectral":         args.spectral,
            "level":            level_label,
            "embed_model":      args.embed_model,
            "sae_dim":          args.sae_dim,
        },
    }
    out_path = OUT_DIR / "nexis_result.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
