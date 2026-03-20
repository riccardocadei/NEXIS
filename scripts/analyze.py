"""
Build the analysis dataset and run NEMS to select SAE features that modify
the treatment effect.

The NEMS model per feature j:
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
    python analyze.py [--embed-model dinov2_vitb14] [--sae-dim 3072]
                      [--alpha 0.05] [--max-steps 20]
                      [--no-w-candidates] [--group-level]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "uganda"
sys.path.insert(0, str(ROOT / "src"))

from nems import nems_select, nems_select_grouped, marginal_select


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embed-model", default="dinov2_vitb14",
                   help="Vision backbone used to produce embeddings "
                        "(determines which results subdir to read).")
    p.add_argument("--sae-dim",     type=int, default=3072,
                   help="SAE hidden dimension (determines results subdir).")
    p.add_argument("--alpha",       type=float, default=0.05)
    p.add_argument("--max-steps",   type=int,   default=20)
    p.add_argument("--no-w-candidates", dest="w_candidates", action="store_false",
                   help="Revert to treating W as nuisance controls "
                        "(partials out T*W unconditionally).")
    p.add_argument("--w-priority", action="store_true",
                   help="Give W covariates priority: if any W candidate clears its "
                        "gate, select from W first regardless of SAE p-values.")
    p.add_argument("--district-dummies", action="store_true",
                   help="One-hot encode district and include all dummies as W candidates.")
    p.add_argument("--group-level", action="store_true",
                   help="Aggregate observations to group level before analysis.")
    p.set_defaults(w_candidates=True)
    return p.parse_args()


def make_lang_dummies(series: pd.Series) -> pd.DataFrame:
    """One-hot encode lang_group (7 levels).

    All dummies are kept (including lang_1) so that T×lang_1 can be tested as
    a candidate effect modifier alongside the other language interactions.
    The resulting main-effects block is rank-deficient by one (dummies sum to
    the intercept), but the QR-based residualisation in nems.py handles this
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
    for col in ["age", "female", "father_educ", "mother_educ",
                "group_female", "karamojan_district"]:
        if col in df.columns:
            renamed = df[[col]].rename(columns={"karamojan_district": "region_karamojan"})
            parts.append(renamed)
    if "lang_group" in df.columns:
        parts.append(make_lang_dummies(df["lang_group"]))
    if district_dummies and "district" in df.columns:
        parts.append(pd.get_dummies(df["district"], prefix="district", dtype=float))
    if not parts:
        return pd.DataFrame(index=df.index)
    return pd.concat(parts, axis=1)


def standardise(df: pd.DataFrame) -> np.ndarray:
    """Z-score each column, fill NaN with 0."""
    out = df.copy().astype(float)
    for c in out.columns:
        mu, sd = out[c].mean(), out[c].std()
        out[c] = (out[c].fillna(0) - mu) / (sd if sd > 1e-8 else 1.0)
    return out.values


# Group-level covariate columns: constant within group, take first value.
_GROUP_LEVEL_COLS = {"lang_group", "karamojan_district", "group_female",
                     "district", "groupid"}


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

    OUT_DIR = ROOT / "results" / "uganda" / f"{args.embed_model}_{args.sae_dim}"
    if not OUT_DIR.exists():
        print(f"ERROR: results directory not found: {OUT_DIR}")
        print("Run train.py first, or check --embed-model / --sae-dim.")
        sys.exit(1)

    # ── Load RCT data ─────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "UgandaDataProcessed.csv", low_memory=False)
    df = df.rename(columns={"Wobs": "T", "Yobs": "Y"})

    # ── Load SAE features (individual level) ──────────────────────────────────
    feat_data      = np.load(OUT_DIR / "individual_features.npz")
    Z_all          = feat_data["features"]
    n_sae_features = Z_all.shape[1]

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
    W_vals = standardise(W_df) if not W_df.empty else None
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
    # W candidates are treated identically to SAE features: they enter Z_full and
    # are tested as 2-regressor [W_k, T*W_k] candidates via the standard FWL OLS.
    # D = [1, T, Z_S, T*Z_S] — no W in D — so Z_tilde for W_k is non-zero and
    # the test of H0: gamma_k = 0 (effect modification) is valid.
    if args.w_candidates and W_vals is not None:
        controls  = None
        main_ctrl = None
        Z_full    = np.hstack([Z_sub, W_vals])
        n_w_cols  = W_vals.shape[1]
    else:
        controls  = W_vals   # partial out W and T*W unconditionally as nuisance
        main_ctrl = None
        Z_full    = Z_sub
        n_w_cols  = 0

    # ── Run NEMS ──────────────────────────────────────────────────────────────
    print(f"Running NEMS  (α={args.alpha}, max_steps={args.max_steps})...")
    if args.w_candidates and n_w_cols > 0:
        groups = {
            "SAE": list(range(n_sae_features)),
            "W":   list(range(n_sae_features, n_sae_features + n_w_cols)),
        }
        priority_groups = ["W"] if args.w_priority else None
        nems_res = nems_select_grouped(Y, T, Z_full, groups=groups,
                                       alpha=args.alpha, max_steps=args.max_steps,
                                       controls=controls, main_controls=main_ctrl,
                                       priority_groups=priority_groups,
                                       verbose=True)
    else:
        nems_res = nems_select(Y, T, Z_full, alpha=args.alpha, max_steps=args.max_steps,
                               controls=controls, main_controls=main_ctrl,
                               verbose=True)
    print(f"  → {len(nems_res.selected)} feature(s) selected: {nems_res.selected}")

    # ── Marginal Bonferroni baseline ──────────────────────────────────────────
    print(f"\nRunning marginal (Bonferroni) baseline...")
    marg_groups = groups if (args.w_candidates and n_w_cols > 0) else None
    marg_res = marginal_select(Y, T, Z_full, alpha=args.alpha, adjust="bonferroni",
                               controls=controls, main_controls=main_ctrl,
                               groups=marg_groups)
    print(f"  → {len(marg_res.selected)} feature(s) selected: {marg_res.selected}")

    # ── Per-feature summary ───────────────────────────────────────────────────
    site_data  = np.load(OUT_DIR / "site_features.npz")
    site_feats = site_data["site_features"]

    def _feature_label(idx: int) -> str:
        if idx < n_sae_features:
            return f"SAE_{idx}"
        return w_names[idx - n_sae_features]

    def _activation_summary(idx: int) -> str:
        if idx < n_sae_features:
            act = site_feats[:, idx]
            if (act > 0).any():
                return (f"active={(act > 0).mean():.0%} of sites  "
                        f"mean|active={act[act>0].mean():.3f}")
            return "never active"
        return "W covariate"

    print()
    if nems_res.selected:
        print("── NEMS selected features ──────────────────────────────────")
        for rank, feat_idx in enumerate(nems_res.selected):
            p_val  = nems_res.pvalues[feat_idx]
            grp    = (nems_res.selected_groups[rank]
                      if rank < len(nems_res.selected_groups) else "")
            grp_str = f"[{grp}]" if grp else ""
            print(f"  rank={rank+1}  {grp_str:6s} feature={_feature_label(feat_idx):16s}  "
                  f"p={p_val:.2e}  {_activation_summary(feat_idx)}")
    else:
        print(f"NEMS selected no features at α={args.alpha}.")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "nems": {
            "selected": [
                {
                    "idx":   i,
                    "label": _feature_label(i),
                    "group": (nems_res.selected_groups[r]
                              if r < len(nems_res.selected_groups) else ""),
                    "pvalue": nems_res.pvalues[i],
                }
                for r, i in enumerate(nems_res.selected)
            ],
            "pvalues":  nems_res.pvalues.tolist(),
            "alpha":    nems_res.alpha,
            "metadata": nems_res.metadata,
        },
        "marginal_bonferroni": {
            "selected": [
                {"idx": i, "label": _feature_label(i), "pvalue": marg_res.pvalues[i]}
                for i in marg_res.selected
            ],
            "pvalues": marg_res.pvalues.tolist(),
        },
        "feature_meta": {
            "n_sae_features":  n_sae_features,
            "n_w_features":    n_w_cols,
            "w_names":         w_names,
            "w_mode":          "candidates" if args.w_candidates else "controls",
            "w_priority":      args.w_priority if args.w_candidates else False,
            "district_dummies": args.district_dummies,
            "level":           level_label,
            "embed_model":     args.embed_model,
            "sae_dim":         args.sae_dim,
        },
    }
    out_path = OUT_DIR / "nems_result.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
