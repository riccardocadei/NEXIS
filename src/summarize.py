"""
Summarize NEMS results: ATE and CATE/GATE per effect modifier.

Output
------
  1. ATE (Average Treatment Effect) with 95% CI.
  2. For each NEMS-selected feature:
       - Binary feature  → CATE(val=0) and CATE(val=1).
       - Sparse SAE feature (median=0, some nonzero)
                         → GATE(inactive, Z=0) and GATE(active, Z>0).
       - Dense continuous → GATE(below median) and GATE(above median).
     Reports threshold, group sizes, HC1 standard errors and 95% CIs.

Usage
-----
    python src/summarize.py [--embed-model dinov2] [--sae-dim 3072]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "uganda"

sys.path.insert(0, str(Path(__file__).parent))
from uganda import resolve_outcome, w_display


# ── OLS with HC1 standard errors ──────────────────────────────────────────────

def _ols_hc1(y: np.ndarray, X: np.ndarray):
    """Return (beta, se) from OLS with HC1 heteroscedasticity-robust SEs."""
    n, k = X.shape
    try:
        XtXi = np.linalg.pinv(X.T @ X)
        beta  = XtXi @ X.T @ y
        resid = y - X @ beta
        Xe    = X * resid[:, None]
        meat  = Xe.T @ Xe
        V     = XtXi @ meat @ XtXi * (n / (n - k))
        se    = np.sqrt(np.maximum(np.diag(V), 0.0))
        return beta, se
    except np.linalg.LinAlgError:
        return np.full(k, np.nan), np.full(k, np.nan)


def ate_ols(y: np.ndarray, t: np.ndarray, W: np.ndarray | None):
    """OLS: y ~ 1 + t + W.  Returns (estimate, se, n)."""
    n = len(y)
    cols = [np.ones(n), t]
    if W is not None and W.shape[1] > 0:
        keep = np.std(W, axis=0) > 1e-8
        if keep.any():
            cols.append(W[:, keep])
    X = np.column_stack(cols)
    beta, se = _ols_hc1(y, X)
    return float(beta[1]), float(se[1]), n


def ci95(est, se):
    lo, hi = est - 1.96 * se, est + 1.96 * se
    return lo, hi


def fmt_est(est, se):
    lo, hi = ci95(est, se)
    return f"{est:+.4f}  [{lo:+.4f}, {hi:+.4f}]  SE={se:.4f}"


# ── Feature classification ────────────────────────────────────────────────────

def classify_feature(z: np.ndarray):
    """
    Returns ('binary', threshold, label_lo, label_hi) or
            ('continuous', threshold, label_lo, label_hi).

    For sparse SAE features (median=0, some nonzero): split at 0
    → 'inactive' vs 'active'.
    For dense continuous: split at median.
    """
    unique_vals = np.unique(z[np.isfinite(z)])
    if len(unique_vals) <= 2:
        v0, v1 = unique_vals[0], unique_vals[1]
        return "binary", float(v0), str(int(v0)) if v0 == int(v0) else f"{v0:.3f}", \
               str(int(v1)) if v1 == int(v1) else f"{v1:.3f}"

    median = float(np.median(z))
    frac_nonzero = float((z > 0).mean())

    if median == 0.0 and 0.02 < frac_nonzero < 0.98:
        return "sparse", 0.0, f"inactive (Z=0, {1-frac_nonzero:.0%})", \
               f"active   (Z>0, {frac_nonzero:.0%})"

    return "continuous", median, \
           f"≤{median:.4f} ({(z <= median).mean():.0%})", \
           f">{median:.4f}  ({(z > median).mean():.0%})"


# ── Per-feature GATE/CATE ─────────────────────────────────────────────────────

def feature_gate(y, t, W, z, feat_label, p_value, interp_label=None, vlm_label=None, vlm_confidence=None):
    ftype, threshold, lbl_lo, lbl_hi = classify_feature(z)

    if ftype == "binary":
        mask_lo = z == threshold
        mask_hi = ~mask_lo
    else:
        mask_lo = z <= threshold
        mask_hi = z > threshold

    W_lo = W[mask_lo] if W is not None else None
    W_hi = W[mask_hi] if W is not None else None

    gate_lo, se_lo, n_lo = ate_ols(y[mask_lo], t[mask_lo], W_lo)
    gate_hi, se_hi, n_hi = ate_ols(y[mask_hi], t[mask_hi], W_hi)

    return dict(
        label=feat_label, pvalue=p_value, interp=interp_label, vlm_label=vlm_label,
        vlm_confidence=vlm_confidence,
        ftype=ftype, threshold=threshold,
        lbl_lo=lbl_lo, lbl_hi=lbl_hi,
        gate_lo=gate_lo, se_lo=se_lo, n_lo=n_lo,
        gate_hi=gate_hi, se_hi=se_hi, n_hi=n_hi,
        diff=gate_hi - gate_lo,
    )


# ── Data loading ──────────────────────────────────────────────────────────────

def make_lang_dummies(series):
    return pd.get_dummies(series, prefix="lang", dtype=float)


def build_covariates(df):
    parts = []
    for col in ["age", "female", "father_educ", "mother_educ",
                "group_female", "karamojan_district"]:
        if col in df.columns:
            parts.append(df[[col]])
    if "lang_group" in df.columns:
        parts.append(make_lang_dummies(df["lang_group"]))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)


def standardise(df):
    out = df.copy().astype(float)
    for c in out.columns:
        mu, sd = out[c].mean(), out[c].std()
        out[c] = (out[c].fillna(0) - mu) / (sd if sd > 1e-8 else 1.0)
    return out.values


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Summarize NEMS results: ATE and CATE/GATE per effect modifier.")
    p.add_argument("--embed-model", default="dinov2_vitb14")
    p.add_argument("--sae-dim",     type=int, default=3072)
    p.add_argument("--outcome",     default="log_skilled_hours",
                   help="Outcome alias or CSV column name (must match what was passed to analyze.py).")
    p.add_argument("--pipeline",    default="qwen7b", choices=["qwen7b", "qwen72b", "points", "geochat"],
                   help="Which interpretation pipeline's output to use.")
    return p.parse_args()


def main():
    args   = parse_args()
    csv_col = resolve_outcome(args.outcome)
    MODEL_DIR = ROOT / "results" / "uganda" / f"{args.embed_model}_{args.sae_dim}"
    OUT_DIR   = MODEL_DIR / args.outcome

    nems_path = OUT_DIR / "nems_result.json"
    if not nems_path.exists():
        print(f"ERROR: {nems_path} not found."); sys.exit(1)

    with open(nems_path) as f:
        nems_out = json.load(f)

    interp_map      = {}   # feature_idx -> full description sentence
    vlm_label_map   = {}   # feature_idx -> short 2-6 word label
    confidence_map  = {}   # feature_idx -> low | medium | high
    interp_path = OUT_DIR / args.pipeline / "interpretations.json"
    if interp_path.exists():
        with open(interp_path) as f:
            for entry in json.load(f):
                lbl  = entry.get("label", "")
                # activated_concept is the new canonical key; fall back to legacy keys
                desc = (entry.get("activated_concept", "")
                        or entry.get("description", "")
                        or entry.get("group_a_concept", ""))
                interp_map[entry["feature"]]     = desc if desc else lbl
                vlm_label_map[entry["feature"]]  = lbl
                confidence_map[entry["feature"]] = entry.get("confidence", "")

    meta         = nems_out["feature_meta"]
    n_sae        = meta["n_sae_features"]
    w_names      = meta["w_names"]
    selected     = nems_out["nems"]["selected"]

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "UgandaDataProcessed.csv", low_memory=False)
    df = df.rename(columns={"Wobs": "T", csv_col: "Y"})

    feat_data = np.load(MODEL_DIR / "individual_features.npz")
    Z_all     = feat_data["features"]

    has_feat = np.isfinite(Z_all[:, 0])
    mask     = df["Y"].notna() & has_feat
    df_sub   = df[mask].reset_index(drop=True)
    Z_sub    = Z_all[mask]

    W_df  = build_covariates(df_sub)
    W_std = standardise(W_df) if not W_df.empty else None
    W_raw = W_df.values.astype(float)

    Y = df_sub["Y"].values.astype(float)
    T = df_sub["T"].values.astype(float)

    # ── ATE ───────────────────────────────────────────────────────────────────
    ate, ate_se, n_total = ate_ols(Y, T, W_std)
    lo_ate, hi_ate       = ci95(ate, ate_se)

    ate_z = ate / ate_se if ate_se > 0 else np.nan
    ate_p = 2 * (1 - 0.5 * (1 + math.erf(abs(ate_z) / math.sqrt(2))))

    # ── Collect per-feature results ───────────────────────────────────────────
    results = []
    for rank, entry in enumerate(selected):
        idx   = entry["idx"]
        label = entry["label"]
        pval  = entry["pvalue"]
        group = entry.get("group", "")

        if group == "W" or idx >= n_sae:
            col_name = label
            if col_name in W_df.columns:
                z_raw = W_df[col_name].values.astype(float)
            else:
                print(f"  Warning: W column '{col_name}' not found, skipping.")
                continue
        else:
            z_raw = Z_sub[:, idx].astype(float)

        interp      = interp_map.get(idx)      # full description sentence
        vlm_label   = vlm_label_map.get(idx)   # short label for feature name
        vlm_conf    = confidence_map.get(idx)  # low | medium | high
        results.append(feature_gate(Y, T, W_raw, z_raw, label, pval, interp, vlm_label, vlm_conf))

    # ── Print summary ─────────────────────────────────────────────────────────
    sep  = "═" * 65
    line = "─" * 65

    print()
    print(sep)
    print(f"  NEMS Results Summary  —  {args.embed_model}  (SAE dim={args.sae_dim})")
    print(sep)

    print()
    print("ATE (Average Treatment Effect)")
    print(f"  Estimate : {ate:+.4f}")
    print(f"  95% CI   : [{lo_ate:+.4f}, {hi_ate:+.4f}]")
    print(f"  SE       : {ate_se:.4f}    p = {ate_p:.4f}")
    print(f"  n        : {n_total}")

    print()
    print("Effect Modifiers (NEMS-selected, by rank)")
    print(line)

    for rank, r in enumerate(results):
        label = r["label"]
        pval  = r["pvalue"]
        interp = f'  ("{r["interp"]}")' if r["interp"] else ""
        conf   = f'  [VLM conf: {r["vlm_confidence"]}]' if r.get("vlm_confidence") else ""
        print()
        print(f"[{rank+1}] {label}   p={pval:.2e}{interp}{conf}")

        ftype = r["ftype"]
        thr   = r["threshold"]

        if ftype == "binary":
            print(f"    Type      : binary")
            print(f"    CATE({r['lbl_lo']:>2}) : {fmt_est(r['gate_lo'], r['se_lo'])}  n={r['n_lo']}")
            print(f"    CATE({r['lbl_hi']:>2}) : {fmt_est(r['gate_hi'], r['se_hi'])}  n={r['n_hi']}")
        elif ftype == "sparse":
            print(f"    Type      : sparse continuous  (split at Z=0)")
            print(f"    GATE({r['lbl_lo']}) : {fmt_est(r['gate_lo'], r['se_lo'])}  n={r['n_lo']}")
            print(f"    GATE({r['lbl_hi']}) : {fmt_est(r['gate_hi'], r['se_hi'])}  n={r['n_hi']}")
        else:
            print(f"    Type      : continuous  (median split at {thr:.4f})")
            print(f"    GATE({r['lbl_lo']}) : {fmt_est(r['gate_lo'], r['se_lo'])}  n={r['n_lo']}")
            print(f"    GATE({r['lbl_hi']}) : {fmt_est(r['gate_hi'], r['se_hi'])}  n={r['n_hi']}")

        diff = r["diff"]
        lo_d, hi_d = diff - 1.96 * np.sqrt(r["se_hi"]**2 + r["se_lo"]**2), \
                     diff + 1.96 * np.sqrt(r["se_hi"]**2 + r["se_lo"]**2)
        print(f"    Difference: {diff:+.4f}  [{lo_d:+.4f}, {hi_d:+.4f}]  (high − low)")

    print()
    print(sep)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    summary = {
        "model": args.embed_model, "sae_dim": args.sae_dim,
        "ate": {"estimate": ate, "se": ate_se, "ci95": [lo_ate, hi_ate],
                "pvalue": ate_p, "n": n_total},
        "effect_modifiers": [
            {k: (float(v) if isinstance(v, (np.floating, float)) else v)
             for k, v in r.items()}
            for r in results
        ],
    }
    pipeline_dir = OUT_DIR / args.pipeline
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    out_path = pipeline_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved → {out_path}")

    _llm_narrative(summary, results, ate, ate_se, ate_p, OUT_DIR, args.outcome,
                   pipeline=args.pipeline)


# ── Markdown table + narrative ────────────────────────────────────────────────

def _clean_label(label: str, vlm_label: str | None) -> str:
    """Return a short human-readable feature name (for table Feature column and narrative)."""
    if vlm_label:
        return vlm_label
    if label.startswith("SAE_"):
        return label.replace("_", " ")
    return w_display(label)[0]


def _llm_narrative(summary, results, ate, ate_se, ate_p, out_dir: Path,
                   outcome: str = "Yobs", pipeline: str = "qwen7b"):
    """Write a deterministic markdown summary table and templated narrative."""

    model_name = out_dir.name
    lo_ate, hi_ate = ate - 1.96 * ate_se, ate + 1.96 * ate_se
    ate_sig = "*" if ate_p < 0.05 else ""

    header = ("| Rank | Feature | VLM interpretation | VLM conf. | "
              "GATE/CATE (low/0) | GATE/CATE (high/1) | Δ CATE | p-value |\n"
              "|------|---------|-------------------|-----------|"
              "-------------------|-------------------|--------|---------|")
    rows = []
    for i, r in enumerate(results):
        interp  = r.get("interp") or "—"
        conf    = r.get("vlm_confidence") or "—"
        clean   = _clean_label(r["label"], r.get("vlm_label"))
        diff    = r["diff"]
        se_diff = np.sqrt(r["se_hi"]**2 + r["se_lo"]**2)
        lo_d    = diff - 1.96 * se_diff
        hi_d    = diff + 1.96 * se_diff
        sig     = "*" if lo_d * hi_d > 0 else ""
        rows.append(
            f"| {i+1} | {clean} | {interp} | {conf} | "
            f"{r['gate_lo']:+.4f} | {r['gate_hi']:+.4f} | "
            f"{diff:+.4f}{sig} | {r['pvalue']:.2e} |"
        )
    table = "\n".join([header] + rows)

    sig_results = []
    for r in results:
        diff    = r["diff"]
        se_diff = np.sqrt(r["se_hi"]**2 + r["se_lo"]**2)
        lo_d    = diff - 1.96 * se_diff
        hi_d    = diff + 1.96 * se_diff
        if lo_d * hi_d > 0:
            sig_results.append(r)

    ate_sentence = (
        f"The programme had an average treatment effect of {ate:+.4f} "
        f"(SE={ate_se:.4f}, p={ate_p:.4f}, 95% CI [{lo_ate:+.4f}, {hi_ate:+.4f}]), "
        f"{'a statistically significant positive effect on log consumption' if ate_p < 0.05 else 'not statistically significant at the 5% level'}."
    )

    if not sig_results:
        modifier_sentence = (
            "None of the NEMS-selected effect modifiers show a statistically "
            "significant difference in CATE at the 95% level, suggesting "
            "limited detectable heterogeneity given the sample size."
        )
        heterogeneity_sentence = ""
    else:
        parts = []
        for r in sig_results:
            direction = "higher" if r["diff"] > 0 else "lower"
            clean     = _clean_label(r["label"], r.get("vlm_label"))
            ftype     = r["ftype"]
            if ftype == "binary":
                contrast = f"group 1 vs group 0 (Δ={r['diff']:+.4f})"
            else:
                contrast = f"active vs inactive sites (Δ={r['diff']:+.4f})"
            conf_tag = f" [VLM conf: {r['vlm_confidence']}]" if r.get("vlm_confidence") else ""
            parts.append(f"{clean}{conf_tag}: {direction} CATE for {contrast}")
        modifier_sentence = (
            f"Significant treatment effect heterogeneity is found for "
            f"{len(sig_results)} modifier(s): {'; '.join(parts)}."
        )
        positive_mods = [r for r in sig_results if r["diff"] > 0]
        negative_mods = [r for r in sig_results if r["diff"] < 0]
        if positive_mods:
            heterogeneity_sentence = (
                f"The programme was most effective in areas/groups characterised by "
                f"high values of: {', '.join(_clean_label(r['label'], r.get('vlm_label')) for r in positive_mods)}. "
                f"This suggests geographic or socio-economic targeting could improve "
                f"programme efficiency."
            )
        else:
            heterogeneity_sentence = (
                f"The programme was less effective in areas/groups characterised by "
                f"high values of: {', '.join(_clean_label(r['label'], r.get('vlm_label')) for r in negative_mods)}."
            )

    narrative = "\n\n".join(filter(None, [
        ate_sentence, modifier_sentence, heterogeneity_sentence
    ]))

    print()
    print("── Results Summary (Markdown) ───────────────────────────────────────")
    print()
    print(f"**ATE = {ate:+.4f}{ate_sig}**  "
          f"95% CI [{lo_ate:+.4f}, {hi_ate:+.4f}]  "
          f"SE={ate_se:.4f}  p={ate_p:.4f}  n={summary['ate']['n']}")
    print()
    print(table)
    print()
    print(narrative)

    narrative_path = out_dir / pipeline / "narrative.md"
    with open(narrative_path, "w") as f:
        f.write(f"# NEMS Results: {model_name}\n\n")
        f.write(f"**ATE = {ate:+.4f}{ate_sig}**  "
                f"95% CI [{lo_ate:+.4f}, {hi_ate:+.4f}]  "
                f"SE={ate_se:.4f}  p={ate_p:.4f}  n={summary['ate']['n']}\n\n")
        f.write(table + "\n\n")
        f.write("> \\* = 95% CI of Δ CATE excludes zero\n\n")
        f.write(narrative + "\n")
    print(f"Saved → {narrative_path}")


if __name__ == "__main__":
    main()
