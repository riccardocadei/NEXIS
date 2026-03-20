"""
Build the analysis dataset and run NEMS to select SAE features that modify
the treatment effect.

The NEMS model per feature j:
  Y = β₀ + βₜT + βⱼZⱼ + γⱼ(T·Zⱼ) + ε
  H0: γⱼ = 0  (feature j does not modify the treatment effect)

Z is the (individual × 3072) matrix of SAE features, where each individual
inherits the site-level feature vector of their satellite image.

Usage
-----
    python analyze.py [--alpha 0.05] [--max-steps 20]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "uganda"
OUT_DIR  = ROOT / "results" / "uganda"
sys.path.insert(0, str(ROOT / "src"))

from nems import nems_select, marginal_select


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha",     type=float, default=0.05)
    p.add_argument("--max-steps", type=int,   default=20,
                   help="Max NEMS forward steps (caps computation)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load RCT data ─────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "UgandaDataProcessed.csv", low_memory=False)
    df = df.rename(columns={"Wobs": "T", "Yobs": "Y"})

    W_COLS = ["age", "female", "group_female", "father_educ", "mother_educ",
              "karamojan_district"]
    W_COLS = [c for c in W_COLS if c in df.columns]

    # ── Load SAE features (individual level — same vector for each site) ───────
    feat_data = np.load(OUT_DIR / "individual_features.npz")
    Z_all     = feat_data["features"]   # (3142, hidden_dim)

    # ── Build analysis dataset ────────────────────────────────────────────────
    # Keep only obs with observed Y and valid SAE features
    has_feat = np.isfinite(Z_all[:, 0])
    mask     = df["Y"].notna() & has_feat
    df_sub   = df[mask].reset_index(drop=True)
    Z_sub    = Z_all[mask]

    Y = df_sub["Y"].values.astype(float)
    T = df_sub["T"].values.astype(float)

    n, p = Z_sub.shape
    print(f"Analysis dataset:  n={n}  features={p}")
    print(f"Treatment rate:    {T.mean():.1%}  ({int(T.sum())} treated)")
    print()

    # ── Run NEMS ──────────────────────────────────────────────────────────────
    print(f"Running NEMS  (α={args.alpha}, max_steps={args.max_steps})...")
    nems_res = nems_select(Y, T, Z_sub, alpha=args.alpha, max_steps=args.max_steps)
    print(f"  → {len(nems_res.selected)} feature(s) selected: {nems_res.selected}")

    # ── Marginal Bonferroni baseline ──────────────────────────────────────────
    print(f"\nRunning marginal (Bonferroni) baseline...")
    marg_res = marginal_select(Y, T, Z_sub, alpha=args.alpha, adjust="bonferroni")
    print(f"  → {len(marg_res.selected)} feature(s) selected: {marg_res.selected}")

    # ── Per-feature summary ───────────────────────────────────────────────────
    site_data    = np.load(OUT_DIR / "site_features.npz")
    site_feats   = site_data["site_features"]   # (N_exp, hidden_dim)
    site_keys    = site_data["site_keys"]        # (N_exp,)

    print()
    if nems_res.selected:
        print("── NEMS selected features ──────────────────────────────────")
        for rank, feat_idx in enumerate(nems_res.selected):
            act = site_feats[:, feat_idx]
            active_sites = site_keys[act > 0]
            p_val = nems_res.pvalues[feat_idx]
            print(f"  rank={rank+1}  feature={feat_idx:4d}  "
                  f"p={p_val:.2e}  "
                  f"active={( act > 0).mean():.0%} of sites  "
                  f"mean|active={act[act>0].mean():.3f}")
    else:
        print("NEMS selected no features at α={args.alpha}.")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "nems": {
            "selected":  nems_res.selected,
            "pvalues":   nems_res.pvalues.tolist(),
            "alpha":     nems_res.alpha,
            "metadata":  nems_res.metadata,
        },
        "marginal_bonferroni": {
            "selected":  marg_res.selected,
            "pvalues":   marg_res.pvalues.tolist(),
        },
    }
    out_path = OUT_DIR / "nems_result.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
