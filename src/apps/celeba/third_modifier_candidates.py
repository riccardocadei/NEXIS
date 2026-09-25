#!/usr/bin/env python3
"""
Rank CelebA attributes as the third direct modifier of the r = 3 DGP.

For every attribute, on the main k = 20 SigLIP dictionary (19,867-image pool):

  j*        best-threshold-F1 coordinate on the pre-activations (the ground-truth rule)
  F1, gap   its F1 and the gap to the runner-up coordinate
  F1 base   F1 of predicting every image positive, 2p/(1+p): best-threshold F1 is
            inflated for high-prevalence labels, so F1 is read against this floor
  AUC       of the pre-activation z_pre[:, j*], and of the sparse code z[:, j*] that
            NEXIS actually regresses on (ties at 0 when the code does not fire)
  code P/R  precision and recall of "code fires" (z[:, j*] > 0) for the label
  lift      share of positives among the 100 images with the largest z_pre[:, j*],
            over the prevalence (its maximum is 1/p)
  distinct  j* differs from the hat and glasses principals 5348 and 5537
  n_max     the independent sampler (scm.generate_celeba_rct) draws the three
            attributes independently at CelebA prevalence and takes images without
            replacement from the matching joint cell; n_max is the largest n of the
            sweep grid for which all 50 seeds find every cell populated with
            probability >= 0.99 (binomial cell counts); "(1,1,1)" is the number of
            images with hat, glasses and the attribute.

Reads the cached F1 matrix results/celeba/interleaved_backward/align/
f1_sae_precode_k20.npz (interleaved_backward_align.py) and writes
results/celeba/figures_v2/third_modifier_candidates.md.

    python src/apps/celeba/third_modifier_candidates.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent.parent
N_GRID = [50, 100, 200, 350, 500, 750, 1000, 2000, 3500, 5000, 10000]
PRINCIPALS = {"Wearing_Hat": 5348, "Eyeglasses": 5537}


def main():
    f1z = np.load(ROOT / "results/celeba/interleaved_backward/align/f1_sae_precode_k20.npz",
                  allow_pickle=True)
    F1, attrs = f1z["f1"], [str(a) for a in f1z["attrs"]]
    L = pd.read_parquet(ROOT / "data/celeba/labels.parquet")
    P = np.load(ROOT / "data/celeba/embeddings/sae_precode_k20.npy", mmap_mode="r")
    S = np.load(ROOT / "data/celeba/embeddings/sae_k20.npy", mmap_mode="r")
    hat, gl = L.Wearing_Hat.values.astype(int), L.Eyeglasses.values.astype(int)
    ph, pg = hat.mean(), gl.mean()

    rows = []
    for i, a in enumerate(attrs):
        if a in PRINCIPALS:
            continue
        y = L[a].values.astype(int)
        p = y.mean()
        order = np.argsort(F1[i])[::-1]
        j = int(order[0])
        zp, zc = np.asarray(P[:, j]), np.asarray(S[:, j])
        fires = zc > 0
        top = np.argsort(zp)[::-1][:100]
        p_ok = []
        for n in N_GRID:
            ok = 1.0
            for h in (0, 1):
                for g in (0, 1):
                    for v in (0, 1):
                        cnt = int(((hat == h) & (gl == g) & (y == v)).sum())
                        pc = (ph if h else 1 - ph) * (pg if g else 1 - pg) * (p if v else 1 - p)
                        ok *= binom.cdf(cnt, n, pc)
            p_ok.append(ok ** 50)
        feas = [n for n, q in zip(N_GRID, p_ok) if q >= 0.99]
        rows.append({
            "attribute": a, "prevalence": p, "j*": j, "F1": F1[i, j],
            "gap": F1[i, j] - F1[i, order[1]], "F1 base": 2 * p / (1 + p),
            "AUC z_pre": roc_auc_score(y, zp), "AUC code": roc_auc_score(y, zc),
            "code P": y[fires].mean() if fires.any() else np.nan, "code R": fires[y == 1].mean(),
            "lift": y[top].mean() / p, "max lift": 1 / p,
            "distinct": j not in PRINCIPALS.values(),
            "(1,1,1)": int(((hat == 1) & (gl == 1) & (y == 1)).sum()),
            "n_max": max(feas) if feas else 0,
        })
    df = pd.DataFrame(rows).sort_values("F1", ascending=False)

    def fmt(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "NO"
        if isinstance(v, float):
            return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
        return str(v)

    cols = list(df.columns)
    md = ["# Third direct modifier for the r = 3 DGP: alignment and feasibility", "",
          "Main k=20 SigLIP dictionary, 19,867-image pool; Wearing_Hat (5348) and "
          "Eyeglasses (5537) excluded. Columns: see src/apps/celeba/third_modifier_candidates.py. "
          "n_max = largest sweep n (grid 50..10,000) at which the independent sampler "
          "populates every (hat, glasses, attribute) cell for all 50 seeds with "
          "probability >= 0.99; 0 = not even n = 50.", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    out = ROOT / "results/celeba/figures_v2/third_modifier_candidates.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
