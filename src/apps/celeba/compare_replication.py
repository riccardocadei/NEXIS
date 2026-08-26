#!/usr/bin/env python3
"""
Side-by-side comparison of the primary test-ablation run (seeds 0–49) against an
independent replication on a disjoint seed block (seeds 50–99).

The seed *is* the Monte-Carlo draw, so the two blocks are independent samples of the
same estimand; agreement within a couple of standard errors is what "replicates" means
here.  Reported per test and per metric: primary, replication, and the difference in
units of the pooled Monte-Carlo standard error.

Usage
-----
    python src/apps/celeba/compare_replication.py --dgp attr
    python src/apps/celeba/compare_replication.py --dgp ushape
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent

from apps.celeba.table_test_ablation import TESTS, BASELINE, THR, _threshold, _fmt_thr

TREES = {
    "attr":   (("experiment",        "sae"),
               ("experiment_rep",    "sae")),
    "ushape": (("experiment_ushape", "sae_precode"),
               ("experiment_ushape_rep", "sae_precode")),
}


def _load(tree: str, feat: str, k: int):
    base = ROOT / "results/celeba" / tree / f"k{k}" / feat
    return (pd.read_parquet(base / "n_sweep.parquet"),
            pd.read_parquet(base / "effect_sweep.parquet"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dgp", choices=["attr", "ushape"], default="attr")
    p.add_argument("--k", type=int, default=20)
    args = p.parse_args()

    (t1, f1), (t2, f2) = TREES[args.dgp]
    n1, e1 = _load(t1, f1, args.k)
    n2, e2 = _load(t2, f2, args.k)

    s1 = sorted(n1["seed"].unique()); s2 = sorted(n2["seed"].unique())
    print(f"\nprimary     : {t1}/k{args.k}/{f1}  seeds {min(s1)}–{max(s1)} ({len(s1)})")
    print(f"replication : {t2}/k{args.k}/{f2}  seeds {min(s2)}–{max(s2)} ({len(s2)})")
    if set(s1) & set(s2):
        print(f"WARNING: seed blocks overlap on {sorted(set(s1) & set(s2))[:5]}… "
              f"— this is NOT an independent replication")

    ref_e = max(n1["fixed_effect"].unique())
    ref_n = max(e1["fixed_n"].unique())
    print(f"reference operating point: n={ref_n:g}, η={ref_e:g}\n")

    methods = {BASELINE: "marginal (FWER)", **TESTS}
    rows = []
    for key, label in methods.items():
        r: dict[str, object] = {"test": label}
        for eta in sorted(n1["fixed_effect"].unique()):
            a = _threshold(n1, "n", "fixed_effect", eta, key)
            b = _threshold(n2, "n", "fixed_effect", eta, key)
            r[f"n*(η={eta:g})"] = f"{_fmt_thr(a)} / {_fmt_thr(b)}"
        for metric, short in [("precision", "Prec"), ("recall", "Rec")]:
            va, vb = [], []
            for df, acc in ((n1, va), (n2, vb)):
                sub = df[(df["fixed_effect"] == ref_e) & (df["n"] == ref_n)
                         & (df["method"] == key)]
                acc.extend(sub[metric].to_numpy(dtype=float))
            va, vb = np.asarray(va), np.asarray(vb)
            if va.size == 0 or vb.size == 0:
                r[short] = "—"
                continue
            sa = va.std(ddof=1) / np.sqrt(va.size) if va.size > 1 else 0.0
            sb = vb.std(ddof=1) / np.sqrt(vb.size) if vb.size > 1 else 0.0
            pooled = np.hypot(sa, sb)
            z = (vb.mean() - va.mean()) / pooled if pooled > 1e-12 else 0.0
            r[short] = f"{va.mean():.2f} / {vb.mean():.2f}  ({z:+.1f}σ)"
        rows.append(r)

    tab = pd.DataFrame(rows)
    cols = list(tab.columns)
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in tab.iterrows():
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    print("\nEach cell: primary / replication.  (±σ) is the difference in pooled "
          "Monte-Carlo standard errors.")


if __name__ == "__main__":
    main()
