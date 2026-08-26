#!/usr/bin/env python3
"""
Agreement *rates* between two dictionaries, run by run — rather than averages of
precision/recall/IoU over heterogeneous design cells.

Every replication is a (design cell, seed) pair, and the two arms see the identical dataset
there (the RCT draw is seeded independently of the representation), so their IoUs are directly
comparable. The agreement rate at tolerance tau is

    A(tau) = share of replications with | IoU_B - IoU_A | <= tau .

A(tau) on its own is not interpretable: even one fixed dictionary does not reproduce its own
IoU across Monte Carlo draws. The script therefore reports the same statistic for a
*same-dictionary* reference — arm A at seed s versus arm A at seed s+S/2 in the same design
cell — which is the agreement one gets from sampling noise alone with the dictionary held
fixed. Cross-dictionary agreement close to that reference means the two dictionaries are
interchangeable at the resolution the experiment can resolve.

Cell-level rows aggregate first over the 50 seeds of a cell and then compare curves, which is
the unit the paper's figures actually plot.

Usage
-----
    python src/apps/celeba/agreement_rates.py --tag b1
    python src/apps/celeba/agreement_rates.py --tag b1 --metric precision --taus 0 0.1 0.2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

CONFIGS = [(20, "sae"), (20, "sae_precode"), (5, "sae"), (5, "sae_precode")]
VIEW = {"sae": "z", "sae_precode": "z_pre"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag",          default="b1")
    p.add_argument("--results-root", type=Path, default="results/celeba")
    p.add_argument("--method",       default="NEXIS")
    p.add_argument("--metric",       default="iou", choices=["iou", "precision", "recall"])
    p.add_argument("--taus",         type=float, nargs="+", default=[0.0, 0.1, 0.2])
    p.add_argument("--cell-taus",    type=float, nargs="+", default=[0.05, 0.10])
    p.add_argument("--out-dir",      type=Path, default=None)
    return p.parse_args()


def load(base: Path, k: int, feature: str, method: str, metric: str) -> pd.DataFrame:
    """Long frame with one row per (cell, seed) for one method."""
    frames = []
    for fname, sweep_col, fixed_col in [("effect_sweep.parquet", "effect_scale", "fixed_n"),
                                        ("n_sweep.parquet",      "n",            "fixed_effect")]:
        d = pd.read_parquet(base / f"k{k}" / feature / fname)
        d = d[d["method"] == method].copy()
        d["cell"] = (fname[0] + "|" + d[fixed_col].astype(str) + "|" + d[sweep_col].astype(str))
        frames.append(d[["cell", "seed", metric]])
    return (pd.concat(frames, ignore_index=True)
              .sort_values(["cell", "seed"]).reset_index(drop=True))


def rates(diff: np.ndarray, taus) -> dict[float, float]:
    return {t: float(np.mean(np.abs(diff) <= t + 1e-12)) for t in taus}


def main() -> int:
    args = parse_args()
    root = ROOT / args.results_root
    base_a, base_b = root / "experiment", root / f"experiment_resample_{args.tag}"
    out_dir = args.out_dir or (root / f"agreement_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    m = args.metric

    rows, cell_rows = [], []
    for k, feature in CONFIGS:
        a = load(base_a, k, feature, args.method, m)
        b = load(base_b, k, feature, args.method, m)
        assert (a["cell"].values == b["cell"].values).all()
        assert (a["seed"].values == b["seed"].values).all()

        # cross-dictionary, same dataset
        d_cross = b[m].values - a[m].values

        # same dictionary, different draw: seed s vs seed s + S/2 in the same cell
        wide = a.pivot(index="cell", columns="seed", values=m)
        seeds = list(wide.columns)
        half = len(seeds) // 2
        d_within = np.concatenate([wide[seeds[i]].values - wide[seeds[i + half]].values
                                   for i in range(half)])

        r_cross, r_within = rates(d_cross, args.taus), rates(d_within, args.taus)
        rows.append({"k": k, "view": VIEW[feature], "n_runs": len(d_cross),
                     **{f"cross_tau{t:g}":  r_cross[t]  for t in args.taus},
                     **{f"within_tau{t:g}": r_within[t] for t in args.taus},
                     "mean_abs_diff_cross":  float(np.abs(d_cross).mean()),
                     "mean_abs_diff_within": float(np.abs(d_within).mean())})

        # cell level: average over the 50 seeds first, then compare curves
        ca = a.groupby("cell")[m].mean()
        cb = b.groupby("cell")[m].mean()
        dc = (cb - ca).values
        cw = np.concatenate([wide[seeds[:half]].mean(axis=1).values
                             - wide[seeds[half:]].mean(axis=1).values])
        cell_rows.append({"k": k, "view": VIEW[feature], "n_cells": len(dc),
                          **{f"cross_tau{t:g}":  rates(dc, [t])[t] for t in args.cell_taus},
                          **{f"within_tau{t:g}": rates(cw, [t])[t] for t in args.cell_taus},
                          "mean_abs_diff_cross":  float(np.abs(dc).mean()),
                          "mean_abs_diff_within": float(np.abs(cw).mean())})

    df, dfc = pd.DataFrame(rows), pd.DataFrame(cell_rows)
    df.to_csv(out_dir / f"agreement_rates_run_{m}.csv", index=False)
    dfc.to_csv(out_dir / f"agreement_rates_cell_{m}.csv", index=False)

    lines = [f"# Agreement rates on {m.upper()} ({args.method}, tag {args.tag})\n",
             "Cross = the two dictionaries on the identical dataset. "
             "Within = one fixed dictionary (the paper's) on two different Monte Carlo draws "
             "of the same design cell — the noise floor.\n",
             "## Replication level (one row per design cell × seed)\n"]
    hdr = "| k | view | runs | " + " | ".join(
        [f"cross \\|Δ\\|≤{t:g} | within \\|Δ\\|≤{t:g}" for t in args.taus]) + \
        " | mean \\|Δ\\| cross | within |"
    lines += [hdr, "|" + "---|" * (3 + 2 * len(args.taus) + 2)]
    for _, r in df.iterrows():
        cells = " | ".join([f"{r[f'cross_tau{t:g}']:.2f} | {r[f'within_tau{t:g}']:.2f}"
                            for t in args.taus])
        lines.append(f"| {r['k']} | {r['view']} | {r['n_runs']} | {cells} | "
                     f"{r['mean_abs_diff_cross']:.3f} | {r['mean_abs_diff_within']:.3f} |")

    lines += ["\n## Design-cell level (seed-averaged curves, 42 cells)\n"]
    hdr = "| k | view | cells | " + " | ".join(
        [f"cross \\|Δ\\|≤{t:g} | within \\|Δ\\|≤{t:g}" for t in args.cell_taus]) + \
        " | mean \\|Δ\\| cross | within |"
    lines += [hdr, "|" + "---|" * (3 + 2 * len(args.cell_taus) + 2)]
    for _, r in dfc.iterrows():
        cells = " | ".join([f"{r[f'cross_tau{t:g}']:.2f} | {r[f'within_tau{t:g}']:.2f}"
                            for t in args.cell_taus])
        lines.append(f"| {r['k']} | {r['view']} | {r['n_cells']} | {cells} | "
                     f"{r['mean_abs_diff_cross']:.3f} | {r['mean_abs_diff_within']:.3f} |")

    report = "\n".join(lines) + "\n"
    (out_dir / f"agreement_rates_{m}.md").write_text(report)
    print(report)
    print(f"Wrote {out_dir}/agreement_rates_{m}.md (+ CSVs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
