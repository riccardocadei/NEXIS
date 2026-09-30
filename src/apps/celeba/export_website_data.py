#!/usr/bin/env python3
"""
Export the CelebA sweep curves shown on the project website (docs/index.html).

Reads the terminal-backward-step runs behind paper Figure 3 and the Appendix C
ablations and writes, per dictionary / DGP and sweep, the mean and standard error
over seeds of precision, recall and IoU for every method. This is the same
aggregation as src/apps/celeba/visualize.py::plot_sweep (mean, pandas .sem(); the
site draws the +-1.96 SE band).

  results/celeba/experiment_v2/<k>/<view>/        dictionary and algorithm ablations
                                                  (scripts/celeba/submit_experiment.sh)
  results/celeba/experiment_v2_r{1,3}/k20/sae/    DGP ablation, r = 1 and r = 3 direct
                                                  modifiers (scripts/celeba/submit_dgp_extra.sh)
  results/celeba/experiment_v2_r0_fixbeta/k20/sae/  DGP ablation, r = 0 (n sweep only,
                                                  200 seeds); as in figure_dgp_extra.py
  results/celeba/experiment_v2_resample_b1/k20/sae/  main setting on the independently
                                                  retrained k = 20 replica SAE (paper
                                                  Figure replica_k20)

At r = 0 there is no target, so precision and recall are undefined; the site shows the
share of runs with at least one false discovery (paper Table "DGP ablation: no effect
modification"), with its binomial standard error, and the mean number of false
discoveries.

Method names drop the "-v2" tag, so the paper default is "NEXIS" and its ablations
are "NEXIS (test=...)", "NEXIS (adjust=...)", "NEXIS (rho=...)",
"NEXIS (terminal=False)", "NEXIS (interleaved=True)", ...

Output: docs/assets/celeba_data.json
  {"k20_sae" | "k20_precode" | "k5_sae" | "k20_sae_rep" | "dgp_r1" | "dgp_r3":
      {"n_sweep":      [{method, n, fixed_effect, <metric>_mean, <metric>_se}, ...],
       "effect_sweep": [{method, effect_scale, fixed_n, <metric>_mean, <metric>_se}, ...]},
   "dgp_r0":
      {"n_sweep":      [{method, n, fixed_effect, anyfd_mean, anyfd_se, fd_mean}, ...]}}

"k20_sae" is also the r = 2 reference of the DGP ablation.

Usage
-----
    python src/apps/celeba/export_website_data.py
    python src/apps/celeba/export_website_data.py --results-dir /path/to/results/celeba
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent

# website key -> results subdirectory, relative to results/celeba
VARIANTS = {
    "k20_sae":     "experiment_v2/k20/sae",           # main setting: k = 20, sparse codes Z
    "k20_precode": "experiment_v2/k20/sae_precode",   # k = 20, dense pre-activations Z_pre
    "k5_sae":      "experiment_v2/k5/sae",            # k = 5, sparse codes Z
    "k20_sae_rep": "experiment_v2_resample_b1/k20/sae",  # replica SAE, k = 20, Z
    "dgp_r1":      "experiment_v2_r1/k20/sae",        # one direct modifier (Wearing_Hat)
    "dgp_r3":      "experiment_v2_r3/k20/sae",        # three (+ Sideburns)
}
R0 = ("dgp_r0", "experiment_v2_r0_fixbeta/k20/sae")  # no direct modifier
SWEEPS = {
    "n_sweep":      ("n", "fixed_effect"),
    "effect_sweep": ("effect_scale", "fixed_n"),
}
METRICS = ("precision", "recall", "iou")


def rename(method: str) -> str:
    return method.replace("NEXIS-v2", "NEXIS")


def aggregate(df: pd.DataFrame, xcol: str, fcol: str) -> list[dict]:
    g = df.groupby(["method", fcol, xcol])[list(METRICS)]
    mean, se = g.mean(), g.sem()
    rows = []
    for (method, fval, x), m in mean.iterrows():
        s = se.loc[(method, fval, x)]
        row = {"method": rename(method), xcol: _num(x), fcol: _num(fval)}
        for k in METRICS:
            row[f"{k}_mean"] = round(float(m[k]), 4)
            row[f"{k}_se"] = round(float(s[k]), 4)
        rows.append(row)
    return rows


def aggregate_r0(df: pd.DataFrame) -> list[dict]:
    """Share of runs with any false discovery (every selection is one at r = 0)."""
    df = df.assign(anyfd=(df["n_selected"] > 0).astype(float))
    g = df.groupby(["method", "fixed_effect", "n"])
    mean, se, fd = g["anyfd"].mean(), g["anyfd"].sem(), g["n_selected"].mean()
    return [{"method": rename(m), "n": _num(n), "fixed_effect": _num(f),
             "anyfd_mean": round(float(mean[m, f, n]), 4),
             "anyfd_se": round(float(se[m, f, n]), 4),
             "fd_mean": round(float(fd[m, f, n]), 4)}
            for (m, f, n) in mean.index]


def _num(v):
    v = float(v)
    return int(v) if v.is_integer() else v


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=Path("results/celeba"))
    p.add_argument("--out", type=Path, default=Path("docs/assets/celeba_data.json"))
    args = p.parse_args()
    res = args.results_dir if args.results_dir.is_absolute() else ROOT / args.results_dir
    out = args.out if args.out.is_absolute() else ROOT / args.out

    data = {}
    for key, sub in VARIANTS.items():
        data[key] = {}
        for sweep, (xcol, fcol) in SWEEPS.items():
            df = pd.read_parquet(res / sub / f"{sweep}.parquet")
            data[key][sweep] = aggregate(df, xcol, fcol)
            methods = sorted({r["method"] for r in data[key][sweep]})
            print(f"{key:12s} {sweep:13s} {len(data[key][sweep]):4d} rows, "
                  f"{len(methods)} methods")

    key, sub = R0
    df = pd.read_parquet(res / sub / "n_sweep.parquet")
    data[key] = {"n_sweep": aggregate_r0(df)}
    nex = df[df["method"] == "NEXIS-v2"]
    print(f"{key:12s} n_sweep       {len(data[key]['n_sweep']):4d} rows; NEXIS share of runs "
          f"with any false discovery, pooled over {len(nex)} runs: "
          f"{(nex['n_selected'] > 0).mean():.3f}")

    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
