#!/usr/bin/env python3
"""
Export the CelebA sweep curves shown on the project website (docs/index.html).

Reads the terminal-backward-step runs behind paper Figure 3 and the Appendix C
ablations (results/celeba/experiment_v2/<k>/<view>/{n,effect}_sweep.parquet, see
scripts/celeba/submit_experiment_v2.sh) and writes, per dictionary and sweep, the
mean and standard error over seeds of precision, recall and IoU for every method.
This is the same aggregation as src/apps/celeba/visualize.py::plot_sweep (mean,
pandas .sem(); the site draws the +-1.96 SE band).

Method names drop the "-v2" tag, so the paper default is "NEXIS" and its ablations
are "NEXIS (test=...)", "NEXIS (adjust=...)", "NEXIS (rho=...)",
"NEXIS (terminal=False)", "NEXIS (interleaved=True)", ...

Output: docs/assets/celeba_data.json
  {"k20_sae" | "k20_precode" | "k5_sae":
      {"n_sweep":      [{method, n, fixed_effect, <metric>_mean, <metric>_se}, ...],
       "effect_sweep": [{method, effect_scale, fixed_n, <metric>_mean, <metric>_se}, ...]}}

Usage
-----
    python src/apps/celeba/export_website_data.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent

# website key -> results subdirectory (the dictionaries reported in the paper)
VARIANTS = {
    "k20_sae":     "k20/sae",           # main setting: k = 20, sparse codes Z
    "k20_precode": "k20/sae_precode",   # k = 20, dense pre-activations Z_pre
    "k5_sae":      "k5/sae",            # k = 5, sparse codes Z
}
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


def _num(v):
    v = float(v)
    return int(v) if v.is_integer() else v


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment-dir", type=Path, default=Path("results/celeba/experiment_v2"))
    p.add_argument("--out", type=Path, default=Path("docs/assets/celeba_data.json"))
    args = p.parse_args()
    exp = args.experiment_dir if args.experiment_dir.is_absolute() else ROOT / args.experiment_dir
    out = args.out if args.out.is_absolute() else ROOT / args.out

    data = {}
    for key, sub in VARIANTS.items():
        data[key] = {}
        for sweep, (xcol, fcol) in SWEEPS.items():
            df = pd.read_parquet(exp / sub / f"{sweep}.parquet")
            data[key][sweep] = aggregate(df, xcol, fcol)
            methods = sorted({r["method"] for r in data[key][sweep]})
            print(f"{key:12s} {sweep:13s} {len(data[key][sweep]):4d} rows, "
                  f"{len(methods)} methods")

    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
