#!/usr/bin/env python3
"""Aggregate the rho x terminal-filter ablation (ablation_rho_filter.py) into summary.md.

Precision/recall/IoU: mean over seeds within a config (tree, sweep, fixed, param), then
macro-average over configs (std across configs in parentheses), as for the all6 numbers.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "results/celeba/ablation_rho_filter"
REF = "rho05_filter"
ORDER = ["rho05", "rho05_filter", "rho0", "rho0_filter", "fwd_rho0", "fwd_rho0_filter",
         "scr_rho05", "scr_rho05_filter", "scr_rho0", "scr_rho0_filter"]
CFG = ["tree", "sweep", "fixed", "param"]
TREES = ["k20/sae", "k20/sae_precode", "k5/sae"]

files = sorted(D.glob("rf_*.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df["uncertifiable"] = df["uncertifiable"].astype(bool)
variants = [v for v in ORDER if v in set(df.variant)]
out = [f"# rho x terminal filter ablation (CelebA grid)\n",
       f"{len(files)} files, {df.groupby(CFG).ngroups} configs, "
       f"{df[df.variant == 'rho05'].shape[0]} runs per variant.\n"]


def fmt(m, s):
    return f"{m:.3f} ({s:.3f})"


def macro(sub):
    c = sub.groupby(CFG + ["variant"])[["precision", "recall", "iou"]].mean().reset_index()
    return c.groupby("variant")[["precision", "recall", "iou"]].agg(["mean", "std"])


def table(sub, title):
    a = macro(sub)
    lines = [f"\n## {title}\n", "| variant | precision | recall | IoU |", "|---|---|---|---|"]
    for v in variants:
        if v in a.index:
            r = a.loc[v]
            lines.append(f"| {v} | {fmt(*r['precision'])} | {fmt(*r['recall'])} | "
                         f"{fmt(*r['iou'])} |")
    return lines


out += table(df, "Overall (macro over configs; std across configs in parentheses)")

# Per-tree IoU
lines = ["\n## IoU per tree (macro over that tree's configs)\n",
         "| variant | " + " | ".join(TREES) + " |", "|---" * (len(TREES) + 1) + "|"]
per = {t: macro(df[df.tree == t]) for t in TREES}
for v in variants:
    lines.append(f"| {v} | " + " | ".join(f"{per[t].loc[v, ('iou', 'mean')]:.3f}"
                                         for t in TREES) + " |")
out += lines

# Run-level comparison vs REF
key = CFG + ["seed"]
w = df.pivot_table(index=key, columns="variant", values="iou")
lines = [f"\n## Runs improved / unchanged / degraded in IoU vs {REF}\n",
         "| variant | improved | unchanged | degraded |", "|---|---|---|---|"]
for v in variants:
    if v == REF:
        continue
    d = w[v] - w[REF]
    lines.append(f"| {v} | {(d > 1e-12).sum()} | {(d.abs() <= 1e-12).sum()} | "
                 f"{(d < -1e-12).sum()} |")
out += lines

# Size and cost
lines = ["\n## Selection size and filter cost\n",
         "filter_tests counts the p-values computed by the built-in terminal filter "
         "(nexis(terminal_filter=True)); it tries subsets largest first and drops j at "
         "its first failing subset, so it is below the full |S~| 2^(|S~|-1) enumeration. "
         "sec is the wall time of the whole nexis call (selection + filter).\n",
         "| variant | median/max \\|S~\\| | median/max \\|S_hat\\| | median/max tests | "
         "mean/max sec | uncertifiable |", "|---|---|---|---|---|---|"]
for v in variants:
    s = df[df.variant == v]
    lines.append(f"| {v} | {s.size_pre.median():g} / {s.size_pre.max()} | "
                 f"{s.size_post.median():g} / {s.size_post.max()} | "
                 f"{s.filter_tests.median():g} / {s.filter_tests.max()} | "
                 f"{s.nexis_filter_sec.mean():.2f} / {s.nexis_filter_sec.max():.1f} | "
                 f"{int(s.uncertifiable.sum())} |")
out += lines

# Regime breakdown: configs where rho05 recall (mean over seeds) >= 0.95 vs not
rec = (df[df.variant == "rho05"].groupby(CFG).recall.mean() >= 0.95).rename("high")
dfr = df.merge(rec.reset_index(), on=CFG)
nh, nl = int(rec.sum()), int((~rec).sum())
out += table(dfr[dfr.high], f"High-power configs (rho05 mean recall >= 0.95; {nh} configs)")
out += table(dfr[~dfr.high], f"Low-power configs (rho05 mean recall < 0.95; {nl} configs)")

# Run-level regime split of rho0_filter vs rho05_filter
lines = ["\n## rho0_filter vs rho05_filter by regime (run-level IoU)\n",
         "| regime | improved | unchanged | degraded | mean dIoU |", "|---|---|---|---|---|"]
wr = w.reset_index().merge(rec.reset_index(), on=CFG)
for name, m in [("high-power", wr.high), ("low-power", ~wr.high)]:
    d = (wr.loc[m, "rho0_filter"] - wr.loc[m, REF])
    lines.append(f"| {name} | {(d > 1e-12).sum()} | {(d.abs() <= 1e-12).sum()} | "
                 f"{(d < -1e-12).sum()} | {d.mean():+.4f} |")
out += lines

# Per sweep IoU for the key variants
lines = ["\n## IoU per sweep (macro over trees and params)\n"]
sw = df.assign(sweep_id=df.sweep + "@" + df.fixed.map(lambda x: f"{x:g}"))
c = sw.groupby(CFG + ["sweep_id", "variant"]).iou.mean().reset_index()
p = c.groupby(["sweep_id", "variant"]).iou.mean().unstack()
ids = list(p.index)
lines += ["| variant | " + " | ".join(ids) + " |", "|---" * (len(ids) + 1) + "|"]
for v in variants:
    lines.append(f"| {v} | " + " | ".join(f"{p.loc[i, v]:.3f}" for i in ids) + " |")
out += lines

text = "\n".join(out) + "\n"
(D / "summary.md").write_text(text)
print(text)
