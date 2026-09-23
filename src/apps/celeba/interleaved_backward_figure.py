#!/usr/bin/env python3
"""Figure for the interleaved-backward experiment, in the layout of the paper's
method_backward.pdf (src/apps/celeba/figure_appendix.py): 4 rows x 3 columns,

  row 0: n sweep at eta = 5      row 1: n sweep at eta = 2
  row 2: eta sweep at n = 2000   row 3: eta sweep at n = 500
  columns: Precision | Recall | IoU, mean +- 1.96 SE over seeds,

comparing forward only (f), forward + backward (fb, the default) and forward +
interleaved backward + backward (fib).  Self-contained (no import of the paper's
figure code, which is being edited separately); forward only and the default keep the
colours of src/apps/celeba/visualize.py, the interleaved arm is orange for contrast.

Usage: interleaved_backward_figure.py <tag> [--with-fi]
Writes results/celeba/interleaved_backward/<tag>/method_interleaved_backward.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
tag = sys.argv[1]
with_fi = "--with-fi" in sys.argv
D = ROOT / "results/celeba/interleaved_backward" / tag

plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "axes.titlesize": 13,
                     "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 12})

ARMS = {
    "f":   dict(color="#2ca02c", lw=1.5, marker="o", ms=3, ls="--", label="forward only"),
    "fb":  dict(color="#08519c", lw=2.5, marker="o", ms=3,
                label="forward + backward (default)"),
    "fib": dict(color="#d95f02", lw=2.0, marker="^", ms=4,
                label="forward + interleaved backward + backward"),
}
if with_fi:
    ARMS["fi"] = dict(color="#00441b", lw=1.5, marker="s", ms=3, ls="--",
                      label="forward + interleaved backward")
LBL = {"precision": "Precision", "recall": "Recall", "iou": "IoU"}


def load(sweep, fixed):
    return pd.read_csv(D / f"{sweep}_{fixed:g}.csv")


rows = [(load("n", 5), "n", r"Sample size $n$", r"varying $n$,  $\eta=5$ fixed"),
        (load("n", 2), "n", r"Sample size $n$", r"varying $n$,  $\eta=2$ fixed"),
        (load("effect", 2000), "effect_scale", r"Effect size $\eta$",
         r"varying $\eta$,  $n=2000$ fixed"),
        (load("effect", 500), "effect_scale", r"Effect size $\eta$",
         r"varying $\eta$,  $n=500$ fixed")]

fig, axes = plt.subplots(4, 3, figsize=(13, 15))
for r, (df, xcol, xlabel, _) in enumerate(rows):
    for c, metric in enumerate(["precision", "recall", "iou"]):
        ax = axes[r, c]
        for arm, st in ARMS.items():
            g = df.groupby(xcol)[f"{arm}_{metric}"]
            mu, se = g.mean(), g.sem()
            ax.plot(mu.index.values, mu.values, **st)
            ax.fill_between(mu.index.values, (mu - 1.96 * se).values,
                            (mu + 1.96 * se).values, color=st["color"], alpha=0.15)
        if xcol == "n":
            ax.set_xscale("log")
        ax.set_xlim(left=df[xcol].min() * 0.95)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(LBL[metric])
        ax.grid(True, alpha=0.25)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.tight_layout(rect=[0, 0.05, 1, 1.0], h_pad=2.5)
for r, (*_, title) in enumerate(rows):
    x = (axes[r, 0].get_position().x0 + axes[r, 2].get_position().x1) / 2
    y = max(a.get_position().y1 for a in axes[r]) + 0.004
    fig.text(x, y, title, ha="center", va="bottom", fontsize=13)
fig.canvas.draw()
rend = fig.canvas.get_renderer()
h = fig.get_window_extent(rend).height
ymin = min(a.get_tightbbox(rend).y0 / h for a in axes.flat)
fig.legend(handles, labels, loc="upper center", ncol=2,
           bbox_to_anchor=(0.5, ymin + 0.01), frameon=False, fontsize=12)
out = D / ("method_interleaved_backward" + ("_fi" if with_fi else "") + ".pdf")
fig.savefig(out, bbox_inches="tight")
print("Saved ->", out)
