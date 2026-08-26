"""
Figure: behaviour of NEXIS under a controlled violation of Principal Alignment.

Reads results/celeba/experiment/violation/violation_sweep.parquet (produced by
violation_sweep.py) and draws one row per sample size, three panels:

  (a) Sufficiency   — population R^2 of tau on Z^S_hat, relative to the minimal
                      sufficient set of the perturbed dictionary.  This is the
                      property the causal claim of Eq. 12 rests on.
  (b) Recall        — against the minimal sufficient set {j1, j_new, j2}: does the
                      procedure find the coordinates that jointly encode W1?
  (c) Selection size and precision against the one-to-one target {j1, j2}: the
                      resolution that a split concept costs.

x-axis is the *measured* leakage ratio eps_hat, not the nominal split fraction, so
the panels can be read against the threshold rho of Proposition 1.

Colours: NEXIS keeps the paper's #08519c; the rho=0 variant is drawn in a more
saturated blue than the rho-ablation ramp (#2b8cbe rather than #c6dbef) because
here it is a primary comparison rather than one step of a four-step ramp, and the
palette is validated for CVD separation and contrast at these three values.
Line style carries the method too, so identity is never colour-alone.

Usage
-----
python src/apps/celeba/figure_violation.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
})

STYLES = {
    "NEXIS":                   dict(color="#08519c", ls="-",  lw=2.4, marker="o", ms=4,
                                    label=r"NEXIS ($\rho=0.5$)"),
    "NEXIS (rho=0)":           dict(color="#2b8cbe", ls="--", lw=1.8, marker="s", ms=4,
                                    label=r"NEXIS ($\rho=0$)"),
    "Marginal Testing (FWER)": dict(color="#ff7f0e", ls=":",  lw=1.8, marker="^", ms=4,
                                    label="Marginal Testing (FWER)"),
}
GRID = dict(color="#e6e6e6", lw=0.7, zorder=0)


def _band(ax, df, xcol, ycol, method, style):
    sub = df[df["method"] == method].groupby(xcol)[ycol]
    x = np.asarray(sub.mean().index, dtype=float)
    mu = sub.mean().values
    se = sub.sem().values
    ax.plot(x, mu, zorder=3, **style)
    ax.fill_between(x, mu - 1.96 * se, mu + 1.96 * se,
                    color=style["color"], alpha=0.15, lw=0, zorder=2)
    return x, mu


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path,
                   default="results/celeba/experiment/violation")
    p.add_argument("--out", type=Path,
                   default="results/celeba/appendix/violation.pdf")
    p.add_argument("--rho", type=float, default=0.5)
    args = p.parse_args()

    in_dir = ROOT / args.in_dir if not args.in_dir.is_absolute() else args.in_dir
    out = ROOT / args.out if not args.out.is_absolute() else args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    # violation_sweep.py writes one file per sample-size arm (--tag)
    files = sorted(in_dir.glob("violation_sweep*.parquet"))
    if not files:
        raise SystemExit(f"no violation_sweep*.parquet in {in_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"read {len(df)} rows from {len(files)} file(s): "
          f"{[f.name for f in files]}")
    ns = sorted(df["n"].unique())

    # mathtext (no \bm): use \mathbf for the vector Z
    panels = [
        ("suff_ratio",      r"Sufficiency: $R^2(\tau;\mathbf{Z}^{\widehat{S}})\,/\,R^2(\tau;\mathbf{Z}^{S^\star_{split}})$"),
        ("recall_split",    r"Recall vs $S^\star_{split}=\{j_1,j_{new},j_2\}$"),
        ("precision_split", r"Precision vs $S^\star_{split}$"),
        ("precision_orig",  r"Precision vs one-to-one $\{j_1,j_2\}$"),
        ("n_selected",      r"$|\widehat{S}_n|$"),
    ]

    fig, axes = plt.subplots(len(ns), len(panels),
                             figsize=(4.1 * len(panels), 3.1 * len(ns)),
                             squeeze=False)

    for r, n in enumerate(ns):
        dn = df[df["n"] == n]
        for c, (ycol, title) in enumerate(panels):
            ax = axes[r][c]
            ax.set_axisbelow(True)
            ax.grid(True, **GRID)
            # rho threshold of Proposition 1
            ax.axvline(args.rho, color="#666666", lw=1.0, ls=(0, (4, 3)), zorder=1)
            for method, style in STYLES.items():
                if method in dn["method"].values:
                    _band(ax, dn, "eps_hat", ycol, method, style)
            ax.set_xlabel(r"measured leakage ratio $\widehat{\varepsilon}$")
            if r == 0:
                ax.set_title(title, fontsize=10.5)
            if c == 0:
                ax.set_ylabel(f"$n = {n}$", fontsize=12)
            if ycol == "n_selected":
                ax.set_yscale("log")
            else:
                ax.set_ylim(-0.04, 1.12)
            # label the rho threshold once per row
            if c == 0:
                ax.annotate(rf"$\rho={args.rho}$", xy=(args.rho, 0.04),
                            xytext=(4, 0), textcoords="offset points",
                            fontsize=9, color="#666666", ha="left", va="bottom")

    # Direct labels go on the one-to-one precision panel, the only one where all
    # three series separate at the right edge; on the sufficiency panel they all
    # sit at ~1.0 and the labels would collide.  These labels (plus the CSV table
    # view written below) supply the relief the low-contrast orange requires.
    lab_col = next(i for i, (y, _) in enumerate(panels) if y == "precision_orig")
    ax = axes[0][lab_col]
    dn = df[df["n"] == ns[0]]
    for method, style in STYLES.items():
        sub = dn[dn["method"] == method].groupby("eps_hat")["precision_orig"].mean()
        if len(sub):
            ax.annotate(style["label"], xy=(sub.index[-1], sub.values[-1]),
                        xytext=(5, 0), textcoords="offset points",
                        fontsize=8.5, color=style["color"], va="center",
                        annotation_clip=False)

    handles = [plt.Line2D([], [], **{k: v for k, v in s.items() if k != "label"},
                          label=s["label"]) for s in STYLES.values()]
    fig.legend(handles=handles, loc="lower center", ncol=len(STYLES),
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")

    # table view (the accessibility fallback, and what goes in the rebuttal)
    tab = (df.groupby(["n", "method", "eps"])
             [["eps_hat", "suff_ratio", "recall_split", "precision_split",
               "recall_orig", "precision_orig", "n_selected"]]
             .mean().round(3))
    csv = out.with_suffix(".csv")
    tab.to_csv(csv)
    print(f"wrote {csv}\n")
    print(tab.to_string())


if __name__ == "__main__":
    main()
