#!/usr/bin/env python3
"""
Main figure for the CelebA semi-synthetic experiment (SAE k=20).

4 panels in a single row:
  [Precision η=5 / n] [Recall η=5 / n]  ‹gap›  [Precision n=2000 / η] [Recall n=2000 / η]

Group titles "η=5" and "n=2000" are placed centred above each pair.
A single horizontal legend sits above all panels.

Usage
-----
    python src/apps/celeba/figure_main.py
    python src/apps/celeba/figure_main.py --out-path results/celeba/experiment/k20/sae/figure_main.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent

from apps.celeba.visualize import plot_sweep, MAIN_METHODS, _METRIC_LABEL

MAIN_METHODS_PAPER = {**MAIN_METHODS, "NEXIS": "NEXIS (ours)"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path,
                   default="results/celeba/experiment/k20/sae")
    p.add_argument("--out-path", type=Path,
                   default="results/celeba/experiment/k20/sae/figure_main.pdf")
    p.add_argument("--fixed-effect", type=float, default=5.0,
                   help="Fixed η for the left pair (default: 5.0)")
    p.add_argument("--fixed-n", type=int, default=2000,
                   help="Fixed n for the right pair (default: 2000)")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = ROOT / args.data_dir if not args.data_dir.is_absolute() else args.data_dir
    out_path = ROOT / args.out_path if not args.out_path.is_absolute() else args.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_n      = pd.read_parquet(data_dir / "n_sweep.parquet")
    df_effect = pd.read_parquet(data_dir / "effect_sweep.parquet")

    df_left  = df_n[df_n["fixed_effect"] == args.fixed_effect]
    df_right = df_effect[df_effect["fixed_n"] == args.fixed_n]

    if df_left.empty:
        raise ValueError(
            f"No rows with fixed_effect={args.fixed_effect} in n_sweep. "
            f"Available: {sorted(df_n['fixed_effect'].unique())}"
        )
    if df_right.empty:
        raise ValueError(
            f"No rows with fixed_n={args.fixed_n} in effect_sweep. "
            f"Available: {sorted(df_effect['fixed_n'].unique())}"
        )

    plt.rcParams.update({
        "font.size":        15,
        "axes.labelsize":   15,
        "axes.titlesize":   15,
        "xtick.labelsize":  14,
        "ytick.labelsize":  14,
        "legend.fontsize":  14,
    })
    label_size = 15

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))

    # Left pair — n sweep at fixed η  (precision | recall)
    plot_sweep(df_left,  "n",            "precision", xlabel=r"Sample size $n$",
               ax=axes[0], methods=MAIN_METHODS_PAPER)
    plot_sweep(df_left,  "n",            "recall",    xlabel=r"Sample size $n$",
               ax=axes[1], methods=MAIN_METHODS_PAPER)

    # Right pair — effect sweep at fixed n  (precision | recall)
    plot_sweep(df_right, "effect_scale", "precision", xlabel=r"Effect size $\eta$",
               ax=axes[2], methods=MAIN_METHODS_PAPER)
    plot_sweep(df_right, "effect_scale", "recall",    xlabel=r"Effect size $\eta$",
               ax=axes[3], methods=MAIN_METHODS_PAPER)

    # Collect legend handles before removing per-panel legends
    handles, labels = axes[0].get_legend_handles_labels()

    # Remove per-panel titles and legends (we use group titles + shared legend instead)
    for ax in axes:
        ax.set_title("")
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    # Reserve top margin for group titles, bottom margin for legend
    fig.tight_layout(rect=[0, 0.10, 1, 0.92])

    # Shift right pair to create extra centre gap
    extra_gap = 0.04
    for ax in axes[2:]:
        pos = ax.get_position()
        ax.set_position([pos.x0 + extra_gap, pos.y0, pos.width, pos.height])

    # Shared horizontal legend just below the x-axis titles
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_h = fig.get_window_extent(renderer).height
    tight_ymin = min(
        ax.get_tightbbox(renderer).y0 / fig_h for ax in axes
    )
    fig.legend(handles, labels,
               loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, tight_ymin + 0.01),
               frameon=False, fontsize=14)

    # Group titles centred above each pair
    def _pair_xcenter(ax_l, ax_r):
        return (ax_l.get_position().x0 + ax_r.get_position().x1) / 2

    ytitle = 0.97
    eta_str = (str(int(args.fixed_effect))
               if args.fixed_effect == int(args.fixed_effect)
               else str(args.fixed_effect))
    fig.text(_pair_xcenter(axes[0], axes[1]), ytitle,
             rf"$\eta={eta_str}$",
             ha="center", va="top", fontsize=label_size)
    fig.text(_pair_xcenter(axes[2], axes[3]), ytitle,
             rf"$n={args.fixed_n}$",
             ha="center", va="top", fontsize=label_size)

    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
