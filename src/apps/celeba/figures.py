#!/usr/bin/env python3
"""CelebA figures and tables of the paper (Section 3, Appendix C), one subcommand each.

  main        Figure 3: precision and recall against n (eta = 5) and against eta
              (n = 2000), four panels in one row (figure_main.pdf)
  appendix    the 4-row x 3-column appendix figures: dgp, model_k5, model_precode,
              method_{test,adjust,rho,backward}.pdf
  dgp-extra   reproducibility on the replica SAE and the r = 0, 1, 3 DGPs:
              replica_k20.pdf, dgp_r1.pdf, dgp_r3.pdf, dgp_r0.pdf, dgp_r0_table.tex,
              dgp_extra.md
  test-story  comparison of the CATE-equivalence tests on the linear and the U-shape
              DGP: test_story.{pdf,md,tex}

The paper's versions (README.md, Section 7):

    python src/apps/celeba/figures.py main --data-dir results/celeba/experiment_v2/k20/sae \\
        --out-path results/celeba/figures_v2/figure_main.pdf --nexis-key NEXIS-v2
    python src/apps/celeba/figures.py appendix --experiment-dir results/celeba/experiment_v2 \\
        --out-dir results/celeba/figures_v2
    python src/apps/celeba/figures.py dgp-extra
    python src/apps/celeba/figures.py test-story

Every subcommand reads the sweep parquets of run_experiment.py / run_experiment_dgp.py
(untracked, under results/celeba/) and takes --help.  The plotting helpers (method
styles, plot_sweep, make_12panel) are shared with alignment.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results/celeba"


# ===========================================================================
# Method styles and the sweep plot
# ===========================================================================

METHOD_STYLES: dict = {
    # Baselines (red/orange/pink family)
    'Marginal Testing':        dict(color='#d62728', lw=1.5, marker='o', ms=3, label='Marginal Testing'),
    'Marginal Testing (FWER)': dict(color='#ff7f0e', lw=1.5, marker='o', ms=3, label='Marginal Testing (FWER)'),
    'Marginal Testing (FDR)':  dict(color='#e377c2', lw=1.5, marker='o', ms=3, label='Marginal Testing (FDR)'),
    # NEXIS default (bold blue)
    'NEXIS':                   dict(color='#08519c', lw=2.5, marker='o', ms=3, label='NEXIS'),
    # test ablation (purple family)
    'NEXIS (test=GCM: quadratic)': dict(color='#9467bd', lw=1.5, marker='o', ms=3, label='NEXIS (test=GCM: quadratic)'),
    'NEXIS (test=GCM: lgbm)':      dict(color='#5c3493', lw=2.0, marker='o', ms=3, label='NEXIS (test=GCM: lgbm)'),
    'NEXIS (test=PCM: quadratic)': dict(color='#e6a532', lw=1.5, marker='s', ms=3, label='NEXIS (test=PCM: quadratic)'),
    'NEXIS (test=PCM: lgbm)':      dict(color='#a86a12', lw=2.0, marker='s', ms=3, label='NEXIS (test=PCM: lgbm)'),
    # adjust ablation (teal family)
    'NEXIS (adjust=None)':     dict(color='#41b6c4', lw=1.5, marker='o', ms=3, label='NEXIS (adjust=None)'),
    'NEXIS (adjust=FDR)':      dict(color='#1d91c0', lw=1.5, marker='o', ms=3, label='NEXIS (adjust=FDR)'),
    # rho ablation (blue family, lighter = smaller rho)
    'NEXIS (rho=0)':           dict(color='#c6dbef', lw=1.5, marker='o', ms=3, label='NEXIS (ρ=0)'),
    'NEXIS (rho=0.2)':         dict(color='#9ecae1', lw=1.5, marker='o', ms=3, label='NEXIS (ρ=0.2)'),
    'NEXIS (rho=0.8)':         dict(color='#6baed6', lw=1.5, marker='o', ms=3, label='NEXIS (ρ=0.8)'),
    # backward ablation (green)
    'NEXIS (backward=False)':  dict(color='#2ca02c', lw=1.5, marker='o', ms=3, label='NEXIS (backward=False)'),
}

# Methods shown in the main comparison figure (NEXIS vs all marginal baselines)
MAIN_METHODS: dict[str, str] = {
    'Marginal Testing':        'Marginal Testing',
    'Marginal Testing (FDR)':  'Marginal Testing (FDR)',
    'Marginal Testing (FWER)': 'Marginal Testing (FWER)',
    'NEXIS':                   'NEXIS',
}

# ── Terminal-backward-step default ("NEXIS-v2", see experiment.V2_METHODS) ────
# The v2 default and baselines reuse the published styles, so the v2 figures are
# drop-in replacements; the ablation variants are restyled below for legibility.
_V2_STYLE_FROM = {
    'NEXIS-v2':                       'NEXIS',
    'NEXIS-v2 (test=GCM: quadratic)': 'NEXIS (test=GCM: quadratic)',
    'NEXIS-v2 (test=GCM: lgbm)':      'NEXIS (test=GCM: lgbm)',
    'NEXIS-v2 (test=PCM: quadratic)': 'NEXIS (test=PCM: quadratic)',
    'NEXIS-v2 (test=PCM: lgbm)':      'NEXIS (test=PCM: lgbm)',
    'NEXIS-v2 (adjust=None)':         'NEXIS (adjust=None)',
    'NEXIS-v2 (adjust=FDR)':          'NEXIS (adjust=FDR)',
    'NEXIS-v2 (rho=0)':               'NEXIS (rho=0)',
    'NEXIS-v2 (rho=0.2)':             'NEXIS (rho=0.2)',
    'NEXIS-v2 (rho=0.8)':             'NEXIS (rho=0.8)',
    'NEXIS-v2 (terminal=False)':      'NEXIS (backward=False)',
}
METHOD_STYLES.update({k: {**METHOD_STYLES[v], 'label': k}
                      for k, v in _V2_STYLE_FROM.items()})

# v2 ablation lines: Okabe-Ito colours without its blue, so the default keeps the
# published blue.  Lines that overlap another one (identical or nearly identical
# results) are dashed and drawn on top, so the line underneath shows in the gaps.
_OI_ORANGE, _OI_GREEN, _OI_VERMILLION, _OI_PURPLE = '#E69F00', '#009E73', '#D55E00', '#CC79A7'
_ON_TOP = dict(ls='--', zorder=3)
for _k, _s in {
    'NEXIS-v2 (test=GCM: quadratic)': dict(color=_OI_PURPLE,     lw=1.5, marker='o', ms=3, **_ON_TOP),
    'NEXIS-v2 (test=GCM: lgbm)':      dict(color=_OI_VERMILLION, lw=2.0, marker='o', ms=3),
    'NEXIS-v2 (test=PCM: quadratic)': dict(color=_OI_ORANGE,     lw=1.5, marker='s', ms=3, **_ON_TOP),
    'NEXIS-v2 (test=PCM: lgbm)':      dict(color=_OI_GREEN,      lw=2.0, marker='s', ms=3),
    'NEXIS-v2 (adjust=None)':         dict(color=_OI_VERMILLION, lw=1.5, marker='o', ms=3),
    'NEXIS-v2 (adjust=FDR)':          dict(color=_OI_GREEN,      lw=1.5, marker='o', ms=3, **_ON_TOP),
    'NEXIS-v2 (rho=0)':               dict(color=_OI_ORANGE,     lw=1.5, marker='o', ms=3, **_ON_TOP),
    'NEXIS-v2 (rho=0.2)':             dict(color=_OI_GREEN,      lw=1.5, marker='o', ms=3),
    'NEXIS-v2 (rho=0.8)':             dict(color=_OI_VERMILLION, lw=1.5, marker='o', ms=3),
    # Backward-step ablation: no markers; the interleaved variants are dashed.
    'NEXIS-v2 (terminal=False)':      dict(color=_OI_ORANGE,     lw=2.0),
    'NEXIS-v2 (interleaved=True, terminal=False)':
                                      dict(color=_OI_VERMILLION, lw=2.0, **_ON_TOP),
    'NEXIS-v2 (interleaved=True)':    dict(color=_OI_GREEN,      lw=2.0, **_ON_TOP),
}.items():
    METHOD_STYLES[_k] = {**_s, 'label': _k}


MAIN_METHODS_V2: dict[str, str] = {
    **{k: v for k, v in MAIN_METHODS.items() if k != 'NEXIS'},
    'NEXIS-v2': 'NEXIS',
}

# One entry per ablation dimension: the v2 default + the ablated variants.  Labels are
# short (no "NEXIS" prefix).  "backward step" is the terminal step, "interleaved backward
# step" the published per-round one.
ABLATION_GROUPS_V2: dict[str, dict] = {
    'test': {
        'title': 'Test statistic',
        'legend_ncol': 5,
        'methods': {
            'NEXIS-v2':                       'linear (default)',
            'NEXIS-v2 (test=GCM: quadratic)': 'GCM: quadratic',
            'NEXIS-v2 (test=GCM: lgbm)':      'GCM: LightGBM',
            'NEXIS-v2 (test=PCM: quadratic)': 'PCM: quadratic',
            'NEXIS-v2 (test=PCM: lgbm)':      'PCM: LightGBM',
        },
    },
    'adjust': {
        'title': 'Adjustment',
        'methods': {
            'NEXIS-v2 (adjust=None)': 'None',
            'NEXIS-v2 (adjust=FDR)':  'FDR',
            'NEXIS-v2':               'FWER (default)',
        },
    },
    'rho': {
        'title': 'ρ threshold',
        'methods': {
            'NEXIS-v2 (rho=0)':   '0',
            'NEXIS-v2 (rho=0.2)': '0.2',
            'NEXIS-v2':           '0.5 (default)',
            'NEXIS-v2 (rho=0.8)': '0.8',
        },
    },
    'backward': {
        'title': 'Backward step',
        'legend_ncol': 2,
        'methods': {
            'NEXIS-v2 (terminal=False)':                   'forward',
            'NEXIS-v2 (interleaved=True, terminal=False)': 'forward + interleaved backward',
            'NEXIS-v2':                                    'forward + terminal backward (default)',
            'NEXIS-v2 (interleaved=True)':                 'forward + interleaved and terminal backward',
        },
        # per-figure style overrides: the default line without markers
        'styles': {'NEXIS-v2': dict(marker=None)},
    },
}


def _add_type1_region(ax, df, xcol: str, metric: str) -> None:
    if xcol == 'effect_scale' and metric in ('recall', 'iou', 'precision'):
        ax.axvspan(-0.05, 0.05, color='gray', alpha=0.10)


_METRIC_LABEL = {'iou': 'IoU', 'recall': 'Recall', 'precision': 'Precision'}


def plot_sweep(
    df: pd.DataFrame,
    xcol: str,
    metric: str,
    xlabel: str,
    title: str | None = None,
    ax=None,
    methods: dict[str, str] | None = None,
    legend_title: str | None = None,
    styles: dict[str, dict] | None = None,
) -> plt.Axes:
    """Mean ± 1.96 SE curves for each method vs. a sweep parameter.

    methods: optional dict {method_key: display_label}. If None, all METHOD_STYLES
             are shown with their default labels.
    styles:  optional {method_key: style overrides} applied on top of METHOD_STYLES.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    log_x = (xcol == 'n')
    items = (methods.items() if methods is not None
             else {m: s['label'] for m, s in METHOD_STYLES.items()}.items())
    for method, label in items:
        if method not in METHOD_STYLES:
            continue
        style = {**METHOD_STYLES[method], **(styles or {}).get(method, {}), 'label': label}
        sub = df[df['method'] == method].groupby(xcol)[metric]
        mu, se = sub.mean(), sub.sem()
        if mu.empty:
            continue
        ax.plot(mu.index.values, mu.values, **style)
        ax.fill_between(mu.index.values,
                        (mu - 1.96 * se).values, (mu + 1.96 * se).values,
                        color=style['color'], alpha=0.15)
    _add_type1_region(ax, df, xcol, metric)
    if log_x:
        ax.set_xscale('log')
    ax.set_xlim(left=df[xcol].min() * 0.95)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(_METRIC_LABEL.get(metric, metric.capitalize()))
    if title is not None:
        ax.set_title(title)
    ax.legend(fontsize=9, frameon=False, title=legend_title)
    ax.grid(True, alpha=0.25)
    return ax


# ===========================================================================
# Global style of the appendix figures (`main` sets its own larger fonts)
# ===========================================================================

plt.rcParams.update({
    "font.size":       13,
    "axes.labelsize":  13,
    "axes.titlesize":  13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})
LABEL_SIZE = 13

ETA_MAIN, ETA_ALT = 5.0, 2.0
N_MAIN,   N_ALT   = 2000, 500


def _abs(p: Path) -> Path:
    return p if p.is_absolute() else ROOT / p


# ===========================================================================
# main: Figure 3
# ===========================================================================

MAIN_METHODS_PAPER = {**MAIN_METHODS, "NEXIS": "NEXIS (ours)"}


def cmd_main(argv: list[str]) -> None:
    """Main figure for the CelebA semi-synthetic experiment (SAE k=20).

    4 panels in a single row:
      [Precision η=5 / n] [Recall η=5 / n]  ‹gap›  [Precision n=2000 / η] [Recall n=2000 / η]
    Group titles "η=5" and "n=2000" are placed centred above each pair; a single
    horizontal legend sits below all panels.
    """
    p = argparse.ArgumentParser(prog="figures.py main", description=cmd_main.__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path,
                   default="results/celeba/experiment/k20/sae")
    p.add_argument("--out-path", type=Path,
                   default="results/celeba/experiment/k20/sae/figure_main.pdf")
    p.add_argument("--fixed-effect", type=float, default=5.0,
                   help="Fixed η for the left pair (default: 5.0)")
    p.add_argument("--fixed-n", type=int, default=2000,
                   help="Fixed n for the right pair (default: 2000)")
    p.add_argument("--nexis-key", default="NEXIS",
                   help="Method key drawn as 'NEXIS (ours)' (default: NEXIS; "
                        "NEXIS-v2 for the terminal-backward-step default)")
    args = p.parse_args(argv)

    data_dir = _abs(args.data_dir)
    out_path = _abs(args.out_path)
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

    methods = {k: v for k, v in MAIN_METHODS_PAPER.items() if k != "NEXIS"}
    methods[args.nexis_key] = MAIN_METHODS_PAPER["NEXIS"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))

    # Left pair — n sweep at fixed η  (precision | recall)
    plot_sweep(df_left,  "n",            "precision", xlabel=r"Sample size $n$",
               ax=axes[0], methods=methods)
    plot_sweep(df_left,  "n",            "recall",    xlabel=r"Sample size $n$",
               ax=axes[1], methods=methods)

    # Right pair — effect sweep at fixed n  (precision | recall)
    plot_sweep(df_right, "effect_scale", "precision", xlabel=r"Effect size $\eta$",
               ax=axes[2], methods=methods)
    plot_sweep(df_right, "effect_scale", "recall",    xlabel=r"Effect size $\eta$",
               ax=axes[3], methods=methods)

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


# ===========================================================================
# appendix: the 4-row x 3-column figures
# ===========================================================================

def make_12panel(
    df_n: pd.DataFrame,       # n_sweep: must contain fixed_effect ∈ {ETA_ALT, ETA_MAIN}
    df_e: pd.DataFrame,       # effect_sweep: must contain fixed_n ∈ {N_ALT, N_MAIN}
    methods: dict[str, str],  # {method_key: display_label}
    out_path: Path,
    extra_styles: dict | None = None,
    legend_ncol: int | None = None,
    styles: dict[str, dict] | None = None,
) -> None:
    """
    4-row × 3-col figure; rows = DGP settings, cols = Precision | Recall | IoU.

      Row 0: n-sweep   @ η=ETA_MAIN  (fixed η, varying n)
      Row 1: n-sweep   @ η=ETA_ALT   (fixed η, varying n)
      Row 2: η-sweep   @ n=N_MAIN    (fixed n, varying η)
      Row 3: η-sweep   @ n=N_ALT     (fixed n, varying η)

    A row title sits above each row naming its DGP condition.
    A single shared legend is placed below all panels.
    """
    if extra_styles:
        METHOD_STYLES.update(extra_styles)

    try:
        fig, axes = plt.subplots(4, 3, figsize=(13, 15))

        row_cfg = [
            (df_n[df_n["fixed_effect"] == ETA_MAIN], "n",            r"Sample size $n$",
             rf"varying $n$,  $\eta={int(ETA_MAIN)}$ fixed"),
            (df_n[df_n["fixed_effect"] == ETA_ALT],  "n",            r"Sample size $n$",
             rf"varying $n$,  $\eta={int(ETA_ALT)}$ fixed"),
            (df_e[df_e["fixed_n"]      == N_MAIN],   "effect_scale", r"Effect size $\eta$",
             rf"varying $\eta$,  $n={N_MAIN}$ fixed"),
            (df_e[df_e["fixed_n"]      == N_ALT],    "effect_scale", r"Effect size $\eta$",
             rf"varying $\eta$,  $n={N_ALT}$ fixed"),
        ]

        for r, (sub, xcol, xlabel, _) in enumerate(row_cfg):
            for c, metric in enumerate(["precision", "recall", "iou"]):
                plot_sweep(sub, xcol, metric, xlabel=xlabel,
                           ax=axes[r, c], methods=methods, styles=styles)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        for ax in axes.flat:
            ax.set_title("")
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

        # h_pad leaves vertical room for per-row titles
        fig.tight_layout(rect=[0, 0.05, 1, 1.0], h_pad=2.5)

        # ── Row titles ────────────────────────────────────────────────────────
        def _row_xcenter(r):
            return (axes[r, 0].get_position().x0 + axes[r, 2].get_position().x1) / 2

        for r, (_, _, _, title) in enumerate(row_cfg):
            ytop = max(ax.get_position().y1 for ax in axes[r]) + 0.004
            fig.text(_row_xcenter(r), ytop, title,
                     ha="center", va="bottom", fontsize=LABEL_SIZE)

        # ── Single legend below all panels ────────────────────────────────────
        fig.canvas.draw()
        renderer  = fig.canvas.get_renderer()
        fig_h     = fig.get_window_extent(renderer).height
        tight_ymin = min(ax.get_tightbbox(renderer).y0 / fig_h for ax in axes.flat)
        n_cols_leg = legend_ncol or min(len(handles), 4)
        fig.legend(handles, labels,
                   loc="upper center", ncol=n_cols_leg,
                   bbox_to_anchor=(0.5, tight_ymin + 0.01),
                   frameon=False, fontsize=12)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out_path}")

    finally:
        if extra_styles:
            for k in extra_styles:
                METHOD_STYLES.pop(k, None)


def cmd_appendix(argv: list[str]) -> None:
    """Appendix C figures of NEXIS-v2, 4-row × 3-col layout (one row per DGP setting,
    precision | recall | IoU per row):

      dgp.pdf             main setting (k=20, Z), NEXIS against the marginal baselines
      model_k5.pdf        SAE k=5
      model_precode.pdf   k=20 pre-activations Z_pre
      method_test.pdf     test statistic ablation
      method_adjust.pdf   multiple-testing adjustment ablation
      method_rho.pdf      ρ threshold ablation
      method_backward.pdf backward step ablation
    """
    p = argparse.ArgumentParser(prog="figures.py appendix", description=cmd_appendix.__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment-dir", type=Path, default=RES / "experiment_v2",
                   help="Sweep-results root containing k{5,20}/{sae,sae_precode}/ "
                        "(default: results/celeba/experiment_v2)")
    p.add_argument("--out-dir", type=Path, default=RES / "figures_v2",
                   help="Where to write the figures (default: results/celeba/figures_v2)")
    args = p.parse_args(argv)
    exp, out = _abs(args.experiment_dir), _abs(args.out_dir)
    print(f"Reading sweeps from {exp}")

    def load(k: int, feat: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        base = exp / f"k{k}" / feat
        return (pd.read_parquet(base / "n_sweep.parquet"),
                pd.read_parquet(base / "effect_sweep.parquet"))

    for (k, feat), name in [((20, "sae"), "dgp.pdf"), ((5, "sae"), "model_k5.pdf"),
                            ((20, "sae_precode"), "model_precode.pdf")]:
        n, e = load(k, feat)
        make_12panel(n, e, MAIN_METHODS_V2, out / name)
    for key in ("test", "adjust", "rho", "backward"):
        n, e = load(20, "sae")
        grp = ABLATION_GROUPS_V2[key]
        make_12panel(n, e, grp["methods"], out / f"method_{key}.pdf",
                     legend_ncol=grp.get("legend_ncol"), styles=grp.get("styles"))
    print(f"\nAll appendix assets saved to {out}")


# ===========================================================================
# dgp-extra: replica SAE and r = 0, 1, 3
# ===========================================================================

TREES = {
    "main":    RES / "experiment_v2/k20/sae",
    "replica": RES / "experiment_v2_resample_b1/k20/sae",
    "r0":      RES / "experiment_v2_r0_fixbeta/k20/sae",
    "r1":      RES / "experiment_v2_r1/k20/sae",
    "r3":      RES / "experiment_v2_r3/k20/sae",
    # published NEXIS, for context in the reproducibility table
    "main_pub":    RES / "experiment/k20/sae",
    "replica_pub": RES / "experiment_resample_b1/k20/sae",
}
BASELINES = ["Marginal Testing", "Marginal Testing (FWER)", "Marginal Testing (FDR)"]
# The four rows of dgp.pdf: (sweep, column fixed, value fixed, x column)
ROWS = [("n", "fixed_effect", 5.0, "n"), ("n", "fixed_effect", 2.0, "n"),
        ("effect", "fixed_n", 2000, "effect_scale"), ("effect", "fixed_n", 500, "effect_scale")]
# r = 0: the DGP conditions left once gamma = 0 (panel title, sweep, column fixed, value)
R0_PANELS = [(r"no modifier ($r=0$), $\beta = (0.3, -0.2)$", "n", "fixed_effect", 1.0)]
PER_MOD = RES / "per_modifier"
# At r = 0 the first forward step of NEXIS is the Bonferroni marginal test, so the
# corrected procedures coincide run by run; the baselines are dashed and drawn on top
# so the NEXIS line shows in the gaps.
R0_OVERLAP = {"Marginal Testing (FWER)": dict(ls="--", zorder=3),
              "Marginal Testing (FDR)": dict(ls=":", zorder=4)}


def _load_tree(key: str) -> dict[str, pd.DataFrame]:
    base = TREES[key]
    return {s: pd.read_parquet(base / f"{f}.parquet")
            for s, f in (("n", "n_sweep"), ("effect", "effect_sweep"))
            if (base / f"{f}.parquet").exists()}


def _row_df(dfs, i, method):
    sweep, fc, fv, _ = ROWS[i]
    d = dfs[sweep]
    return d[(d[fc] == fv) & (d["method"] == method)]


def _macro(dfs, method, metric):
    """Mean over the figure's cells of the per-cell seed mean."""
    cells = [_row_df(dfs, i, method).groupby(ROWS[i][3])[metric].mean() for i in range(4)]
    return float(pd.concat(cells).mean())


def _first_hit_row(dfs, i, method, metric, thr=0.95):
    m = _row_df(dfs, i, method).groupby(ROWS[i][3])[metric].mean()
    hit = m[m >= thr]
    if hit.empty:
        return "never"
    v = hit.index.min()
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def _at(dfs, method, metric, n=2000, eta=5.0):
    d = dfs["effect"]
    d = d[(d.fixed_n == n) & (d.effect_scale == eta) & (d.method == method)]
    return float(d[metric].mean()), float(d[metric].sem())


def _metric_table(entries: list[tuple[str, dict, str]]) -> list[str]:
    """entries: (label, dfs, method).  Macro P/R/IoU, value at (n=2000, eta=5) and the
    first grid value reaching 0.95 in each of the four rows, per metric."""
    out = ["| line | macro precision | macro recall | macro IoU | IoU at n=2000, eta=5 |",
           "|---|---|---|---|---|"]
    for lab, dfs, m in entries:
        iou, se = _at(dfs, m, "iou")
        out.append(f"| {lab} | {_macro(dfs, m, 'precision'):.3f} | {_macro(dfs, m, 'recall'):.3f}"
                   f" | {_macro(dfs, m, 'iou'):.3f} | {iou:.3f} ± {se:.3f} |")
    out += ["", "First grid value where the seed-mean metric reaches 0.95, per row "
            "(n @ eta=5 / n @ eta=2 / eta @ n=2000 / eta @ n=500):", "",
            "| line | precision | recall | IoU |", "|---|---|---|---|"]
    for lab, dfs, m in entries:
        cols = [" / ".join(_first_hit_row(dfs, i, m, met) for i in range(4))
                for met in ("precision", "recall", "iou")]
        out.append(f"| {lab} | " + " | ".join(cols) + " |")
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def _r0_table(dfs, methods: dict[str, str], alpha: float = 0.05) -> tuple[list[str], list[str]]:
    """Compact r = 0 table, one row per n: P(any false discovery) (= FWER) of every
    method with the Wilson 95% interval for NEXIS, and the mean number of false
    discoveries of NEXIS (every selection is false when S* is empty).  Returns the
    Markdown lines and a LaTeX tabular."""
    d = dfs["n"]
    labs = list(methods.values())
    md = [f"alpha = {alpha:g}. P(any FD) = share of runs selecting at least one coordinate.", "",
          "| n | runs | " + " | ".join(f"P(any FD): {l}" for l in labs)
          + f" | mean FD: {methods['NEXIS-v2']} |", "|---|---|" + "---|" * (len(labs) + 1)]
    tex = [r"\begin{tabular}{r" + "c" * (len(labs) + 1) + "}", r"\toprule",
           r" & \multicolumn{" + str(len(labs)) + r"}{c}{$P(\text{any false discovery})$,"
           rf" $\alpha = {alpha:g}$}} & mean false \\",
           "$n$ & " + " & ".join(l.replace("Marginal Testing", "Marginal") for l in labs)
           + rf" & discoveries ({methods['NEXIS-v2']}) \\", r"\midrule"]

    def cell(g, ci=False):
        k, runs = int((g.n_selected > 0).sum()), len(g)
        if not ci:
            return f"{k / runs:.3f}"
        lo, hi = wilson(k, runs)
        return f"{k / runs:.3f} [{lo:.3f}, {hi:.3f}]"

    groups = [("pooled", d)] + list(d.groupby("n"))
    for n, g in groups[1:] + groups[:1]:
        by = {m: g[g.method == m] for m in methods}
        runs = len(by["NEXIS-v2"])
        vals = [cell(by[m], ci=(m == "NEXIS-v2")) for m in methods]
        fd = f"{by['NEXIS-v2'].n_selected.mean():.3f}"
        name = f"{int(n)}" if n != "pooled" else "all n"
        md.append(f"| {name} | {runs} | " + " | ".join(vals) + f" | {fd} |")
        texv = [v.replace("[", "{\\scriptsize[").replace("]", "]}") for v in vals]
        if n == "pooled":
            tex.append(r"\midrule")
        tex.append(f"{name.replace(chr(32), '~')} & " + " & ".join(texv) + f" & {fd} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    return md, tex


def _per_modifier_table() -> list[str]:
    """Per-modifier recall of NEXIS-v2 at r = 1, 2, 3 (per_modifier_recall.py): macro over
    the 42 cells and per dgp.pdf row."""
    out = ["| DGP | modifier | macro | n @ eta=5 | n @ eta=2 | eta @ n=2000 | eta @ n=500 |",
           "|---|---|---|---|---|---|---|"]
    for r in ("r1", "r2", "r3"):
        f = PER_MOD / f"{r}.parquet"
        if not f.exists():
            return []
        d = pd.read_parquet(f)
        cell = d.groupby(["row", "x"])[[c for c in d.columns if c.startswith("hit_")]].mean()
        for c in cell.columns:
            per_row = cell[c].groupby(level="row").mean()
            out.append(f"| r={r[1]} | {c[4:]} | {cell[c].mean():.3f} | "
                       + " | ".join(f"{per_row[i]:.3f}" for i in range(4)) + " |")
    return out


def _fig_r0(dfs, methods: dict[str, str], out_path: Path, alpha: float = 0.05) -> None:
    """One panel per r = 0 DGP condition, in a single row: P(any false discovery)
    against n.  The uncorrected marginal test sits at 1 and is noted off scale."""
    npan = len(R0_PANELS)
    fig, axes = plt.subplots(1, npan, figsize=(5.2 * npan, 3.9), squeeze=False)
    axes = axes[0]
    top = 0.10
    for ax, (title, sweep, fc, fv) in zip(axes, R0_PANELS):
        d = dfs[sweep]
        d = d[d[fc] == fv]
        off = []
        for m, lab in methods.items():
            g = d[d.method == m].groupby("n")["n_selected"]
            if not len(g):
                continue
            k, runs = g.apply(lambda s: int((s > 0).sum())), g.size()
            p = k / runs
            if p.min() > top:
                off.append((m, lab, p.min(), p.max()))
                continue
            st = {**METHOD_STYLES[m], "label": lab, **R0_OVERLAP.get(m, {})}
            ax.plot(p.index.values, p.values, **st)
            if m == "NEXIS-v2":
                ci = np.array([wilson(int(a), int(b)) for a, b in zip(k, runs)])
                ax.fill_between(p.index.values, ci[:, 0], ci[:, 1], color=st["color"],
                                alpha=0.15, lw=0, label="NEXIS, 95% interval")
        ax.axhline(alpha, color="k", ls="--", lw=1.0, zorder=1)
        ax.text(d.n.max(), alpha + 0.003, rf"$\alpha = {alpha:g}$", ha="right", va="bottom",
                fontsize=LABEL_SIZE - 1)
        # Text stays in ink; the short coloured rule beside it carries the identity.
        for i, (m, lab, lo, hi) in enumerate(off):
            val = f"{lo:.3f}" if hi - lo < 5e-4 else f"{lo:.3f} to {hi:.3f}"
            y = 0.95 - 0.18 * i
            ax.plot([0.03, 0.08], [y, y], transform=ax.transAxes,
                    color=METHOD_STYLES[m]["color"], lw=1.5)
            ax.text(0.10, y + 0.035, f"{lab} (uncorrected): {val}\nat every $n$, off scale",
                    transform=ax.transAxes, ha="left", va="top", fontsize=10)
        ax.set_xscale("log")
        ax.set_ylim(-0.004, top)
        ax.set_xlabel(r"Sample size $n$")
        ax.set_title(title, fontsize=LABEL_SIZE)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("P(any false discovery)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=2, frameon=False, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def cmd_dgp_extra(argv: list[str]) -> None:
    """Appendix C extras for NEXIS-v2: reproducibility on an independently trained SAE and
    data-generating processes with r = 0, 1, 3 direct modifiers.

    Reads the sweeps of scripts/celeba/submit_dgp_extra.sh and the main NEXIS-v2 sweep:

      results/celeba/experiment_v2/k20/sae/              main setting (r = 2), reference
      results/celeba/experiment_v2_resample_b1/k20/sae/  replica dictionary
      results/celeba/experiment_v2_r{1,3}/k20/sae/       r = 1, 3
      results/celeba/experiment_v2_r0_fixbeta/k20/sae/   r = 0 (n sweep only, 200 seeds)
      results/celeba/per_modifier/r{1,2,3}.parquet       per-modifier recall of NEXIS-v2
                                                         (per_modifier_recall.py), optional

    and writes to --out-dir (default results/celeba/figures_v2/):

      replica_k20.pdf  dgp.pdf layout (4 DGP rows x precision | recall | IoU), replica SAE
      dgp_r1.pdf       same layout, r = 1
      dgp_r3.pdf       same layout, r = 3
      dgp_r0.pdf       P(any false discovery) = FWER against n.  At r = 0 (gamma = 0) there
                       is no effect size, so the four dgp.pdf rows collapse to one condition.
      dgp_r0_table.tex the r = 0 table
      dgp_extra.md     tables: macro metrics and first grid value reaching 0.95, per-modifier
                       recall, FWER and mean false discoveries per n at r = 0

    Run `alignment.py third-modifier` first: dgp_extra.md cites its output when present.
    """
    p = argparse.ArgumentParser(prog="figures.py dgp-extra", description=cmd_dgp_extra.__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=RES / "figures_v2")
    p.add_argument("--only", nargs="+", choices=["replica", "r0", "r1", "r3"],
                   default=["replica", "r0", "r1", "r3"])
    a = p.parse_args(argv)
    out = _abs(a.out_dir)

    main_ = _load_tree("main")
    md = ["# NEXIS-v2: reproducibility and additional DGPs (Appendix C extras)", "",
          "NEXIS-v2 = nexis(rho=0.5, backward=False, terminal_filter=True), published defaults "
          "otherwise; k=20 SigLIP TopK-SAE sparse codes, alpha=0.05, 50 seeds per cell. "
          "Macro = mean over the figure's cells of the per-cell seed mean (42 cells: "
          "11 n values x 2 eta rows + 10 eta values x 2 n rows).", ""]
    v2 = [(MAIN_METHODS_V2[m], m) for m in BASELINES] + [("NEXIS-v2", "NEXIS-v2")]

    if "replica" in a.only:
        rep = _load_tree("replica")
        make_12panel(rep["n"], rep["effect"], MAIN_METHODS_V2, out / "replica_k20.pdf")
        ent = [("NEXIS-v2, main SAE", main_, "NEXIS-v2"),
               ("NEXIS-v2, replica SAE", rep, "NEXIS-v2")]
        try:
            ent += [("NEXIS (published), main SAE", _load_tree("main_pub"), "NEXIS"),
                    ("NEXIS (published), replica SAE", _load_tree("replica_pub"), "NEXIS")]
        except FileNotFoundError:
            pass
        ent += [(f"{lab}, {tag} SAE", d, m) for tag, d in (("main", main_), ("replica", rep))
                for lab, m in v2[:3]]
        md += ["## Reproducibility: independently trained SAE (replica_k20.pdf)", "",
               "Replica = TopK SAE, same architecture and schedule, trained on 19,867 "
               "train-split images instead of the valid split (src/apps/celeba/README.md, Section 8); "
               "same encoded data, DGP, grid and seeds. S* is the F1-argmax coordinate per "
               "attribute in each dictionary (main: 5348, 5537; replica: 197, 4833).", ""]
        md += _metric_table(ent) + [""]

    for r in ("r1", "r3"):
        if r not in a.only:
            continue
        d = _load_tree(r)
        make_12panel(d["n"], d["effect"], MAIN_METHODS_V2, out / f"dgp_{r}.pdf")
        desc = {"r1": "r=1: Wearing_Hat (gamma=+1) modifies the effect; Eyeglasses is "
                      "sampled as in the main setting but only prognostic (beta=-0.2, gamma=0). "
                      "S* = {5348}.",
                "r3": "r=3: Wearing_Hat (+1), Eyeglasses (-1), Sideburns (+1); betas "
                      "0.3/-0.2/0.3; attributes sampled independently at CelebA prevalence. "
                      "S* = {5348, 5537, 1683}."}[r]
        md += [f"## {desc.split(':')[0]} (dgp_{r}.pdf)", "", desc, ""]
        md += _metric_table([(f"{lab} ({r})", d, m) for lab, m in v2]
                            + [("NEXIS-v2 (r=2, main)", main_, "NEXIS-v2")]) + [""]

    pm = _per_modifier_table()
    if pm and ("r1" in a.only or "r3" in a.only):
        md += ["## Per-modifier recall of NEXIS-v2 (r = 1, 2, 3)", "",
               "Share of runs whose selection contains each modifier's coordinate, from "
               "per_modifier_recall.py, which replays NEXIS-v2 on the same draws as the "
               "sweeps (replayed recall matches the stored sweeps in all 2,100 runs per DGP). "
               "Macro = mean over the 42 cells; the other columns average over one dgp.pdf row.",
               ""] + pm + [""]
        md += ["Why r=1 and r=2 look alike: at r=2 Eyeglasses is found at smaller n or eta "
               "than Wearing_Hat (macro recall 0.787 vs 0.713), and Hat is the binding "
               "modifier. Hat's recall drops only slightly from r=1 (0.749) to r=2 (0.713), "
               "the cost of the glasses heterogeneity left in the residual, and averaging it "
               "with the easier Eyeglasses gives r=2 macro recall 0.750, the same as r=1. "
               "At r=3, Sideburns is the binding modifier (0.597) and Hat and Eyeglasses "
               "also lose recall (0.657, 0.726).", ""]
    # The quoted file and script names are kept as in the published dgp_extra.md
    # (the script is now `alignment.py third-modifier`).
    if (out / "third_modifier_candidates.md").exists() and "r3" in a.only:
        md += ["## Choice of the third modifier at r=3", "",
               "third_modifier_candidates.md (third_modifier_candidates.py) ranks every "
               "CelebA attribute by the alignment of its best coordinate (F1 against the "
               "all-positive floor 2p/(1+p), AUC on the pre-activation and on the sparse "
               "code NEXIS regresses on, top-100 lift, gap to the runner-up) and by the "
               "largest n the independent sampler supports. Sideburns (1683) stays: the "
               "attributes with higher F1 are either high-prevalence labels whose sparse "
               "code almost never fires (Male: code recall 0.03; No_Beard, Young, "
               "Wearing_Lipstick, Heavy_Makeup: 0.00) or only weakly separated from the "
               "runner-up (Smiling: gap 0.069, code AUC 0.60, coordinate shared with "
               "Mouth_Slightly_Open and High_Cheekbones); the two that are better aligned "
               "on the code, Blond_Hair (code AUC 0.954 vs 0.805) and Bangs, have ONE "
               "image with hat and glasses, so the independent sampler fails even at "
               "n = 50 for some of 50 seeds. Sideburns supports every n up to 10,000.", ""]

    if "r0" in a.only:
        d = _load_tree("r0")
        meth0 = {"NEXIS-v2": "NEXIS",
                 **{m: MAIN_METHODS_V2[m] for m in BASELINES}}
        _fig_r0(d, meth0, out / "dgp_r0.pdf")
        md += ["## r=0: no effect modification (dgp_r0.pdf)", "",
               "tau = tau_0 = 0.5 for every unit, S* = {}; Y = 0.3 W_hat - 0.2 W_glasses "
               "+ 0.5 T + N(0,1), the main setting's prognostic effects. Images, W and T are "
               "drawn exactly as in the main setting (seeds 0-49 give the same units). With "
               "gamma = 0 there is no effect size: eta multiplies nothing, and a larger "
               "constant effect tau_0 leaves every interaction t-statistic unchanged (T is in "
               "the nuisance design), so the four dgp.pdf rows collapse to one condition, "
               "varying n; it runs with 200 seeds per n instead of 50. "
               "P(any false discovery) = FWER = share of runs selecting at least one "
               "coordinate; FD = |S_hat|.", ""]
        tmd, tex = _r0_table(d, meth0)
        md += tmd + ["", "LaTeX version: dgp_r0_table.tex.", ""]
        (out / "dgp_r0_table.tex").write_text("\n".join(tex) + "\n")

    if len(a.only) < 4:   # partial run: do not clobber the full tables
        print("\n".join(md))
        return
    (out / "dgp_extra.md").write_text("\n".join(md))
    print(f"Saved → {out / 'dgp_extra.md'}")


# ===========================================================================
# test-story: the CATE-equivalence tests on the linear and the U-shape DGP
# ===========================================================================

def first_hit(df: pd.DataFrame, x: str, method: str, metric="iou", thr=0.95) -> str:
    """Smallest grid value of x where the seed-mean metric reaches thr ("never" if none)."""
    m = df[df.method == method].groupby(x)[metric].mean()
    hit = m[m >= thr]
    if hit.empty:
        return "never"
    v = hit.index.min()
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def cmd_test_story(argv: list[str]) -> None:
    """One-glance comparison of the CATE-equivalence tests for the appendix (NEXIS-v2).

      A  main benchmark: tau linear in two binary attributes, k=20 SAE codes
         (results/celeba/experiment_v2/k20/sae/)
      B  U-shape: tau an orthogonalised quadratic of the two true coordinates, zero
         covariance with each, k=20 SAE pre-activations
         (results/celeba/experiment_v2_ushape/k20/sae_precode/)

    Both under NEXIS-v2 = nexis(rho=0.5, backward=False, terminal_filter=True) and the
    calibrated 2·min PCM rule, 50 seeds per cell.  Writes to --out-dir (default
    results/celeba/figures_v2/):

      test_story.pdf  IoU vs n at eta=5, panel A | panel B, v2 test-ablation styles
      test_story.md   table (n* for IoU >= 0.95 on the n grid at eta=5, IoU at n=2000,
                      eta=5) and a draft paragraph
      test_story.tex  the same table as a LaTeX tabular
    """
    p = argparse.ArgumentParser(prog="figures.py test-story", description=cmd_test_story.__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=RES / "figures_v2")
    OUT = _abs(p.parse_args(argv).out_dir)
    PANELS = {"A": ("Main benchmark (linear CATE)", RES / "experiment_v2/k20/sae"),
              "B": ("U-shape (zero covariance)", RES / "experiment_v2_ushape/k20/sae_precode")}
    ETA, N = 5.0, 2000
    METHODS = ABLATION_GROUPS_V2["test"]["methods"]

    def load(base: Path):
        n = pd.read_parquet(base / "n_sweep.parquet")
        e = pd.read_parquet(base / "effect_sweep.parquet")
        return n[n.fixed_effect == ETA], e[(e.fixed_n == N) & (e.effect_scale == ETA)]

    data = {k: load(p) for k, (_, p) in PANELS.items()}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (k, (title, _)) in zip(axes, PANELS.items()):
        plot_sweep(data[k][0], "n", "iou", xlabel=r"Sample size $n$  ($\eta=5$)",
                   ax=ax, methods=METHODS)
        ax.set_title(f"{k}. {title}")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.get_legend().remove()
    axes[1].set_ylabel("")
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=12)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "test_story.pdf", bbox_inches="tight")
    plt.close(fig)

    rows = []
    for m, lab in METHODS.items():
        r = {"test": lab}
        for k in PANELS:
            dn, de = data[k]
            g = de[de.method == m]
            hit = first_hit(dn, "n", m)
            r[f"{k}: n*"] = "—" if hit == "never" else hit
            r[f"{k}: IoU"] = f"{g.iou.mean():.2f}"
        rows.append(r)
    t = pd.DataFrame(rows)

    tex = [r"\begin{tabular}{lcccc}", r"\toprule",
           r" & \multicolumn{2}{c}{A. linear CATE} & \multicolumn{2}{c}{B. U-shape} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r"Test & $n^\star$ & IoU & $n^\star$ & IoU \\", r"\midrule"]
    tex += [" & ".join(str(v) for v in r.values()).replace("—", "--") + r" \\" for r in rows]
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "test_story.tex").write_text("\n".join(tex) + "\n")

    md = ["# Which CATE-equivalence test: one-glance comparison (NEXIS-v2)", "",
          "A = main benchmark (tau linear in Wearing_Hat, Eyeglasses; k=20 SAE codes). "
          "B = U-shape (tau = 0.5 + eta (g(Z^5348) - g(Z^5537)), g an orthogonalised "
          "quadratic with zero covariance with Z^j; k=20 SAE pre-activations). "
          "NEXIS-v2, calibrated 2·min PCM, 50 seeds, alpha = 0.05, m = 9,216. "
          "n* = smallest n on the grid {50, ..., 10000} with seed-mean IoU >= 0.95 at "
          "eta = 5 ('—' = never); IoU at n = 2000, eta = 5.", "",
          t.to_markdown(index=False, disable_numparse=True), "",
          "Figure: test_story.pdf (IoU vs n at eta = 5, A | B). LaTeX: test_story.tex.", "",
          "## Draft paragraph", "",
          "NEXIS only needs a valid and consistent test of H0(j|S), and the choice of test "
          "trades power for the class of alternatives it can detect. When the conditional "
          "CATE is linear in the coordinates (panel A), the linear interaction test is the "
          "most powerful, reaching IoU 0.95 at n = 750 against 2000 for the GCM and 3500 "
          "for the PCM, which pays for sample splitting. The GCM, however, only detects a "
          "non-zero conditional covariance between tau and Z^j, so a non-monotone modifier "
          "with zero covariance (panel B) is invisible to it, and to the linear test, at "
          "every sample size (recall 0 on the whole grid). The Projected Covariance Measure "
          "(Lundborg et al., 2024), applied to the pseudo-outcome, learns the direction of "
          "the conditional-mean contrast on one half of the sample and recovers both "
          "modifiers (IoU 1.00 from n = 1000 with a LightGBM projection). We therefore keep "
          "the linear test as the default and recommend the PCM when non-monotone "
          "heterogeneity cannot be ruled out."]
    (OUT / "test_story.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"Saved -> {OUT / 'test_story.pdf'}, test_story.md, test_story.tex")


# ===========================================================================

COMMANDS = {"main": cmd_main, "appendix": cmd_appendix, "dgp-extra": cmd_dgp_extra,
            "test-story": cmd_test_story}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if argv and argv[0] in ("-h", "--help") else 2)
    COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
