"""
Visualization helpers for the CelebA semi-synthetic experiment notebook.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from apps.celeba.experiment import compute_f1_scores


# ---------------------------------------------------------------------------
# Importance spectra
# ---------------------------------------------------------------------------

def plot_importance(
    gt: dict,
    labels_df: pd.DataFrame,
    out_path,
    title_prefix: str = '',
    features: np.ndarray | None = None,
    precode: bool = False,
) -> None:
    """Bar chart of top-200 F1 scores with principal-alignment verdict.

    precode=False — use 'w1_f1_scores' (z, sparse codes / raw embeddings)
    precode=True  — use 'w1_f1_scores_pre' (z_pre, continuous pre-activations)

    Pass *features* to recompute from scratch (slow, overrides precode).
    """
    if features is not None:
        print('Computing F1 scores on-the-fly …')
        f1_w1 = compute_f1_scores(features, labels_df[gt['w1_attr']].values)
        f1_w2 = compute_f1_scores(features, labels_df[gt['w2_attr']].values)
    elif precode and 'w1_f1_scores_pre' in gt:
        f1_w1 = np.array(gt['w1_f1_scores_pre'])
        f1_w2 = np.array(gt['w2_f1_scores_pre'])
    elif 'w1_f1_scores' in gt:
        f1_w1 = np.array(gt['w1_f1_scores'])
        f1_w2 = np.array(gt['w2_f1_scores'])
    else:
        raise ValueError(
            "F1 scores not cached in ground_truth.json. "
            "Either re-run run_experiment.py or pass features=<array>."
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    for ax, f1, attr in [(axes[0], f1_w1, gt['w1_attr']),
                         (axes[1], f1_w2, gt['w2_attr'])]:
        sorted_f1 = np.sort(f1)[::-1]
        n_show    = min(200, len(sorted_f1))
        best_idx  = int(np.argmax(f1))
        gap       = sorted_f1[0] / (sorted_f1[1] + 1e-8)

        ax.bar(np.arange(1, n_show + 1), sorted_f1[:n_show],
               color='#aec7e8', width=1.0, label='other dimensions')
        ax.bar([1], [sorted_f1[0]], color='#1f77b4', width=1.0,
               label='principal dimension')
        ax.axhline(sorted_f1[1], color='#1f77b4', ls='--', lw=1.0, alpha=0.6)
        ax.set_xlabel('Top dimensions (sorted by F1)')
        ax.set_ylabel('F1 score')
        ax.set_title(
            f'{title_prefix}{attr}\n'
            f'dim {best_idx}  |  F1={sorted_f1[0]:.3f}  |  gap={gap:.1f}x'
        )
        ax.legend(fontsize=9, frameon=False)
        ax.set_xlim(0.5, n_show + 0.5)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved → {out_path}')

    for f1, attr in [(f1_w1, gt['w1_attr']), (f1_w2, gt['w2_attr'])]:
        sorted_f1 = np.sort(f1)[::-1]
        gap = sorted_f1[0] / (sorted_f1[1] + 1e-8)
        verdict = 'PRINCIPALLY ALIGNED' if gap > 2.0 else 'not principally aligned'
        print(f'  {attr}: gap={gap:.1f}x  →  {verdict}')
    plt.show()


# ---------------------------------------------------------------------------
# Method-comparison sweeps
# ---------------------------------------------------------------------------

METHOD_STYLES: dict = {
    # Baselines (red/orange family)
    'Marginal':       dict(color='#d62728', marker='s', lw=1.5, ls='--', label='Marginal'),
    'Marginal (Bon)': dict(color='#ff7f0e', marker='s', lw=1.5, ls=':',  label='Marginal (Bon)'),
    # NEXIS rho ablation (blue family, darker = larger rho)
    'NEXIS (rho=0)':   dict(color='#9ecae1', marker='D', lw=1.5, ls='--', label='NEXIS (ρ=0, no stop)'),
    'NEXIS (rho=0.1)': dict(color='#4292c6', marker='D', lw=1.5, label='NEXIS (ρ=0.1)'),
    'NEXIS':           dict(color='#08519c', marker='D', lw=2.5, label='NEXIS (ρ=0.5, default)'),
    # Backward ablation (green)
    'NEXIS (no-bwd)':  dict(color='#2ca02c', marker='v', lw=1.5, ls='-.', label='NEXIS (no backward)'),
    # Test ablation (purple family)
    'NEXIS (poly2)':   dict(color='#9467bd', marker='o', lw=1.5, label='NEXIS (GCM, poly2)'),
    'NEXIS (GCM)':     dict(color='#5c3493', marker='o', lw=1.5, ls='-.', label='NEXIS (GCM, lgbm)'),
}

REPR_STYLES: dict = {
    'Raw SigLIP': dict(color='#d62728', marker='s', lw=2.0, ls='--', label='Raw SigLIP'),
    'SigLIP+SAE': dict(color='#1f77b4', marker='o', lw=2.0,           label='SigLIP+SAE'),
}


def _add_type1_region(ax, df, xcol: str, metric: str) -> None:
    if xcol == 'effect_scale' and metric in ('recall', 'iou', 'precision'):
        ax.axvspan(-0.05, 0.05, color='gray', alpha=0.10)
        ax.text(0.04, 0.04, 'Type I', rotation=90, color='gray', fontsize=8,
                transform=ax.get_xaxis_transform())


_METRIC_LABEL = {'iou': 'IoU', 'recall': 'Recall', 'precision': 'Precision'}


def plot_sweep(df: pd.DataFrame, xcol: str, metric: str,
               xlabel: str, title: str | None = None, ax=None) -> plt.Axes:
    """Mean ± 1.96 SE curves for each method vs. a sweep parameter."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))
    log_x = (xcol == 'n')
    for method, style in METHOD_STYLES.items():
        sub = df[df['method'] == method].groupby(xcol)[metric]
        mu, se = sub.mean(), sub.sem()
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
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.25)
    return ax


def plot_comparison(
    df_effect_raw: pd.DataFrame,
    df_n_raw: pd.DataFrame,
    df_effect_sae: pd.DataFrame,
    df_n_sae: pd.DataFrame,
    gt_sae: dict,
    out_path,
) -> None:
    """2×3 grid comparing NEXIS on raw SigLIP vs SigLIP+SAE."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey='row')
    row_data = [
        (df_effect_raw, df_effect_sae, 'effect_scale', r'Effect size $\eta$'),
        (df_n_raw,      df_n_sae,      'n',            r'Sample size $n$'),
    ]
    for row, (df_a, df_b, xcol, xlabel) in enumerate(row_data):
        for col, metric in enumerate(['iou', 'recall', 'precision']):
            ax = axes[row, col]
            for (_, style), df in zip(REPR_STYLES.items(), [df_a, df_b]):
                sub = df[df['method'] == 'NEXIS'].groupby(xcol)[metric]
                mu, se = sub.mean(), sub.sem()
                ax.plot(mu.index, mu.values, **style)
                ax.fill_between(mu.index,
                                (mu - 1.96 * se).values, (mu + 1.96 * se).values,
                                color=style['color'], alpha=0.15)
            _add_type1_region(ax, df_a, xcol, metric)
            sweep_label = 'effect size' if xcol == 'effect_scale' else 'sample size'
            ax.set_xlim(left=df_a[xcol].min() * 0.95)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(_METRIC_LABEL.get(metric, metric.capitalize()))
            ax.set_title(f'{_METRIC_LABEL.get(metric, metric.capitalize())} vs. {sweep_label}')
            ax.legend(fontsize=9, frameon=False)
            ax.grid(True, alpha=0.25)

    fig.suptitle(
        f'NEXIS — Raw SigLIP vs SigLIP+SAE  |  '
        f'W1={gt_sae["w1_attr"]}, W2={gt_sae["w2_attr"]}',
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved → {out_path}')
    plt.show()


# ---------------------------------------------------------------------------
# Convenience wrapper: full 3-panel sweep grid
# ---------------------------------------------------------------------------

def plot_sweep_grid(
    df: pd.DataFrame,
    xcol: str,
    xlabel: str,
    suptitle: str,
    out_path,
) -> None:
    """Save and show an (R×3) Precision / Recall / IoU grid.

    R = number of distinct fixed-parameter values in the dataframe.
    For effect sweeps the fixed parameter is 'fixed_n'; for n sweeps it is
    'fixed_effect'.  Each row is labelled with its fixed-parameter value.
    Falls back to a single row if the column is absent (legacy data).
    """
    fixed_col  = "fixed_n"     if xcol == "effect_scale" else "fixed_effect"
    fixed_label = "n"          if xcol == "effect_scale" else "η"

    if fixed_col in df.columns:
        fixed_vals = sorted(df[fixed_col].unique())
    else:
        fixed_vals = [None]

    n_rows = len(fixed_vals)
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 4 * n_rows),
                             sharey=False, squeeze=False)

    for row, fval in enumerate(fixed_vals):
        sub = df[df[fixed_col] == fval] if fval is not None else df
        for col, metric in enumerate(['precision', 'recall', 'iou']):
            ax = axes[row, col]
            plot_sweep(sub, xcol, metric, xlabel=xlabel, ax=ax)
            if col == 0:
                if fval is not None:
                    fixed_str = (f"{fixed_label}={int(fval)}"
                                 if xcol == "effect_scale"
                                 else f"{fixed_label}={fval:.1f}")
                    ax.set_ylabel(f"{fixed_str}\n{_METRIC_LABEL.get(metric, metric)}")
                else:
                    ax.set_ylabel(_METRIC_LABEL.get(metric, metric))

    fig.suptitle(suptitle, y=1.01, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved → {out_path}')
    plt.show()


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def summary_table(df: pd.DataFrame, xcol: str) -> pd.DataFrame:
    """Convert long-format sweep results to a mean ± SE summary table."""
    rows = []
    for (val, method), g in df.groupby([xcol, 'method']):
        rows.append({
            xcol: val,
            'method': method,
            'IoU':       f"{g['iou'].mean():.3f} ± {g['iou'].sem():.3f}",
            'Recall':    f"{g['recall'].mean():.3f} ± {g['recall'].sem():.3f}",
            'Precision': f"{g['precision'].mean():.3f} ± {g['precision'].sem():.3f}",
            'FP':        f"{g['fp'].mean():.2f}",
        })
    return pd.DataFrame(rows).sort_values([xcol, 'method'])
