#!/usr/bin/env python3
"""
Backbone consistency check for the CelebA semi-synthetic benchmark.

Overlays two frozen backbones (default: SigLIP-B/16 vs DINOv2-B/14) on the same
4×3 sweep layout used by figure_appendix.py, and writes a report of the
power-frontier thresholds so the two runs can be compared numerically rather
than by eye.

Encoding: method → colour (the paper's METHOD_STYLES palette, unchanged),
backbone → line style (solid = reference, dashed = comparison).  The backbone
dimension is therefore never carried by colour alone.

Outputs (--out-dir, default results/celeba/backbone_comparison/):
  compare_main.pdf         NEXIS + marginal baselines, both backbones
  compare_<ablation>.pdf   one per method ablation (test/adjust/rho/backward)
  comparison.md            saturation thresholds, truth sets, verdicts

Usage
-----
    python src/apps/celeba/compare_backbones.py
    python src/apps/celeba/compare_backbones.py --comparison dinov2 --k 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent

from apps.celeba.backbones import BACKBONES, get_backbone, experiment_dir
from apps.celeba.visualize import MAIN_METHODS, METHOD_STYLES, ABLATION_GROUPS, _METRIC_LABEL

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

# Backbone → line style.  Colour stays with the method, so the two dimensions
# never collide; the style legend is drawn in neutral ink.
BACKBONE_LS = {"reference": "-", "comparison": "--"}


# ── Loading ───────────────────────────────────────────────────────────────────

def load_sweeps(experiment_root: Path, k: int, feat: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = experiment_root / f"k{k}" / feat
    return (pd.read_parquet(base / "n_sweep.parquet"),
            pd.read_parquet(base / "effect_sweep.parquet"))


def load_truth(experiment_root: Path, k: int, feat: str) -> dict:
    path = experiment_root / f"k{k}" / feat / "ground_truth.json"
    if not path.exists():
        return {}
    with open(path) as f:
        gt = json.load(f)
    # F1 spectra are huge; keep only the scalar summary fields
    return {key: gt[key] for key in ("backbone", "w1_neurons", "w2_neurons", "truth")
            if key in gt}


# ── Plotting ──────────────────────────────────────────────────────────────────

def _panel(ax, df_ref, df_cmp, xcol: str, metric: str, xlabel: str,
           methods: dict[str, str]) -> None:
    """One panel: every method drawn twice, once per backbone line style."""
    for df, role in ((df_ref, "reference"), (df_cmp, "comparison")):
        if df is None or df.empty:
            continue
        for method in methods:
            if method not in METHOD_STYLES:
                continue
            style = dict(METHOD_STYLES[method])
            style.pop("label", None)
            style["ls"] = BACKBONE_LS[role]
            sub = df[df["method"] == method].groupby(xcol)[metric]
            mu, se = sub.mean(), sub.sem()
            if mu.empty:
                continue
            ax.plot(mu.index.values, mu.values, **style)
            ax.fill_between(mu.index.values,
                            (mu - 1.96 * se).values, (mu + 1.96 * se).values,
                            color=style["color"], alpha=0.10)
    if xcol == "n":
        ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(_METRIC_LABEL.get(metric, metric.capitalize()))
    ax.grid(True, alpha=0.25)


def make_compare_figure(
    ref: dict, cmp_: dict, methods: dict[str, str], out_path: Path,
    ref_label: str, cmp_label: str,
) -> None:
    """4 rows (DGP conditions) × 3 cols (precision | recall | IoU), both backbones."""
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(4, 3, figsize=(13, 15))

    rows = [
        (ref["n"][ref["n"]["fixed_effect"] == ETA_MAIN],
         cmp_["n"][cmp_["n"]["fixed_effect"] == ETA_MAIN],
         "n", r"Sample size $n$", rf"varying $n$,  $\eta={int(ETA_MAIN)}$ fixed"),
        (ref["n"][ref["n"]["fixed_effect"] == ETA_ALT],
         cmp_["n"][cmp_["n"]["fixed_effect"] == ETA_ALT],
         "n", r"Sample size $n$", rf"varying $n$,  $\eta={int(ETA_ALT)}$ fixed"),
        (ref["e"][ref["e"]["fixed_n"] == N_MAIN],
         cmp_["e"][cmp_["e"]["fixed_n"] == N_MAIN],
         "effect_scale", r"Effect size $\eta$", rf"varying $\eta$,  $n={N_MAIN}$ fixed"),
        (ref["e"][ref["e"]["fixed_n"] == N_ALT],
         cmp_["e"][cmp_["e"]["fixed_n"] == N_ALT],
         "effect_scale", r"Effect size $\eta$", rf"varying $\eta$,  $n={N_ALT}$ fixed"),
    ]

    for r, (sub_ref, sub_cmp, xcol, xlabel, _) in enumerate(rows):
        for c, metric in enumerate(["precision", "recall", "iou"]):
            _panel(axes[r, c], sub_ref, sub_cmp, xcol, metric, xlabel, methods)

    fig.tight_layout(rect=[0, 0.07, 1, 1.0], h_pad=2.5)

    def _row_xcenter(r):
        return (axes[r, 0].get_position().x0 + axes[r, 2].get_position().x1) / 2

    for r, (_, _, _, _, title) in enumerate(rows):
        ytop = max(ax.get_position().y1 for ax in axes[r]) + 0.004
        fig.text(_row_xcenter(r), ytop, title, ha="center", va="bottom",
                 fontsize=LABEL_SIZE)

    # Two legends: methods carry colour, backbones carry line style (neutral ink)
    method_handles = [
        Line2D([], [], color=METHOD_STYLES[m]["color"],
               lw=METHOD_STYLES[m].get("lw", 2.0),
               marker=METHOD_STYLES[m].get("marker"), ms=5, label=lbl)
        for m, lbl in methods.items() if m in METHOD_STYLES
    ]
    backbone_handles = [
        Line2D([], [], color="0.35", lw=2.0, ls=BACKBONE_LS["reference"],  label=ref_label),
        Line2D([], [], color="0.35", lw=2.0, ls=BACKBONE_LS["comparison"], label=cmp_label),
    ]

    fig.canvas.draw()
    renderer   = fig.canvas.get_renderer()
    fig_h      = fig.get_window_extent(renderer).height
    tight_ymin = min(ax.get_tightbbox(renderer).y0 / fig_h for ax in axes.flat)

    fig.legend(handles=method_handles, loc="upper center",
               ncol=min(len(method_handles), 4),
               bbox_to_anchor=(0.5, tight_ymin + 0.008), frameon=False, fontsize=12)
    fig.legend(handles=backbone_handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, tight_ymin - 0.022), frameon=False, fontsize=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Threshold report ──────────────────────────────────────────────────────────

def saturation(df: pd.DataFrame, xcol: str, fixed_col: str, fixed_val,
               method: str, metric: str, thr: float = 0.95):
    """Smallest x at which the mean metric first reaches thr (None = never)."""
    sub = df[(df[fixed_col] == fixed_val) & (df["method"] == method)]
    m   = sub.groupby(xcol)[metric].mean()
    hit = m[m >= thr]
    return hit.index.min() if not hit.empty else None


def _fmt(v) -> str:
    if v is None:
        return "never"
    return str(int(v)) if float(v) == int(v) else str(v)


def _plateau(df: pd.DataFrame, xcol: str, fixed_col: str, fixed_val,
             method: str, metric: str) -> float | None:
    """Mean metric at the largest sweep value — the high-power plateau."""
    sub = df[(df[fixed_col] == fixed_val) & (df["method"] == method)]
    m   = sub.groupby(xcol)[metric].mean()
    return None if m.empty else float(m.loc[m.index.max()])


CONDITIONS = [
    ("n", "fixed_effect", ETA_MAIN, "n",            f"n-sweep @ η={int(ETA_MAIN)}"),
    ("n", "fixed_effect", ETA_ALT,  "n",            f"n-sweep @ η={int(ETA_ALT)}"),
    ("e", "fixed_n",      N_MAIN,   "effect_scale", f"η-sweep @ n={N_MAIN}"),
    ("e", "fixed_n",      N_ALT,    "effect_scale", f"η-sweep @ n={N_ALT}"),
]


def threshold_table(ref: dict, cmp_: dict, methods: list[str], metric: str,
                    ref_label: str, cmp_label: str) -> list[str]:
    lines = [
        f"| Condition | Method | {ref_label} | {cmp_label} |",
        "|---|---|---|---|",
    ]
    for key, fixed_col, fixed_val, xcol, cond_label in CONDITIONS:
        for m in methods:
            a = saturation(ref[key],  xcol, fixed_col, fixed_val, m, metric)
            b = saturation(cmp_[key], xcol, fixed_col, fixed_val, m, metric)
            lines.append(f"| {cond_label} | {m} | {_fmt(a)} | {_fmt(b)} |")
    return lines


def plateau_table(ref: dict, cmp_: dict, methods: list[str], metric: str,
                  ref_label: str, cmp_label: str) -> list[str]:
    lines = [
        f"| Condition | Method | {ref_label} | {cmp_label} |",
        "|---|---|---|---|",
    ]
    for key, fixed_col, fixed_val, xcol, cond_label in CONDITIONS:
        for m in methods:
            a = _plateau(ref[key],  xcol, fixed_col, fixed_val, m, metric)
            b = _plateau(cmp_[key], xcol, fixed_col, fixed_val, m, metric)
            fa = "—" if a is None else f"{a:.2f}"
            fb = "—" if b is None else f"{b:.2f}"
            lines.append(f"| {cond_label} | {m} | {fa} | {fb} |")
    return lines


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment-dir", type=Path, default="results/celeba/experiment",
                   help="Root of the sweep results tree (default: results/celeba/experiment)")
    p.add_argument("--reference",  default="siglip", choices=sorted(BACKBONES))
    p.add_argument("--comparison", default="dinov2", choices=sorted(BACKBONES))
    p.add_argument("--k",          type=int, default=20,
                   help="SAE sparsity of the main setting (default: 20)")
    p.add_argument("--out-dir",    type=Path, default="results/celeba/backbone_comparison")
    return p.parse_args()


def main():
    args = parse_args()
    exp_root = (ROOT / args.experiment_dir
                if not args.experiment_dir.is_absolute() else args.experiment_dir)
    out_dir  = (ROOT / args.out_dir
                if not args.out_dir.is_absolute() else args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_spec, cmp_spec = get_backbone(args.reference), get_backbone(args.comparison)
    ref_root = experiment_dir(exp_root, ref_spec.key)
    cmp_root = experiment_dir(exp_root, cmp_spec.key)
    ref_label, cmp_label = ref_spec.label, cmp_spec.label

    def bundle(root: Path, k: int, feat: str) -> dict:
        n, e = load_sweeps(root, k, feat)
        return {"n": n, "e": e}

    ref = bundle(ref_root, args.k, "sae")
    cmp_ = bundle(cmp_root, args.k, "sae")

    # ── Figures ───────────────────────────────────────────────────────────────
    make_compare_figure(ref, cmp_, MAIN_METHODS, out_dir / "compare_main.pdf",
                        ref_label, cmp_label)
    for key, grp in ABLATION_GROUPS.items():
        make_compare_figure(ref, cmp_, grp["methods"],
                            out_dir / f"compare_{key}.pdf", ref_label, cmp_label)

    # Dictionary ablations: k=5 codes and k=20 pre-activations
    extra_figs = []
    for k, feat, name in [(5, "sae", "k5"), (args.k, "sae_precode", "precode")]:
        try:
            r = bundle(ref_root, k, feat)
            c = bundle(cmp_root, k, feat)
        except FileNotFoundError as exc:
            print(f"Skipping {name}: {exc}")
            continue
        make_compare_figure(r, c, MAIN_METHODS, out_dir / f"compare_{name}.pdf",
                            ref_label, cmp_label)
        extra_figs.append((name, r, c))

    # ── Report ────────────────────────────────────────────────────────────────
    main_methods = list(MAIN_METHODS)
    lines = [
        f"# Backbone consistency: {ref_label} vs {cmp_label}",
        "",
        f"Same DGP, same SAE configuration (k={args.k}, hidden 9216), same 50 seeds, "
        "same NEXIS defaults. The only thing that changes is the frozen encoder "
        "producing the representation the SAE is trained on.",
        "",
        "## Ground-truth coordinates",
        "",
        "| Setting | " + ref_label + " | " + cmp_label + " |",
        "|---|---|---|",
    ]
    for k, feat in [(args.k, "sae"), (5, "sae"), (args.k, "sae_precode")]:
        g_ref = load_truth(ref_root, k, feat)
        g_cmp = load_truth(cmp_root, k, feat)
        if not g_ref and not g_cmp:
            continue
        lines.append(f"| k={k}, {feat} | {g_ref.get('truth', '—')} | {g_cmp.get('truth', '—')} |")

    lines += [
        "",
        "## Recall ≥ 0.95 threshold (smaller = more powerful)",
        "",
        *threshold_table(ref, cmp_, main_methods, "recall", ref_label, cmp_label),
        "",
        "## Precision ≥ 0.95 threshold",
        "",
        *threshold_table(ref, cmp_, main_methods, "precision", ref_label, cmp_label),
        "",
        "## Precision at the high-power end of each sweep",
        "",
        "This is where the experimental power paradox shows up: marginal screening "
        "should *lose* precision as power grows, NEXIS should not.",
        "",
        *plateau_table(ref, cmp_, main_methods, "precision", ref_label, cmp_label),
        "",
        "## Method ablations (recall ≥ 0.95)",
        "",
    ]
    for key, grp in ABLATION_GROUPS.items():
        lines += [f"### {grp['title']}", ""]
        lines += threshold_table(ref, cmp_, list(grp["methods"]), "recall",
                                 ref_label, cmp_label)
        lines += [""]

    out_md = out_dir / "comparison.md"
    out_md.write_text("\n".join(lines))
    print(f"Saved → {out_md}")


if __name__ == "__main__":
    main()
