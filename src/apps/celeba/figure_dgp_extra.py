#!/usr/bin/env python3
"""
Appendix C extras for NEXIS-v2: reproducibility on an independently trained SAE and
data-generating processes with r = 0, 1, 3 direct modifiers.

Reads the sweeps of scripts/celeba/submit_dgp_extra_v2.sh and the main NEXIS-v2 sweep:

  results/celeba/experiment_v2/k20/sae/              main setting (r = 2), reference
  results/celeba/experiment_v2_resample_b1/k20/sae/  replica dictionary
  results/celeba/experiment_v2_r{0,1,3}/k20/sae/     r = 0, 1, 3

and writes to results/celeba/figures_v2/:

  replica_k20.pdf  dgp.pdf layout (4 DGP rows x precision | recall | IoU), replica SAE
  dgp_r1.pdf       same layout, r = 1
  dgp_r3.pdf       same layout, r = 3
  dgp_r0.pdf       4 DGP rows x [FWER | false discoveries]; precision and recall are
                   undefined when S* is empty
  dgp_extra.md     tables: macro metrics and first grid value reaching 0.95, FWER at r = 0

The 12-panel figures reuse figure_appendix.make_12panel and the v2 method set, so they
follow whatever styling those modules define.

    python src/apps/celeba/figure_dgp_extra.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent.parent

import apps.celeba.figure_appendix as fa
from apps.celeba.visualize import METHOD_STYLES, MAIN_METHODS_V2

RES = ROOT / "results/celeba"
TREES = {
    "main":    RES / "experiment_v2/k20/sae",
    "replica": RES / "experiment_v2_resample_b1/k20/sae",
    "r0":      RES / "experiment_v2_r0/k20/sae",
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
ROW_NAME = {0: "n @ eta=5", 1: "n @ eta=2", 2: "eta @ n=2000", 3: "eta @ n=500"}
PUB_STYLE = dict(color="#00441b", lw=1.5, marker="s", ms=3)   # published NEXIS at r=0
PUB_LABEL = "NEXIS, interleaved backward step (published)"


def load(key: str) -> dict[str, pd.DataFrame]:
    base = TREES[key]
    return {"n": pd.read_parquet(base / "n_sweep.parquet"),
            "effect": pd.read_parquet(base / "effect_sweep.parquet")}


def row_df(dfs, i, method):
    sweep, fc, fv, _ = ROWS[i]
    d = dfs[sweep]
    return d[(d[fc] == fv) & (d["method"] == method)]


def macro(dfs, method, metric):
    """Mean over the figure's cells of the per-cell seed mean (as compare_v2.py)."""
    cells = [row_df(dfs, i, method).groupby(ROWS[i][3])[metric].mean() for i in range(4)]
    return float(pd.concat(cells).mean())


def first_hit(dfs, i, method, metric, thr=0.95):
    m = row_df(dfs, i, method).groupby(ROWS[i][3])[metric].mean()
    hit = m[m >= thr]
    if hit.empty:
        return "never"
    v = hit.index.min()
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def at(dfs, method, metric, n=2000, eta=5.0):
    d = dfs["effect"]
    d = d[(d.fixed_n == n) & (d.effect_scale == eta) & (d.method == method)]
    return float(d[metric].mean()), float(d[metric].sem())


# ── tables ───────────────────────────────────────────────────────────────────

def metric_table(entries: list[tuple[str, dict, str]]) -> list[str]:
    """entries: (label, dfs, method).  Macro P/R/IoU, value at (n=2000, eta=5) and the
    first grid value reaching 0.95 in each of the four rows, per metric."""
    out = ["| line | macro precision | macro recall | macro IoU | IoU at n=2000, eta=5 |",
           "|---|---|---|---|---|"]
    for lab, dfs, m in entries:
        iou, se = at(dfs, m, "iou")
        out.append(f"| {lab} | {macro(dfs, m, 'precision'):.3f} | {macro(dfs, m, 'recall'):.3f}"
                   f" | {macro(dfs, m, 'iou'):.3f} | {iou:.3f} ± {se:.3f} |")
    out += ["", "First grid value where the seed-mean metric reaches 0.95, per row "
            "(n @ eta=5 / n @ eta=2 / eta @ n=2000 / eta @ n=500):", "",
            "| line | precision | recall | IoU |", "|---|---|---|---|"]
    for lab, dfs, m in entries:
        cols = [" / ".join(first_hit(dfs, i, m, met) for i in range(4))
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


def fwer_table(dfs, methods: dict[str, str]) -> list[str]:
    out = ["| method | runs | FWER pooled [95% CI] | worst cell FWER (cell) | cells with FWER > 0.05 "
           "| mean false discoveries | FWER at eta=0 (n=500 / n=2000) |",
           "|---|---|---|---|---|---|---|"]
    for m, lab in methods.items():
        runs, cells = [], []
        for i in range(4):
            d = row_df(dfs, i, m)
            runs.append(d)
            for x, g in d.groupby(ROWS[i][3]):
                cells.append((float((g.n_selected > 0).mean()), f"{ROW_NAME[i]}, x={x:g}"))
        r = pd.concat(runs)
        k, n = int((r.n_selected > 0).sum()), len(r)
        lo, hi = wilson(k, n)
        worst = max(cells)
        over = sum(c > 0.05 for c, _ in cells)
        e = dfs["effect"]
        z = [e[(e.method == m) & (e.effect_scale == 0) & (e.fixed_n == nn)] for nn in (500, 2000)]
        z = " / ".join(f"{(g.n_selected > 0).mean():.2f}" if len(g) else "-" for g in z)
        out.append(f"| {lab} | {n} | {k / n:.3f} [{lo:.3f}, {hi:.3f}] | {worst[0]:.2f} ({worst[1]}) "
                   f"| {over}/{len(cells)} | {r.n_selected.mean():.3g} | {z} |")
    return out


# ── r = 0 figure ─────────────────────────────────────────────────────────────

def fig_r0(dfs, methods: dict[str, str], out_path: Path) -> None:
    """4 rows (the dgp.pdf DGP conditions) x [FWER | mean false discoveries]."""
    styles = {**METHOD_STYLES, "NEXIS": {**PUB_STYLE, "label": PUB_LABEL}}
    fig, axes = plt.subplots(4, 2, figsize=(11, 15))
    # FWER axis: the corrected procedures live near 0.05; the unadjusted marginal test
    # sits at 1 and is read off the false-discoveries panel instead.
    fwer_max = 0.0
    for i in range(4):
        _, _, _, xcol = ROWS[i]
        for m, lab in methods.items():
            d = row_df(dfs, i, m)
            if d.empty:
                continue
            st = {**styles[m], "label": lab}
            g = d.groupby(xcol)
            k, n = g["n_selected"].apply(lambda s: int((s > 0).sum())), g.size()
            p = k / n
            ci = np.array([wilson(int(a), int(b)) for a, b in zip(k, n)])
            fd = g["n_selected"].mean()
            fd_se = g["n_selected"].sem()
            x = p.index.values
            if m != "Marginal Testing":
                fwer_max = max(fwer_max, float(ci[:, 1].max()))
            axes[i, 0].plot(x, p.values, **st)
            # The Bonferroni-type procedures coincide at r = 0; one Wilson band (NEXIS-v2)
            # keeps the panel readable instead of stacking four identical fills.
            if m == "NEXIS-v2":
                axes[i, 0].fill_between(x, ci[:, 0], ci[:, 1], color=st["color"], alpha=0.15)
            axes[i, 1].plot(x, fd.values, **st)
            if m in ("NEXIS-v2", "Marginal Testing"):
                axes[i, 1].fill_between(x, np.maximum(fd - 1.96 * fd_se, 0),
                                        fd + 1.96 * fd_se, color=st["color"], alpha=0.15)
        xlabel = r"Sample size $n$" if xcol == "n" else r"Prognostic scale $\eta$"
        for c, ax in enumerate(axes[i]):
            if xcol == "n":
                ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.grid(True, alpha=0.25)
        axes[i, 0].axhline(0.05, color="k", ls="--", lw=1.0)
        axes[i, 0].set_ylabel("FWER")
        axes[i, 1].set_yscale("symlog", linthresh=0.1)
        axes[i, 1].set_ylim(0, 2000)
        axes[i, 1].set_ylabel("False discoveries")
    top = max(0.2, np.ceil(fwer_max * 20) / 20 + 0.05)
    for ax in axes[:, 0]:
        ax.set_ylim(-0.005, top)

    handles, labels = axes[0, 1].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="k", ls="--", lw=1.0))
    labels.append(r"$\alpha = 0.05$")
    fig.tight_layout(rect=[0, 0.05, 1, 1.0], h_pad=2.5)
    titles = [r"varying $n$,  $\eta=5$ fixed", r"varying $n$,  $\eta=2$ fixed",
              r"varying $\eta$,  $n=2000$ fixed", r"varying $\eta$,  $n=500$ fixed"]
    for i, t in enumerate(titles):
        xc = (axes[i, 0].get_position().x0 + axes[i, 1].get_position().x1) / 2
        yt = max(ax.get_position().y1 for ax in axes[i]) + 0.004
        fig.text(xc, yt, t, ha="center", va="bottom", fontsize=fa.LABEL_SIZE)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    h = fig.get_window_extent(rend).height
    ymin = min(ax.get_tightbbox(rend).y0 / h for ax in axes.flat)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, ymin + 0.01))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=RES / "figures_v2")
    p.add_argument("--only", nargs="+", choices=["replica", "r0", "r1", "r3"],
                   default=["replica", "r0", "r1", "r3"])
    a = p.parse_args()
    out = a.out_dir if a.out_dir.is_absolute() else ROOT / a.out_dir

    main_ = load("main")
    md = ["# NEXIS-v2: reproducibility and additional DGPs (Appendix C extras)", "",
          "NEXIS-v2 = nexis(rho=0.5, backward=False, terminal_filter=True), published defaults "
          "otherwise; k=20 SigLIP TopK-SAE sparse codes, alpha=0.05, 50 seeds per cell. "
          "Macro = mean over the figure's cells of the per-cell seed mean (42 cells: "
          "11 n values x 2 eta rows + 10 eta values x 2 n rows).", ""]
    v2 = [(MAIN_METHODS_V2[m], m) for m in BASELINES] + [("NEXIS-v2", "NEXIS-v2")]

    if "replica" in a.only:
        rep = load("replica")
        fa.make_12panel(rep["n"], rep["effect"], MAIN_METHODS_V2, out / "replica_k20.pdf")
        ent = [("NEXIS-v2, main SAE", main_, "NEXIS-v2"),
               ("NEXIS-v2, replica SAE", rep, "NEXIS-v2")]
        try:
            ent += [("NEXIS (published), main SAE", load("main_pub"), "NEXIS"),
                    ("NEXIS (published), replica SAE", load("replica_pub"), "NEXIS")]
        except FileNotFoundError:
            pass
        ent += [(f"{lab}, {tag} SAE", d, m) for tag, d in (("main", main_), ("replica", rep))
                for lab, m in v2[:3]]
        md += ["## Reproducibility: independently trained SAE (replica_k20.pdf)", "",
               "Replica = TopK SAE, same architecture and schedule, trained on 19,867 "
               "train-split images instead of the valid split (docs/celeba_sae_resampling.md); "
               "same encoded data, DGP, grid and seeds. S* is the F1-argmax coordinate per "
               "attribute in each dictionary (main: 5348, 5537; replica: 197, 4833).", ""]
        md += metric_table(ent) + [""]

    for r in ("r1", "r3"):
        if r not in a.only:
            continue
        d = load(r)
        fa.make_12panel(d["n"], d["effect"], MAIN_METHODS_V2, out / f"dgp_{r}.pdf")
        desc = {"r1": "r=1: Wearing_Hat (gamma=+1) modifies the effect; Eyeglasses is "
                      "sampled as in the main setting but only prognostic (beta=-0.2, gamma=0). "
                      "S* = {5348}.",
                "r3": "r=3: Wearing_Hat (+1), Eyeglasses (-1), Sideburns (+1); betas "
                      "0.3/-0.2/0.3; attributes sampled independently at CelebA prevalence. "
                      "S* = {5348, 5537, 1683}."}[r]
        md += [f"## {desc.split(':')[0]} (dgp_{r}.pdf)", "", desc, ""]
        md += metric_table([(f"{lab} ({r})", d, m) for lab, m in v2]
                           + [("NEXIS-v2 (r=2, main)", main_, "NEXIS-v2")]) + [""]

    if "r0" in a.only:
        d = load("r0")
        meth0 = {**{m: MAIN_METHODS_V2[m] for m in BASELINES},
                 "NEXIS": PUB_LABEL, "NEXIS-v2": "NEXIS"}
        fig_r0(d, meth0, out / "dgp_r0.pdf")
        md += ["## r=0: no effect modification (dgp_r0.pdf)", "",
               "tau = tau_0 = 0.5 for every unit, S* = {}. Images, W and T are drawn exactly as "
               "in the main setting (same seeds give the same units); the x-axis 'eta' scales "
               "the prognostic main effects, Y = eta (0.3 W_hat - 0.2 W_glasses) + 0.5 T + N(0,1). "
               "A constant-effect sweep would be degenerate: T is in the nuisance design of the "
               "linear test, so adding c T to Y leaves every interaction statistic unchanged "
               "(an exact identity of the OLS fit). The eta grid adds eta=0 "
               "(pure noise). FWER = share of runs selecting at least one coordinate; false "
               "discoveries = |S_hat|. Cells: 11 n x 2 + 11 eta x 2 = 44, 50 seeds each.", ""]
        md += fwer_table(d, meth0) + [""]

    if len(a.only) < 4:   # partial run: do not clobber the full tables
        print("\n".join(md))
        return
    (out / "dgp_extra.md").write_text("\n".join(md))
    print(f"Saved → {out / 'dgp_extra.md'}")


if __name__ == "__main__":
    main()
