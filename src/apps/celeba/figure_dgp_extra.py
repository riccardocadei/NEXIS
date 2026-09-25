#!/usr/bin/env python3
"""
Appendix C extras for NEXIS-v2: reproducibility on an independently trained SAE and
data-generating processes with r = 0, 1, 3 direct modifiers.

Reads the sweeps of scripts/celeba/submit_dgp_extra_v2.sh and the main NEXIS-v2 sweep:

  results/celeba/experiment_v2/k20/sae/              main setting (r = 2), reference
  results/celeba/experiment_v2_resample_b1/k20/sae/  replica dictionary
  results/celeba/experiment_v2_r{1,3}/k20/sae/       r = 1, 3
  results/celeba/experiment_v2_r0_fixbeta/k20/sae/   r = 0 (n sweep only, 200 seeds)
  results/celeba/per_modifier/r{1,2,3}.parquet       per-modifier recall of NEXIS-v2
                                                     (per_modifier_recall.py), optional

and writes to results/celeba/figures_v2/:

  replica_k20.pdf  dgp.pdf layout (4 DGP rows x precision | recall | IoU), replica SAE
  dgp_r1.pdf       same layout, r = 1
  dgp_r3.pdf       same layout, r = 3
  dgp_r0.pdf       P(any false discovery) = FWER against n, one panel per DGP condition.
                   At r = 0 (gamma = 0) there is no effect size, so the four dgp.pdf rows
                   collapse to one condition: varying n at the main prognostic effects.
  dgp_extra.md     tables: macro metrics and first grid value reaching 0.95, per-modifier
                   recall, FWER and mean false discoveries per n at r = 0

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


def load(key: str) -> dict[str, pd.DataFrame]:
    base = TREES[key]
    return {s: pd.read_parquet(base / f"{f}.parquet")
            for s, f in (("n", "n_sweep"), ("effect", "effect_sweep"))
            if (base / f"{f}.parquet").exists()}


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


def r0_table(dfs, methods: dict[str, str], alpha: float = 0.05) -> tuple[list[str], list[str]]:
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


def per_modifier_table() -> list[str]:
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


# ── r = 0 figure ─────────────────────────────────────────────────────────────

def fig_r0(dfs, methods: dict[str, str], out_path: Path, alpha: float = 0.05) -> None:
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
                fontsize=fa.LABEL_SIZE - 1)
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
        ax.set_title(title, fontsize=fa.LABEL_SIZE)
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

    pm = per_modifier_table()
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
        d = load("r0")
        meth0 = {"NEXIS-v2": "NEXIS",
                 **{m: MAIN_METHODS_V2[m] for m in BASELINES}}
        fig_r0(d, meth0, out / "dgp_r0.pdf")
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
        tmd, tex = r0_table(d, meth0)
        md += tmd + ["", "LaTeX version: dgp_r0_table.tex.", ""]
        (out / "dgp_r0_table.tex").write_text("\n".join(tex) + "\n")

    if len(a.only) < 4:   # partial run: do not clobber the full tables
        print("\n".join(md))
        return
    (out / "dgp_extra.md").write_text("\n".join(md))
    print(f"Saved → {out / 'dgp_extra.md'}")


if __name__ == "__main__":
    main()
