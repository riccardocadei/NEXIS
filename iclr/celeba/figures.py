"""Figures and tables of the CelebA benchmark (Figure 3 and Appendix C).

Every sweep figure uses the 4 x 3 layout of Appendix C: rows are the four DGP conditions
(n sweep at eta = 5, n sweep at eta = 2, eta sweep at n = 2000, eta sweep at n = 500) and
columns are precision | recall | IoU; curves are seed means with +-1.96 SE bands.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
from celeba.experiment import load_results  # noqa: E402

MAIN_RC = {"font.size": 15, "axes.labelsize": 15, "axes.titlesize": 15,
           "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14}
APPENDIX_RC = {"font.size": 13, "axes.labelsize": 13, "axes.titlesize": 13,
               "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12}
LABEL_SIZE = 13
ETA_MAIN, ETA_ALT, N_MAIN, N_ALT = 5.0, 2.0, 2000, 500

# ── styles ────────────────────────────────────────────────────────────────────
# Okabe-Ito colours (without its blue, kept for the default); lines that overlap another
# one are dashed and drawn on top so the line underneath shows through the gaps.
_ORANGE, _GREEN, _VERMILLION, _PURPLE = "#E69F00", "#009E73", "#D55E00", "#CC79A7"
_ON_TOP = dict(ls="--", zorder=3)
STYLES: Dict[str, dict] = {
    "Marginal Testing":        dict(color="#d62728", lw=1.5, marker="o", ms=3),
    "Marginal Testing (FWER)": dict(color="#ff7f0e", lw=1.5, marker="o", ms=3),
    "Marginal Testing (FDR)":  dict(color="#e377c2", lw=1.5, marker="o", ms=3),
    "NEXIS":                   dict(color="#08519c", lw=2.5, marker="o", ms=3),
    "NEXIS (test=GCM: quadratic)": dict(color=_PURPLE, lw=1.5, marker="o", ms=3, **_ON_TOP),
    "NEXIS (test=GCM: lgbm)":      dict(color=_VERMILLION, lw=2.0, marker="o", ms=3),
    "NEXIS (test=PCM: quadratic)": dict(color=_ORANGE, lw=1.5, marker="s", ms=3, **_ON_TOP),
    "NEXIS (test=PCM: lgbm)":      dict(color=_GREEN, lw=2.0, marker="s", ms=3),
    "NEXIS (adjust=None)":     dict(color=_VERMILLION, lw=1.5, marker="o", ms=3),
    "NEXIS (adjust=FDR)":      dict(color=_GREEN, lw=1.5, marker="o", ms=3, **_ON_TOP),
    "NEXIS (rho=0)":           dict(color=_ORANGE, lw=1.5, marker="o", ms=3, **_ON_TOP),
    "NEXIS (rho=0.2)":         dict(color=_GREEN, lw=1.5, marker="o", ms=3),
    "NEXIS (rho=0.8)":         dict(color=_VERMILLION, lw=1.5, marker="o", ms=3),
    "NEXIS (forward)":         dict(color=_ORANGE, lw=2.0),
    "NEXIS (forward + interleaved backward)": dict(color=_VERMILLION, lw=2.0, **_ON_TOP),
    "NEXIS (forward + interleaved and terminal backward)": dict(color=_GREEN, lw=2.0, **_ON_TOP),
}

MAIN_METHODS = {"Marginal Testing": "Marginal Testing",
                "Marginal Testing (FDR)": "Marginal Testing (FDR)",
                "Marginal Testing (FWER)": "Marginal Testing (FWER)",
                "NEXIS": "NEXIS"}

# One figure per NEXIS design axis: {method: legend label}, plus layout options.
ABLATIONS: Dict[str, dict] = {
    "test": {"legend_ncol": 5, "methods": {
        "NEXIS": "linear (default)",
        "NEXIS (test=GCM: quadratic)": "GCM: quadratic",
        "NEXIS (test=GCM: lgbm)": "GCM: LightGBM",
        "NEXIS (test=PCM: quadratic)": "PCM: quadratic",
        "NEXIS (test=PCM: lgbm)": "PCM: LightGBM"}},
    "adjust": {"methods": {
        "NEXIS (adjust=None)": "None",
        "NEXIS (adjust=FDR)": "FDR",
        "NEXIS": "FWER (default)"}},
    "rho": {"methods": {
        "NEXIS (rho=0)": "0",
        "NEXIS (rho=0.2)": "0.2",
        "NEXIS": "0.5 (default)",
        "NEXIS (rho=0.8)": "0.8"}},
    "backward": {"legend_ncol": 2, "styles": {"NEXIS": dict(marker=None)}, "methods": {
        "NEXIS (forward)": "forward",
        "NEXIS (forward + interleaved backward)": "forward + interleaved backward",
        "NEXIS": "forward + terminal backward (default)",
        "NEXIS (forward + interleaved and terminal backward)":
            "forward + interleaved and terminal backward"}},
}
METRIC_LABEL = {"iou": "IoU", "recall": "Recall", "precision": "Precision"}
# The four DGP conditions of the 4 x 3 layout: (sweep, fixed column, fixed value, x column)
ROWS = [("n", "fixed_effect", ETA_MAIN, "n"), ("n", "fixed_effect", ETA_ALT, "n"),
        ("effect", "fixed_n", N_MAIN, "effect_scale"), ("effect", "fixed_n", N_ALT, "effect_scale")]


# ── building blocks ───────────────────────────────────────────────────────────

def plot_sweep(df: pd.DataFrame, xcol: str, metric: str, xlabel: str, ax,
               methods: Dict[str, str], styles: Optional[Dict[str, dict]] = None):
    """Seed mean +-1.96 SE of one metric against the swept parameter, one line per method."""
    for method, label in methods.items():
        style = {**STYLES[method], **(styles or {}).get(method, {}), "label": label}
        sub = df[df["method"] == method].groupby(xcol)[metric]
        mu, se = sub.mean(), sub.sem()
        if mu.empty:
            continue
        ax.plot(mu.index.values, mu.values, **style)
        ax.fill_between(mu.index.values, (mu - 1.96 * se).values, (mu + 1.96 * se).values,
                        color=style["color"], alpha=0.15)
    if xcol == "effect_scale" and metric in ("recall", "iou", "precision"):
        # shaded eta = 0 band; it lies left of the axis limits and only fixes the autoscale
        ax.axvspan(-0.05, 0.05, color="gray", alpha=0.10)
    if xcol == "n":
        ax.set_xscale("log")
    ax.set_xlim(left=df[xcol].min() * 0.95)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(METRIC_LABEL.get(metric, metric.capitalize()))
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.25)
    return ax


def make_12panel(df_n: pd.DataFrame, df_e: pd.DataFrame, methods: Dict[str, str], out: Path,
                 legend_ncol: Optional[int] = None,
                 styles: Optional[Dict[str, dict]] = None) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(13, 15))
    row_cfg = [
        (df_n[df_n["fixed_effect"] == ETA_MAIN], "n", r"Sample size $n$",
         rf"varying $n$,  $\eta={int(ETA_MAIN)}$ fixed"),
        (df_n[df_n["fixed_effect"] == ETA_ALT], "n", r"Sample size $n$",
         rf"varying $n$,  $\eta={int(ETA_ALT)}$ fixed"),
        (df_e[df_e["fixed_n"] == N_MAIN], "effect_scale", r"Effect size $\eta$",
         rf"varying $\eta$,  $n={N_MAIN}$ fixed"),
        (df_e[df_e["fixed_n"] == N_ALT], "effect_scale", r"Effect size $\eta$",
         rf"varying $\eta$,  $n={N_ALT}$ fixed"),
    ]
    for r, (sub, xcol, xlabel, _) in enumerate(row_cfg):
        for c, metric in enumerate(["precision", "recall", "iou"]):
            plot_sweep(sub, xcol, metric, xlabel=xlabel, ax=axes[r, c], methods=methods,
                       styles=styles)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes.flat:
        ax.set_title("")
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    fig.tight_layout(rect=[0, 0.05, 1, 1.0], h_pad=2.5)
    for r, (_, _, _, title) in enumerate(row_cfg):
        ytop = max(ax.get_position().y1 for ax in axes[r]) + 0.004
        xc = (axes[r, 0].get_position().x0 + axes[r, 2].get_position().x1) / 2
        fig.text(xc, ytop, title, ha="center", va="bottom", fontsize=LABEL_SIZE)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_h = fig.get_window_extent(renderer).height
    tight_ymin = min(ax.get_tightbbox(renderer).y0 / fig_h for ax in axes.flat)
    fig.legend(handles, labels, loc="upper center", ncol=legend_ncol or min(len(handles), 4),
               bbox_to_anchor=(0.5, tight_ymin + 0.01), frameon=False, fontsize=12)
    _save(fig, out)


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {out}")


def _concat(*dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    return {s: pd.concat([d[s] for d in dfs if s in d], ignore_index=True)
            for s in ("effect", "n") if any(s in d for d in dfs)}


# ── the figures ───────────────────────────────────────────────────────────────

def figure_main(res: Dict[str, pd.DataFrame], out: Path) -> None:
    """Figure 3: precision and recall against n at eta = 5 and against eta at n = 2000."""
    df_left = res["n"][res["n"]["fixed_effect"] == ETA_MAIN]
    df_right = res["effect"][res["effect"]["fixed_n"] == N_MAIN]
    methods = {**{k: v for k, v in MAIN_METHODS.items() if k != "NEXIS"}, "NEXIS": "NEXIS (ours)"}
    with plt.rc_context(MAIN_RC):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
        plot_sweep(df_left, "n", "precision", r"Sample size $n$", axes[0], methods)
        plot_sweep(df_left, "n", "recall", r"Sample size $n$", axes[1], methods)
        plot_sweep(df_right, "effect_scale", "precision", r"Effect size $\eta$", axes[2], methods)
        plot_sweep(df_right, "effect_scale", "recall", r"Effect size $\eta$", axes[3], methods)
        handles, labels = axes[0].get_legend_handles_labels()
        for ax in axes:
            ax.set_title("")
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        fig.tight_layout(rect=[0, 0.10, 1, 0.92])
        for ax in axes[2:]:                                  # extra gap between the pairs
            pos = ax.get_position()
            ax.set_position([pos.x0 + 0.04, pos.y0, pos.width, pos.height])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fig_h = fig.get_window_extent(renderer).height
        tight_ymin = min(ax.get_tightbbox(renderer).y0 / fig_h for ax in axes)
        fig.legend(handles, labels, loc="upper center", ncol=len(handles),
                   bbox_to_anchor=(0.5, tight_ymin + 0.01), frameon=False, fontsize=14)

        def xcenter(a, b):
            return (a.get_position().x0 + b.get_position().x1) / 2
        fig.text(xcenter(axes[0], axes[1]), 0.97, rf"$\eta={int(ETA_MAIN)}$",
                 ha="center", va="top", fontsize=15)
        fig.text(xcenter(axes[2], axes[3]), 0.97, rf"$n={N_MAIN}$",
                 ha="center", va="top", fontsize=15)
        _save(fig, out)


def figure_12panel(res: Dict[str, pd.DataFrame], out: Path, methods=MAIN_METHODS,
                   legend_ncol=None, styles=None) -> None:
    with plt.rc_context(APPENDIX_RC):
        make_12panel(res["n"], res["effect"], methods, out, legend_ncol=legend_ncol,
                     styles=styles)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


R0_METHODS = {"NEXIS": "NEXIS", **{m: MAIN_METHODS[m] for m in C.BASELINES}}
# At r = 0 the first forward step of NEXIS is the Bonferroni marginal test, so the corrected
# procedures coincide run by run: the baselines are dashed and drawn on top.
R0_OVERLAP = {"Marginal Testing (FWER)": dict(ls="--", zorder=3),
              "Marginal Testing (FDR)": dict(ls=":", zorder=4)}


def r0_panels():
    b = C.DGP_MAIN["betas"]
    if C.R0_MODE == "fixed_beta":
        return [(rf"no modifier ($r=0$), $\beta = ({b[0]:g}, {b[1]:g})$", "n", "fixed_effect",
                 1.0, "n")]
    return [(rf"$r=0$, varying $n$, $\eta={int(ETA_MAIN)}$", "n", "fixed_effect", ETA_MAIN, "n"),
            (rf"$r=0$, varying $n$, $\eta={int(ETA_ALT)}$", "n", "fixed_effect", ETA_ALT, "n"),
            (rf"$r=0$, varying $\eta$, $n={N_MAIN}$", "effect", "fixed_n", N_MAIN, "effect_scale"),
            (rf"$r=0$, varying $\eta$, $n={N_ALT}$", "effect", "fixed_n", N_ALT, "effect_scale")]


def figure_r0(res: Dict[str, pd.DataFrame], out: Path, alpha: float = C.ALPHA) -> None:
    """P(any false discovery) (= FWER, since S* is empty) against the swept parameter.
    The uncorrected marginal test sits near 1 and is noted off scale."""
    panels = r0_panels()
    top = 0.10
    with plt.rc_context(APPENDIX_RC):
        fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.9), squeeze=False)
        axes = axes[0]
        for ax, (title, sweep, fc, fv, x) in zip(axes, panels):
            d = res[sweep]
            d = d[d[fc] == fv]
            off = []
            for m, lab in R0_METHODS.items():
                g = d[d.method == m].groupby(x)["n_selected"]
                if not len(g):
                    continue
                k, runs = g.apply(lambda s: int((s > 0).sum())), g.size()
                p = k / runs
                if p.min() > top:
                    off.append((m, lab, p.min(), p.max()))
                    continue
                st = {**STYLES[m], "label": lab, **R0_OVERLAP.get(m, {})}
                ax.plot(p.index.values, p.values, **st)
                if m == "NEXIS":
                    ci = np.array([wilson(int(a), int(b)) for a, b in zip(k, runs)])
                    ax.fill_between(p.index.values, ci[:, 0], ci[:, 1], color=st["color"],
                                    alpha=0.15, lw=0, label="NEXIS, 95% interval")
            ax.axhline(alpha, color="k", ls="--", lw=1.0, zorder=1)
            ax.text(d[x].max(), alpha + 0.003, rf"$\alpha = {alpha:g}$", ha="right",
                    va="bottom", fontsize=LABEL_SIZE - 1)
            for i, (m, lab, lo, hi) in enumerate(off):
                val = f"{lo:.3f}" if hi - lo < 5e-4 else f"{lo:.3f} to {hi:.3f}"
                y = 0.95 - 0.18 * i
                ax.plot([0.03, 0.08], [y, y], transform=ax.transAxes, color=STYLES[m]["color"],
                        lw=1.5)
                where = "$n$" if x == "n" else r"$\eta$"
                ax.text(0.10, y + 0.035, f"{lab} (uncorrected): {val}\nat every {where}, off scale",
                        transform=ax.transAxes, ha="left", va="top", fontsize=10)
            if x == "n":
                ax.set_xscale("log")
            ax.set_ylim(-0.004, top)
            ax.set_xlabel(r"Sample size $n$" if x == "n" else r"Effect size $\eta$")
            ax.set_title(title, fontsize=LABEL_SIZE)
            ax.grid(True, alpha=0.25)
        axes[0].set_ylabel("P(any false discovery)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.tight_layout()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2,
                   frameon=False, fontsize=11)
        _save(fig, out)


USHAPE_TABLE_METHODS = {"Marginal Testing (FWER)": "Marginal Testing (FWER)",
                        **ABLATIONS["test"]["methods"]}


def figure_ushape(res: Dict[str, pd.DataFrame], out: Path) -> None:
    """U-shape DGP: n sweep at eta = 5 and eta sweep at n = 2000, with false discoveries."""
    dn = res["n"][res["n"].fixed_effect == ETA_MAIN]
    de = res["effect"][res["effect"].fixed_n == N_MAIN]
    methods = ABLATIONS["test"]["methods"]
    with plt.rc_context(APPENDIX_RC):
        fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
        rows = [(dn, "n", r"Sample size $n$", rf"varying $n$,  $\eta={int(ETA_MAIN)}$ fixed"),
                (de, "effect_scale", r"Effect size $\eta$", rf"varying $\eta$,  $n={N_MAIN}$ fixed")]
        for r, (d, x, xl, _) in enumerate(rows):
            for c, met in enumerate(["precision", "recall", "iou", "fp"]):
                ax = plot_sweep(d, x, met, xlabel=xl, ax=axes[r, c], methods=methods)
                if met == "fp":
                    top = d[d.method.isin(methods)].groupby(["method", x]).fp.mean().max()
                    ax.set_ylim(-0.03 * max(top, 1.0), 1.05 * max(top, 1.0))
                    ax.set_ylabel("False discoveries")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        for ax in axes.flat:
            ax.set_title("")
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        fig.tight_layout(rect=[0, 0.07, 1, 1.0], h_pad=2.5)
        for r, (*_, title) in enumerate(rows):
            ytop = max(ax.get_position().y1 for ax in axes[r]) + 0.004
            xc = (axes[r, 0].get_position().x0 + axes[r, 3].get_position().x1) / 2
            fig.text(xc, ytop, title, ha="center", va="bottom", fontsize=LABEL_SIZE)
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=12)
        _save(fig, out)


# ── tables ────────────────────────────────────────────────────────────────────

def first_hit(df: pd.DataFrame, fc: str, fv, x: str, method: str, metric: str,
              thr: float = 0.95) -> str:
    m = df[(df[fc] == fv) & (df.method == method)].groupby(x)[metric].mean()
    hit = m[m >= thr]
    if hit.empty:
        return "never"
    v = hit.index.min()
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def table_12panel(res: Dict[str, pd.DataFrame], methods: Dict[str, str]) -> List[str]:
    """Macro metrics (mean over the cells of the seed means) and, per figure row, the first
    grid value at which the seed-mean metric reaches 0.95."""
    out = ["| method | macro precision | macro recall | macro IoU | IoU at n=2000, eta=5 | "
           "first 0.95: precision | recall | IoU |", "|---|---|---|---|---|---|---|---|"]
    for m, lab in methods.items():
        if not (res["n"].method == m).any():
            continue
        e = res["effect"]
        at = e[(e.fixed_n == N_MAIN) & (e.effect_scale == ETA_MAIN) & (e.method == m)].iou
        macro = {}
        for met in ("precision", "recall", "iou"):
            cells = [res[s][(res[s][fc] == fv) & (res[s].method == m)].groupby(x)[met].mean()
                     for s, fc, fv, x in ROWS]
            macro[met] = float(pd.concat(cells).mean())
        hits = [" / ".join(first_hit(res[s], fc, fv, x, m, met) for s, fc, fv, x in ROWS)
                for met in ("precision", "recall", "iou")]
        out.append(f"| {lab} | {macro['precision']:.3f} | {macro['recall']:.3f} | "
                   f"{macro['iou']:.3f} | {at.mean():.3f} | " + " | ".join(hits) + " |")
    return out + ["", "First 0.95 columns: n at eta=5 / n at eta=2 / eta at n=2000 / "
                  "eta at n=500 (\"never\": not reached on the grid).", ""]


def per_modifier_table(res: Dict[str, pd.DataFrame], coords: Dict[str, int],
                       attrs: List[str]) -> List[str]:
    """Share of NEXIS runs whose selection contains each modifier's principal coordinate."""
    out = ["| modifier (coordinate) | macro | n @ eta=5 | n @ eta=2 | eta @ n=2000 | eta @ n=500 |",
           "|---|---|---|---|---|---|"]
    for a in attrs:
        j = coords[a]
        per_row = []
        for s, fc, fv, x in ROWS:
            d = res[s][(res[s][fc] == fv) & (res[s].method == "NEXIS")]
            per_row.append(d.assign(hit=d.selected.apply(lambda sel: j in list(sel)))
                           .groupby(x).hit.mean())
        macro = float(pd.concat(per_row).mean())
        out.append(f"| {a} ({j}) | {macro:.3f} | "
                   + " | ".join(f"{r.mean():.3f}" for r in per_row) + " |")
    return out + [""]


def table_r0(res: Dict[str, pd.DataFrame], alpha: float = C.ALPHA):
    """P(any false discovery) per n for every method (Wilson 95% interval for NEXIS) and the
    mean number of false discoveries of NEXIS; Markdown lines and a LaTeX tabular."""
    d = res["n"]
    if C.R0_MODE != "fixed_beta":
        d = d[d.fixed_effect == ETA_MAIN]
    labs = list(R0_METHODS.values())
    md = [f"alpha = {alpha:g}. P(any FD) = share of runs selecting at least one coordinate.", "",
          "| n | runs | " + " | ".join(f"P(any FD): {lab}" for lab in labs)
          + " | mean FD: NEXIS |", "|---|---|" + "---|" * (len(labs) + 1)]
    tex = [r"\begin{tabular}{r" + "c" * (len(labs) + 1) + "}", r"\toprule",
           r" & \multicolumn{" + str(len(labs)) + r"}{c}{$P(\text{any false discovery})$,"
           rf" $\alpha = {alpha:g}$}} & mean false \\",
           "$n$ & " + " & ".join(lab.replace("Marginal Testing", "Marginal") for lab in labs)
           + r" & discoveries (NEXIS) \\", r"\midrule"]

    def cell(g, ci=False):
        k, runs = int((g.n_selected > 0).sum()), len(g)
        if not ci:
            return f"{k / runs:.3f}"
        lo, hi = wilson(k, runs)
        return f"{k / runs:.3f} [{lo:.3f}, {hi:.3f}]"

    groups = [("pooled", d)] + list(d.groupby("n"))
    for n, g in groups[1:] + groups[:1]:
        by = {m: g[g.method == m] for m in R0_METHODS}
        vals = [cell(by[m], ci=(m == "NEXIS")) for m in R0_METHODS]
        fd = f"{by['NEXIS'].n_selected.mean():.3f}"
        name = f"{int(n)}" if n != "pooled" else "all n"
        md.append(f"| {name} | {len(by['NEXIS'])} | " + " | ".join(vals) + f" | {fd} |")
        texv = [v.replace("[", "{\\scriptsize[").replace("]", "]}") for v in vals]
        if n == "pooled":
            tex.append(r"\midrule")
        tex.append(f"{name.replace(' ', '~')} & " + " & ".join(texv) + f" & {fd} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    return md, tex


def table_ushape(res: Dict[str, pd.DataFrame]) -> List[str]:
    dn = res["n"][res["n"].fixed_effect == ETA_MAIN]
    de = res["effect"][res["effect"].fixed_n == N_MAIN]
    at = de[de.effect_scale == ETA_MAIN]
    out = ["| method | n* (eta=5) | eta* (n=2000) | precision | recall | IoU | false disc. "
           "| runs with >=1 false disc. |", "|---|---|---|---|---|---|---|---|"]
    for m, lab in USHAPE_TABLE_METHODS.items():
        g = at[at.method == m]

        def f(c):
            return f"{g[c].mean():.2f} ± {g[c].sem():.2f}"
        out.append(f"| {lab} | {first_hit(dn, 'fixed_effect', ETA_MAIN, 'n', m, 'iou')} | "
                   f"{first_hit(de, 'fixed_n', N_MAIN, 'effect_scale', m, 'iou')} | "
                   f"{f('precision')} | {f('recall')} | {f('iou')} | {f('fp')} | "
                   f"{(g.fp > 0).mean():.2f} |")
    return out + ["", "n* / eta*: first grid value with seed-mean IoU >= 0.95; metrics at "
                  "n = 2000, eta = 5 (mean ± SE).", ""]


def table_test_comparison(main_method: Dict[str, pd.DataFrame],
                          ush: Dict[str, pd.DataFrame]) -> List[str]:
    """LaTeX tabular: per CATE-equivalence test, n* (first n with seed-mean IoU >= 0.95 at
    eta = 5) and the IoU at n = 2000, eta = 5, on the main DGP (A) and the U-shape (B)."""
    tex = [r"\begin{tabular}{lcccc}", r"\toprule",
           r" & \multicolumn{2}{c}{A. linear CATE} & \multicolumn{2}{c}{B. U-shape} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r"Test & $n^\star$ & IoU & $n^\star$ & IoU \\", r"\midrule"]
    for m, lab in ABLATIONS["test"]["methods"].items():
        cells = [lab]
        for res in (main_method, ush):
            hit = first_hit(res["n"], "fixed_effect", ETA_MAIN, "n", m, "iou")
            e = res["effect"]
            g = e[(e.fixed_n == N_MAIN) & (e.effect_scale == ETA_MAIN) & (e.method == m)]
            cells += ["--" if hit == "never" else hit, f"{g.iou.mean():.2f}"]
        tex.append(" & ".join(cells) + r" \\")
    return tex + [r"\bottomrule", r"\end{tabular}"]


# ── registry ──────────────────────────────────────────────────────────────────

FIGURES = {
    # name: (experiments needed, drawer)
    "figure_main":     (["main"], lambda r, o: figure_main(r["main"], o)),
    "dgp":             (["main"], lambda r, o: figure_12panel(r["main"], o)),
    "model_k5":        (["model_k5"], lambda r, o: figure_12panel(r["model_k5"], o)),
    "model_precode":   (["model_precode"], lambda r, o: figure_12panel(r["model_precode"], o)),
    "replica_k20":     (["replica"], lambda r, o: figure_12panel(r["replica"], o)),
    "dgp_r1":          (["dgp_r1"], lambda r, o: figure_12panel(r["dgp_r1"], o)),
    "dgp_r3":          (["dgp_r3"], lambda r, o: figure_12panel(r["dgp_r3"], o)),
    "dgp_r0":          (["dgp_r0"], lambda r, o: figure_r0(r["dgp_r0"], o)),
    "ushape":          (["ushape"], lambda r, o: figure_ushape(r["ushape"], o)),
    "violation":       (["main", "violation"],
                        lambda r, o: _violation().render(r["violation"], r["main"], o)),
    **{f"method_{a}": (["main", "method"],
                       lambda r, o, a=a: figure_12panel(
                           _concat(r["main"], r["method"]), o, ABLATIONS[a]["methods"],
                           ABLATIONS[a].get("legend_ncol"), ABLATIONS[a].get("styles")))
       for a in ABLATIONS},
}


def _violation():
    from celeba import violation      # imports this module: loaded on first use
    return violation


def render(names: List[str], runs_dir: Path, fig_dir: Path) -> None:
    cache: Dict[str, Dict[str, pd.DataFrame]] = {}
    for name in names:
        needed, draw = FIGURES[name]
        for e in needed:
            if e not in cache:
                cache[e] = load_results(runs_dir, e)
        missing = [e for e in needed
                   if set(C.EXPERIMENTS[e]["sweeps"]) - set(cache[e])]
        if missing:
            print(f"  skip figure {name}: no results for {missing} in {runs_dir}")
            continue
        draw(cache, fig_dir / f"{name}.pdf")
    write_summary(runs_dir, fig_dir)


def write_summary(runs_dir: Path, fig_dir: Path) -> None:
    """summary.md (and dgp_r0_table.tex) from every experiment found on disk."""
    from celeba.experiment import load_ground_truth
    md = ["# CelebA benchmark: summary tables", ""]
    for exp, title, methods in [
            ("main", "Main setting (Figure 3, dgp.pdf)", MAIN_METHODS),
            ("model_k5", "SAE with k = 5 (model_k5.pdf)", MAIN_METHODS),
            ("model_precode", "Pre-activations z_pre (model_precode.pdf)", MAIN_METHODS),
            ("replica", "Replica SAE (replica_k20.pdf)",
             {**MAIN_METHODS, "NEXIS (forward + interleaved backward)":
              "NEXIS, interleaved instead of terminal backward step"}),
            ("dgp_r1", "r = 1 modifier (dgp_r1.pdf)", MAIN_METHODS),
            ("dgp_r3", "r = 3 modifiers (dgp_r3.pdf)", MAIN_METHODS)]:
        res = load_results(runs_dir, exp)
        if set(res) >= set(C.EXPERIMENTS[exp]["sweeps"]):
            md += [f"## {title}", ""] + table_12panel(res, methods)
            if exp.startswith("dgp_r") or exp == "main":
                dgp = C.EXPERIMENTS[exp]["dgp"]
                mods = [a for a, g in zip(dgp["attrs"], dgp["gammas"]) if g != 0]
                if "selected" in res["n"]:
                    md += ["Per-modifier recall of NEXIS:", ""] + per_modifier_table(
                        res, load_ground_truth(20, False), mods)
    # size of the forward selection S~ and cost of the terminal backward step (Appendix B)
    runs = [d[d.method == "NEXIS"] for e in ("main", "model_k5", "model_precode")
            for d in load_results(runs_dir, e).values()]
    runs = [d for d in runs if "n_forward" in d]
    if runs:
        d = pd.concat(runs, ignore_index=True)
        md += ["## Terminal backward step: |S~| and number of tests", "",
               f"NEXIS runs on the main, k = 5 and Z_pre dictionaries: {len(d)}. |S~| median "
               f"{d.n_forward.median():g}, max {d.n_forward.max():g}; terminal tests median "
               f"{d.terminal_tests.median():g}, max {d.terminal_tests.max():g}.", ""]
    main, method = load_results(runs_dir, "main"), load_results(runs_dir, "method")
    if set(main) >= {"effect", "n"} and set(method) >= {"effect", "n"}:
        both = _concat(main, method)
        for a, g in ABLATIONS.items():
            md += [f"## NEXIS ablation: {a} (method_{a}.pdf)", ""] + table_12panel(both, g["methods"])
    r0 = load_results(runs_dir, "dgp_r0")
    if "n" in r0:
        tmd, tex = table_r0(r0)
        md += ["## r = 0 (dgp_r0.pdf)", ""] + tmd + [""]
        fig_dir.mkdir(parents=True, exist_ok=True)
        (fig_dir / "dgp_r0_table.tex").write_text("\n".join(tex) + "\n")
    ush = load_results(runs_dir, "ushape")
    if set(ush) >= {"effect", "n"}:
        md += ["## U-shape DGP (ushape.pdf)", ""] + table_ushape(ush)
        if set(main) >= {"effect", "n"} and set(method) >= {"effect", "n"}:
            tex = table_test_comparison(_concat(main, method), ush)
            fig_dir.mkdir(parents=True, exist_ok=True)
            (fig_dir / "test_comparison.tex").write_text("\n".join(tex) + "\n")
            md += ["Test comparison, linear CATE (A) vs U-shape (B): test_comparison.tex", ""]
    md += _violation().summary_lines(runs_dir, fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    (fig_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(f"  tables -> {fig_dir / 'summary.md'}")
