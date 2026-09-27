"""Controlled violation of Principal Alignment (Appendix C.3): figures and tables.

The `violation` experiment reruns the main setting with the Wearing_Hat principal
coordinate j1 split in two complementary halves (config.DGP_VIOLATION, scm.split_principal),
so the target is S* = {j1, j_new, j2}.  The `main` experiment, on the same draws, is the
no-split control (S* = {j1, j2}).

    violation.pdf         one row at eta = 5: precision, recall and IoU against n, and the
                          share of NEXIS runs that select each target coordinate
    violation_grid.pdf    the 4 x 3 layout of Appendix C
    violation_table.tex   precision, recall, IoU and |S_hat| at n = 2000 and n = 10,000
    summary.md            macro metrics, per-coordinate recall, both-halves rate
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from celeba.experiment import load_ground_truth, load_results
from celeba.figures import (APPENDIX_RC, ETA_MAIN, MAIN_METHODS, STYLES, _save, figure_12panel,
                            per_modifier_table, plot_sweep, table_12panel)

EXPERIMENT = "violation"
METHODS = {**MAIN_METHODS, "NEXIS (rho=0)": r"NEXIS, $\rho = 0$"}       # figure legends
MD_METHODS = {**MAIN_METHODS, "NEXIS (rho=0)": "NEXIS (rho = 0)"}          # summary.md
TABLE_N = (2000, 10000)
TABLE_METHODS = {"Marginal Testing": "Marginal", "Marginal Testing (FDR)": "Marginal (FDR)",
                 "Marginal Testing (FWER)": "Marginal (FWER)", "NEXIS": "NEXIS"}
COORD_COLORS = ("#08519c", "#6baed6", "#E69F00")      # j1, j_new, j2


def target_coords() -> Dict[str, int]:
    """{role: coordinate} of the split target: the two halves of j1, then j2."""
    spec = C.EXPERIMENTS[EXPERIMENT]
    dgp = spec["dgp"]
    coords = load_ground_truth(spec["k"], spec["replica"])
    m = np.load(C.codes_path(spec["k"], spec["replica"], spec["view"]), mmap_mode="r").shape[1]
    split = dgp["split"]["attr"]
    other = [a for a, g in zip(dgp["attrs"], dgp["gammas"]) if g != 0 and a != split]
    return {f"{split}, half U (j1)": coords[split],
            f"{split}, half 1 - U (j_new)": m,
            **{f"{a} (j2)": coords[a] for a in other}}


def _hits(d: pd.DataFrame, js: List[int], col: str = "selected") -> pd.Series:
    return d[col].apply(lambda sel: set(js) <= set(int(x) for x in sel))


def _eta5(res: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    return res["n"][res["n"].fixed_effect == ETA_MAIN]


def figure_violation(res: Dict[str, pd.DataFrame], ctrl: Dict[str, pd.DataFrame],
                     out: Path) -> None:
    """One row at eta = 5: precision | recall | IoU against n, and per-coordinate recall."""
    d, c = _eta5(res), _eta5(ctrl)
    methods = {k: v for k, v in METHODS.items() if k != "NEXIS (rho=0)"}
    roles = target_coords()
    with plt.rc_context(APPENDIX_RC):
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
        for ax, met in zip(axes[:3], ["precision", "recall", "iou"]):
            plot_sweep(d, "n", met, r"Sample size $n$", ax, methods)
            ref = c[c.method == "NEXIS"].groupby("n")[met].mean()
            ax.plot(ref.index.values, ref.values, color="gray", lw=1.5, ls=":", zorder=1,
                    label=r"NEXIS, no split ($S^\star = \{j_1, j_2\}$)")
        nx = d[d.method == "NEXIS"]
        names = [r"$j_1$ (half $U$)", r"$j_{\mathrm{new}}$ (half $1-U$)", r"$j_2$ (Eyeglasses)"]
        for (role, j), name, col in zip(roles.items(), names, COORD_COLORS):
            share = nx.assign(hit=_hits(nx, [j])).groupby("n").hit.mean()
            ax = axes[3]
            ax.plot(share.index.values, share.values, color=col, lw=2.0, marker="o", ms=3,
                    label=name)
        both = nx.assign(hit=_hits(nx, list(roles.values())[:2])).groupby("n").hit.mean()
        axes[3].plot(both.index.values, both.values, color="k", lw=1.2, ls="--",
                     label="both halves")
        axes[3].set_xscale("log")
        axes[3].set_ylim(-0.03, 1.03)
        axes[3].set_xlabel(r"Sample size $n$")
        axes[3].set_ylabel("Share of NEXIS runs selecting it")
        axes[3].grid(True, alpha=0.25)
        axes[3].legend(fontsize=10, frameon=False, loc="lower right")
        handles, labels = axes[0].get_legend_handles_labels()
        for ax in axes[:3]:
            ax.get_legend().remove()
        fig.tight_layout(rect=[0, 0.1, 1, 1])
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=12)
        fig.text(0.5, 1.0, rf"Split principal coordinate, $\eta={int(ETA_MAIN)}$, "
                 r"$S^\star = \{j_1, j_{\mathrm{new}}, j_2\}$", ha="center", va="bottom",
                 fontsize=13)
        _save(fig, out)


def table_tex(res: Dict[str, pd.DataFrame], ctrl: Dict[str, pd.DataFrame]) -> List[str]:
    """precision / recall / IoU / |S_hat| (seed means) at eta = 5 and n in TABLE_N."""
    d, c = _eta5(res), _eta5(ctrl)
    ns = [n for n in TABLE_N if (d.n == n).any()]
    tex = [r"\begin{tabular}{l" + "cccc" * len(ns) + "}", r"\toprule",
           " & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{$n = {n:,}$}}".replace(",", "{,}")
                              for n in ns) + r" \\",
           "".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(ns))),
           "Method" + r" & Prec. & Rec. & IoU & $|\widehat S|$" * len(ns) + r" \\",
           r"\midrule"]

    def row(df, m, lab):
        cells = [lab]
        for n in ns:
            g = df[(df.method == m) & (df.n == n)]
            cells += [f"{g.precision.mean():.2f}", f"{g.recall.mean():.2f}",
                      f"{g.iou.mean():.2f}", f"{g.n_selected.mean():.1f}"]
        return " & ".join(cells) + r" \\"

    tex += [row(d, m, lab) for m, lab in TABLE_METHODS.items()]
    tex += [r"\midrule", row(c, "NEXIS", r"NEXIS, no split ($S^\star = \{j_1, j_2\}$)")]
    return tex + [r"\bottomrule", r"\end{tabular}"]


def summary_lines(runs_dir: Path, fig_dir: Path) -> List[str]:
    res, ctrl = load_results(runs_dir, EXPERIMENT), load_results(runs_dir, "main")
    if not set(res) >= set(C.EXPERIMENTS[EXPERIMENT]["sweeps"]):
        return []
    roles = target_coords()
    j1, jn, j2 = roles.values()
    md = ["## Controlled violation of Principal Alignment (violation.pdf)", "",
          f"S* = {{j1, j_new, j2}} = {{{j1}, {jn}, {j2}}}: the Wearing_Hat principal {j1} is "
          f"split into {j1} (half U) and the appended {jn} (half 1 - U).", ""]
    md += table_12panel(res, MD_METHODS)
    md += ["Per-coordinate recall of NEXIS:", ""] + per_modifier_table(
        res, {r: j for r, j in roles.items()}, list(roles))
    lines = ["| n | " + " | ".join(f"{MD_METHODS[m]}: P / R / IoU / size" for m in
                                   ["Marginal Testing (FWER)", "NEXIS", "NEXIS (rho=0)"])
             + " | NEXIS: both halves | NEXIS: both halves in S~ | NEXIS no split: IoU |",
             "|---|---|---|---|---|---|---|"]
    d, c = _eta5(res), _eta5(ctrl) if "n" in ctrl else None
    for n, g in d.groupby("n"):
        cells = []
        for m in ["Marginal Testing (FWER)", "NEXIS", "NEXIS (rho=0)"]:
            h = g[g.method == m]
            cells.append(f"{h.precision.mean():.2f} / {h.recall.mean():.2f} / "
                         f"{h.iou.mean():.2f} / {h.n_selected.mean():.1f}" if len(h) else "-")
        nx = g[g.method == "NEXIS"]
        cells.append(f"{_hits(nx, [j1, jn]).mean():.2f}")
        cells.append(f"{_hits(nx, [j1, jn], 'forward').mean():.2f}"
                     if "forward" in nx else "-")
        if c is not None:
            r = c[(c.method == "NEXIS") & (c.n == n)]
            cells.append(f"{r.iou.mean():.2f}" if len(r) else "-")
        else:
            cells.append("-")
        lines.append(f"| {int(n)} | " + " | ".join(cells) + " |")
    md += [f"n sweep at eta = {int(ETA_MAIN)}, seed means. P / R / IoU / size: precision, "
           "recall, IoU and |S_hat|. Both halves: share of runs selecting j1 and j_new; in S~: "
           "before the terminal backward step.", ""] + lines + [""]
    if "n" in ctrl:
        fig_dir.mkdir(parents=True, exist_ok=True)
        (fig_dir / "violation_table.tex").write_text("\n".join(table_tex(res, ctrl)) + "\n")
        md += ["LaTeX table at eta = 5: violation_table.tex", ""]
    return md


def render(res: Dict[str, pd.DataFrame], ctrl: Dict[str, pd.DataFrame], out: Path) -> None:
    """violation.pdf (one row) and violation_grid.pdf (4 x 3) next to it."""
    figure_violation(res, ctrl, out)
    figure_12panel(res, out.with_name("violation_grid.pdf"), methods=METHODS)
