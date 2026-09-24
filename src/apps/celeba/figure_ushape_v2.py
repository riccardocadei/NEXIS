#!/usr/bin/env python3
"""
GCM-blind U-shape DGP (rebuttal panel B) under NEXIS-v2 and the calibrated PCM.

Reads results/celeba/experiment_v2_ushape/k20/sae_precode/{n,effect}_sweep.parquet
(scripts/celeba/submit_experiment_v2_ushape.sh) and, for comparison, the rebuttal run
results/celeba/experiment_ushape/ (published NEXIS, invalid "crossfit" PCM rule), and
writes to results/celeba/figures_v2/:

  ushape.pdf  2 rows (n @ eta=5, eta @ n=2000) x [precision | recall | IoU | false
              discoveries], the test-ablation lines of the v2 appendix; styles from
              visualize.py, layout as figure_appendix.make_12panel
  ushape.md   per test (and marginal testing, FWER): n* and eta* (first grid value with seed-mean IoU >= 0.95),
              precision / recall / IoU / false discoveries at n=2000, eta=5, the share
              of runs with at least one false discovery, and the rebuttal-run n*

    python src/apps/celeba/figure_ushape_v2.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent

import apps.celeba.figure_appendix as fa          # noqa: F401  (global rcParams)
from apps.celeba.visualize import plot_sweep, ABLATION_GROUPS_V2, MAIN_METHODS_V2

RES = ROOT / "results/celeba"
V2 = RES / "experiment_v2_ushape/k20/sae_precode"
PUB = RES / "experiment_ushape/k20/sae_precode"
ETA, N = 5.0, 2000
METHODS = {"Marginal Testing (FWER)": MAIN_METHODS_V2["Marginal Testing (FWER)"],
           **ABLATION_GROUPS_V2["test"]["methods"]}
# the figure shows the test-ablation lines only (marginal testing makes thousands of
# false discoveries here and would flatten the false-discovery panel); the table has all
FIG_METHODS = ABLATION_GROUPS_V2["test"]["methods"]
# rebuttal-run method with the same test (published NEXIS, crossfit PCM)
PUB_OF = {"Marginal Testing (FWER)": "Marginal Testing (FWER)", "NEXIS-v2": "NEXIS",
          **{f"NEXIS-v2 (test={t})": f"NEXIS (test={t})"
             for t in ("GCM: quadratic", "GCM: lgbm", "PCM: quadratic", "PCM: lgbm")}}


def load(base: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    n, e = (pd.read_parquet(base / f) for f in ("n_sweep.parquet", "effect_sweep.parquet"))
    return n[n.fixed_effect == ETA], e[e.fixed_n == N]


def first_hit(df: pd.DataFrame, x: str, method: str, metric="iou", thr=0.95) -> str:
    m = df[df.method == method].groupby(x)[metric].mean()
    hit = m[m >= thr]
    if hit.empty:
        return "never"
    v = hit.index.min()
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def figure(dn: pd.DataFrame, de: pd.DataFrame, out: Path) -> None:
    metrics = ["precision", "recall", "iou", "fp"]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    rows = [(dn, "n", r"Sample size $n$", rf"varying $n$,  $\eta={int(ETA)}$ fixed"),
            (de, "effect_scale", r"Effect size $\eta$", rf"varying $\eta$,  $n={N}$ fixed")]
    for r, (d, x, xl, _) in enumerate(rows):
        for c, met in enumerate(metrics):
            ax = plot_sweep(d, x, met, xlabel=xl, ax=axes[r, c], methods=FIG_METHODS)
            if met == "fp":                  # plot_sweep fixes [0, 1]; counts need more
                top = d[d.method.isin(FIG_METHODS)].groupby(["method", x]).fp.mean().max()
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
        fig.text(xc, ytop, title, ha="center", va="bottom", fontsize=fa.LABEL_SIZE)
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def table(dn, de, pub) -> list[str]:
    at = de[de.effect_scale == ETA]
    out = ["| test | n* (eta=5) | eta* (n=2000) | precision | recall | IoU | false disc. "
           "| runs with >=1 false disc. | s/run | rebuttal n* (eta=5) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for m, lab in METHODS.items():
        g = at[at.method == m]
        f = lambda c: f"{g[c].mean():.2f} ± {g[c].sem():.2f}"
        pn = first_hit(pub[0], "n", PUB_OF[m]) if pub is not None else "n/a"
        out.append(f"| {lab} | {first_hit(dn, 'n', m)} | {first_hit(de, 'effect_scale', m)} "
                   f"| {f('precision')} | {f('recall')} | {f('iou')} | {f('fp')} "
                   f"| {(g.fp > 0).mean():.2f} | {g.time_s.mean():.0f} | {pn} |")
    return out


def fp_grid(dn, de) -> list[str]:
    out = []
    for d, x, name in ((dn, "n", f"n sweep at eta={int(ETA)}"),
                       (de, "effect_scale", f"eta sweep at n={N}")):
        t = d.groupby(["method", x]).fp.mean().unstack(0)[list(METHODS)].rename(columns=METHODS)
        out += [f"Mean false discoveries per run, {name}:", "",
                t.round(2).to_markdown(), ""]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=RES / "figures_v2")
    a = p.parse_args()
    out = a.out_dir if a.out_dir.is_absolute() else ROOT / a.out_dir
    dn, de = load(V2)
    pub = load(PUB) if PUB.exists() else None
    figure(dn, de, out / "ushape.pdf")
    md = ["# U-shape (GCM-blind) DGP under NEXIS-v2 and the calibrated PCM", "",
          "tau = 0.5 + eta (g(Z^5348) - g(Z^5537)), g the orthogonalised quadratic of "
          "scm._ortho_quadratic_map (zero mean and zero covariance with Z^j on the full "
          "pool); k=20 SigLIP SAE pre-activations, m = 9,216, |S*| = 2, alpha = 0.05, "
          "50 seeds per cell. NEXIS-v2 = nexis(rho=0.5, backward=False, "
          "terminal_filter=True); PCM combines the two split directions by 2·min(p1, p2). "
          "n* / eta*: first grid value where the seed-mean IoU reaches 0.95. Metrics at "
          "n=2000, eta=5 (mean ± SE). The last column is the rebuttal run (published NEXIS, "
          "invalid crossfit PCM rule, results/celeba/experiment_ushape/).", ""]
    md += table(dn, de, pub) + [""] + fp_grid(dn, de)
    (out / "ushape.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"Saved -> {out / 'ushape.md'}")


if __name__ == "__main__":
    main()
