#!/usr/bin/env python3
"""
One-glance comparison of the CATE-equivalence tests for the appendix (NEXIS-v2).

  A  main benchmark: tau linear in two binary attributes, k=20 SAE codes
     (results/celeba/experiment_v2/k20/sae/)
  B  U-shape: tau an orthogonalised quadratic of the two true coordinates, zero
     covariance with each, k=20 SAE pre-activations
     (results/celeba/experiment_v2_ushape/k20/sae_precode/)

Both under NEXIS-v2 = nexis(rho=0.5, backward=False, terminal_filter=True) and the
calibrated 2·min PCM rule, 50 seeds per cell.  Writes to results/celeba/figures_v2/:

  test_story.pdf  IoU vs n at eta=5, panel A | panel B, v2 test-ablation styles
  test_story.md   table (n* for IoU >= 0.95 on the n grid at eta=5, IoU at n=2000,
                  eta=5) and a draft paragraph
  test_story.tex  the same table as a LaTeX tabular

    python src/apps/celeba/figure_test_story_v2.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent

import apps.celeba.figure_appendix as fa          # noqa: F401  (global rcParams)
from apps.celeba.visualize import plot_sweep, ABLATION_GROUPS_V2
from apps.celeba.figure_ushape_v2 import first_hit

RES = ROOT / "results/celeba"
OUT = RES / "figures_v2"
PANELS = {"A": ("Main benchmark (linear CATE)", RES / "experiment_v2/k20/sae"),
          "B": ("U-shape (zero covariance)", RES / "experiment_v2_ushape/k20/sae_precode")}
ETA, N = 5.0, 2000
METHODS = ABLATION_GROUPS_V2["test"]["methods"]


def load(base: Path):
    n = pd.read_parquet(base / "n_sweep.parquet")
    e = pd.read_parquet(base / "effect_sweep.parquet")
    return n[n.fixed_effect == ETA], e[(e.fixed_n == N) & (e.effect_scale == ETA)]


def main():
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


if __name__ == "__main__":
    main()
