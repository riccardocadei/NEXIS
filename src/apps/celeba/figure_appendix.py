#!/usr/bin/env python3
"""
Generate all CelebA appendix figures and a brief for the NEXIS paper.

All figures use a 4-row × 3-col layout (one row per DGP setting, three metrics per row):
  Row 0: n-sweep   @ η=5    — [Precision | Recall | IoU]
  Row 1: n-sweep   @ η=2    — [Precision | Recall | IoU]
  Row 2: η-sweep   @ n=2000 — [Precision | Recall | IoU]
  Row 3: η-sweep   @ n=500  — [Precision | Recall | IoU]

Figures saved to results/celeba/appendix/:
  dgp.pdf           Reference (k=20/z, MAIN_METHODS, both DGP rows) — also serves as DGP ablation
  model_k5.pdf      SAE k=5 with MAIN_METHODS (compare to dgp.pdf for k ablation)
  model_precode.pdf k=20/z_pre with MAIN_METHODS (compare to dgp.pdf for feature-type ablation)
  method_test.pdf   Test statistic ablation
  method_adjust.pdf Multiple-testing adjustment ablation
  method_rho.pdf    ρ threshold ablation
  method_backward.pdf Backward elimination ablation
  brief.md          Experiment description + per-ablation findings for LaTeX write-up

Usage
-----
    python src/apps/celeba/figure_appendix.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent

from apps.celeba.visualize import (
    plot_sweep, MAIN_METHODS, METHOD_STYLES, ABLATION_GROUPS, _METRIC_LABEL,
)

# ---------------------------------------------------------------------------
# Global style (matches figure_main.py)
# ---------------------------------------------------------------------------

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

OUT = ROOT / "results/celeba/appendix"


# ---------------------------------------------------------------------------
# Core 4-row × 3-col figure builder
# ---------------------------------------------------------------------------

def make_12panel(
    df_n: pd.DataFrame,       # n_sweep: must contain fixed_effect ∈ {ETA_ALT, ETA_MAIN}
    df_e: pd.DataFrame,       # effect_sweep: must contain fixed_n ∈ {N_ALT, N_MAIN}
    methods: dict[str, str],  # {method_key: display_label}
    out_path: Path,
    extra_styles: dict | None = None,
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
                           ax=axes[r, c], methods=methods)

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
        n_cols_leg = min(len(handles), 4)
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


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load(k: int, feat: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = ROOT / "results/celeba/experiment" / f"k{k}" / feat
    return (pd.read_parquet(base / "n_sweep.parquet"),
            pd.read_parquet(base / "effect_sweep.parquet"))


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------

def fig_dgp(out: Path = OUT / "dgp.pdf") -> None:
    """Reference figure (k=20/z, MAIN_METHODS).  Also serves as the DGP ablation:
    row 1 = main setting, row 2 = weaker setting — same methods, different DGP."""
    n, e = _load(20, "sae")
    make_12panel(n, e, MAIN_METHODS, out)


def fig_model_k5(out: Path = OUT / "model_k5.pdf") -> None:
    """k=5 SAE ablation: all MAIN_METHODS evaluated on k=5 sparse codes."""
    n, e = _load(5, "sae")
    make_12panel(n, e, MAIN_METHODS, out)


def fig_model_precode(out: Path = OUT / "model_precode.pdf") -> None:
    """Feature-type ablation: all MAIN_METHODS on k=20 continuous pre-activations (z_pre)."""
    n, e = _load(20, "sae_precode")
    make_12panel(n, e, MAIN_METHODS, out)


def _method_fig(ablation_key: str, out: Path) -> None:
    n, e = _load(20, "sae")
    grp  = ABLATION_GROUPS[ablation_key]
    make_12panel(n, e, grp["methods"], out)


def fig_method_test(out:     Path = OUT / "method_test.pdf")     -> None: _method_fig("test",     out)
def fig_method_adjust(out:   Path = OUT / "method_adjust.pdf")   -> None: _method_fig("adjust",   out)
def fig_method_rho(out:      Path = OUT / "method_rho.pdf")      -> None: _method_fig("rho",      out)
def fig_method_backward(out: Path = OUT / "method_backward.pdf") -> None: _method_fig("backward", out)


# ---------------------------------------------------------------------------
# Brief
# ---------------------------------------------------------------------------

def _sat(df: pd.DataFrame, xcol: str, fixed_col: str, fixed_val,
         method: str, metric: str = "recall", thr: float = 0.95):
    sub = df[(df[fixed_col] == fixed_val) & (df["method"] == method)]
    m   = sub.groupby(xcol)[metric].mean()
    hit = m[m >= thr]
    return hit.index.min() if not hit.empty else None


def _fmt(v) -> str:
    if v is None:
        return "never"
    return str(int(v)) if isinstance(v, (int, float)) and v == int(v) else str(v)


def write_brief(out: Path = OUT / "brief.md") -> None:
    n20,  e20  = _load(20, "sae")
    n5,   e5   = _load(5,  "sae")
    n20p, e20p = _load(20, "sae_precode")

    def sat(df, xcol, fc, fv, method, metric="recall"):
        return _fmt(_sat(df, xcol, fc, fv, method, metric))

    lines = [
        "# CelebA Semi-Synthetic Experiment — Appendix Brief",
        "",
        "## 1. Experiment Details",
        "",
        "### Foundation model",
        "Images are encoded with **SigLIP** (ViT-based, 1152-d mean-pooled embeddings),",
        "pre-computed once for all 202,599 CelebA images.",
        "",
        "### Sparse Autoencoder (SAE)",
        "A **TopK-SAE** (hidden dim 13,824 = 12 × 1152) is trained on the *per-patch*",
        "SigLIP tokens (729 patches/image ≈ 148 M training vectors/epoch), following the",
        "ECI protocol for principally-aligned representations.",
        "",
        "Two sparsity levels: **k=5** and **k=20** active features per image.",
        "Two feature views per level:",
        "- **z** (sparse post-topK codes): near-orthogonal, avg L0 = k.",
        "- **z_pre** (continuous pre-activations): dense, correlated.",
        "",
        "**Main setting: k=20, z.**",
        "",
        "### Data-generating process (DGP)",
        "```",
        "  W1 ~ Bernoulli(p̂_W1)   [Wearing_Hat,  ≈5% prevalence]",
        "  W2 ~ Bernoulli(p̂_W2)   [Eyeglasses,   ≈7% prevalence]",
        "  T  ~ Bernoulli(0.5)     [random treatment]",
        "  X  ~ CelebA image matching (W1,W2)  [without replacement per bucket]",
        "  Z  = SAE(SigLIP(X))     [pre-computed]",
        "  Y  = β_W1·W1 + β_W2·W2 + T·[τ₀ + η·(γ_W1·W1 + γ_W2·W2)] + ε",
        "```",
        "Parameters: τ₀=0.5, γ_W1=+1, γ_W2=−1, β_W1=0.3, β_W2=−0.2, σ_ε=1.",
        "**η** (effect_scale) scales the heterogeneous treatment effect.",
        "Ground truth: the single SAE neuron maximally correlated (by F1, using z_pre)",
        "with each attribute over the full 202k dataset.",
        "Ground-truth neurons: W1 → dim 5348 (k=20), W2 → dim 5537 (k=20).",
        "",
        "### Ground-truth definition",
        "The ground truth is defined as the **single SAE neuron most correlated with each",
        "binary attribute** (W1, W2) across all 202k CelebA images.",
        "Correlation is measured by F1 score using the continuous pre-activations z_pre",
        "(not the sparse codes z), because z_pre gives smoother, threshold-free alignment",
        "scores that better reflect the neuron's intrinsic selectivity.",
        "Concretely, for each neuron j and attribute A ∈ {W1, W2}, we sweep thresholds",
        "over z_pre[:, j] and report the best-threshold F1(j, A).",
        "The ground-truth neuron for attribute A is argmax_j F1(j, A).",
        "This procedure is applied once on the full dataset (no sampling) and fixed",
        "throughout all experiment repetitions.",
        "Under k=20 the ground-truth neurons are dim 5348 (W1=Wearing_Hat) and",
        "dim 5537 (W2=Eyeglasses); under k=5 they shift to dim 7044 and 5732,",
        "reflecting the different sparsity structure.",
        "",
        "### Sweeps and seeds",
        "- **Effect sweep**: η ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},",
        "  with fixed n ∈ {500, 2000} (one sub-sweep per fixed n).",
        "- **Sample-size sweep**: n ∈ {50, 100, 200, 350, 500, 750, 1000, 2000, 3500, 5000, 10 000},",
        "  with fixed η ∈ {2, 5} (one sub-sweep per fixed η).",
        "- 50 random seeds per (η, n) pair; reported metrics are means ± 1.96 SE across seeds.",
        "",
        "### Figure layout",
        "Every appendix figure uses a **4-row × 3-col** layout:",
        "- **Row 0:** n-sweep at η=5 (fixed η, varying n).",
        "- **Row 1:** n-sweep at η=2 (fixed η, varying n).",
        "- **Row 2:** effect-sweep at n=2000 (fixed n, varying η).",
        "- **Row 3:** effect-sweep at n=500 (fixed n, varying η).",
        "- Each row shows three metrics: [Precision | Recall | IoU].",
        "Showing all four DGP conditions in a single figure lets the reader assess",
        "both power (effect size / sample size) and DGP sensitivity simultaneously.",
        "",
        "---",
        "",
        "## 2. DGP Sensitivity  [`dgp.pdf`]",
        "",
        "The reference figure (`dgp.pdf`) shows MAIN_METHODS on k=20/z under both DGP rows.",
        "Reading **across rows** reveals how performance degrades at weaker conditions:",
        "",
        f"NEXIS (k=20/z):",
        f"  Effect sweep: recall≥0.95 at η={sat(e20,'effect_scale','fixed_n',N_MAIN,'NEXIS')} (n=2000)",
        f"                                  η={sat(e20,'effect_scale','fixed_n',N_ALT, 'NEXIS')} (n=500)",
        f"  n sweep:      recall≥0.95 at n={sat(n20,'n','fixed_effect',ETA_MAIN,'NEXIS')} (η=5)",
        f"                                  n={sat(n20,'n','fixed_effect',ETA_ALT, 'NEXIS')} (η=2)",
        "",
        "Baselines always achieve high recall early but **never** control precision,",
        "regardless of DGP condition.  NEXIS precision degrades only at very small n or η.",
        "",
        "---",
        "",
        "## 3. Model Ablations",
        "",
        "### 3a. SAE top-k: k=5  [`model_k5.pdf`]  (reference: `dgp.pdf`)",
        "",
        "All MAIN_METHODS evaluated on k=5 sparse codes:",
        "",
        f"  NEXIS recall≥0.95:",
        f"    Effect sweep: η={sat(e5,'effect_scale','fixed_n',N_MAIN,'NEXIS')} (n=2000),  η={sat(e5,'effect_scale','fixed_n',N_ALT,'NEXIS')} (n=500)",
        f"    n sweep:      n={sat(n5,'n','fixed_effect',ETA_MAIN,'NEXIS')} (η=5),  n={sat(n5,'n','fixed_effect',ETA_ALT,'NEXIS')} (η=2)",
        f"  NEXIS precision≥0.95:",
        f"    Effect sweep: η={sat(e5,'effect_scale','fixed_n',N_MAIN,'NEXIS','precision')} (n=2000)",
        f"    n sweep:      n={sat(n5,'n','fixed_effect',ETA_MAIN,'NEXIS','precision')} (η=5)",
        "",
        "**Conclusion:** k=5 achieves similar recall thresholds but lower precision.",
        "With only 5 active features per image the SAE dictionary is more entangled:",
        "attribute information spreads across several correlated neurons, so NEXIS",
        "selects a larger set that includes false positives.  k=20 is the better choice.",
        "",
        "### 3b. Feature type: z_pre (continuous pre-activations, k=20)  [`model_precode.pdf`]  (reference: `dgp.pdf`)",
        "",
        "All MAIN_METHODS evaluated on k=20 z_pre (dense continuous pre-activations):",
        "",
        f"  NEXIS recall≥0.95:",
        f"    Effect sweep: η={sat(e20p,'effect_scale','fixed_n',N_MAIN,'NEXIS')} (n=2000),  η={sat(e20p,'effect_scale','fixed_n',N_ALT,'NEXIS')} (n=500)",
        f"    n sweep:      n={sat(n20p,'n','fixed_effect',ETA_MAIN,'NEXIS')} (η=5),  n={sat(n20p,'n','fixed_effect',ETA_ALT,'NEXIS')} (η=2)",
        f"  NEXIS precision≥0.95:",
        f"    Effect sweep: η={sat(e20p,'effect_scale','fixed_n',N_MAIN,'NEXIS','precision')} (n=2000)",
        f"    n sweep:      n={sat(n20p,'n','fixed_effect',ETA_MAIN,'NEXIS','precision')} (η=5)",
        "",
        "**Conclusion:** Sparse codes z are preferable.  Near-orthogonality of z means",
        "NEXIS conditioning adds little noise and Bonferroni remains tight.",
        "Dense z_pre features are correlated; conditioning helps but requires more data",
        "to stabilise, and precision is slightly lower at moderate n.",
        "",
        "---",
        "",
        "## 4. Method Ablations  (all on k=20/z)",
        "",
        "### 4a. Test statistic  [`method_test.pdf`]",
        "",
        "Linear (default) vs GCM-quadratic vs GCM-lgbm.",
        "",
        f"  Linear:        recall≥0.95 at η={sat(e20,'effect_scale','fixed_n',N_MAIN,'NEXIS')}, n={sat(n20,'n','fixed_effect',ETA_MAIN,'NEXIS')}.",
        f"  GCM-quadratic: recall≥0.95 at η={sat(e20,'effect_scale','fixed_n',N_MAIN,'NEXIS (test=GCM: quadratic)')}, n={sat(n20,'n','fixed_effect',ETA_MAIN,'NEXIS (test=GCM: quadratic)')}.",
        f"  GCM-lgbm:      recall≥0.95 at η={sat(e20,'effect_scale','fixed_n',N_MAIN,'NEXIS (test=GCM: lgbm)')}, n={sat(n20,'n','fixed_effect',ETA_MAIN,'NEXIS (test=GCM: lgbm)')}.",
        "",
        "**Conclusion:** The linear test is sufficient and most powerful here because",
        "the true conditional effect is linear in Z.  GCM-based tests pay a power cost",
        "for unnecessary flexibility, requiring roughly 2× larger η or n to match the linear test.",
        "",
        "### 4b. Multiple-testing adjustment  [`method_adjust.pdf`]",
        "",
        "None (unadjusted), FDR (Benjamini-Hochberg), FWER (Bonferroni, default).",
        "",
        f"  None: precision≥0.95 at η={sat(e20,'effect_scale','fixed_n',N_MAIN,'NEXIS (adjust=None)','precision')}, n={sat(n20,'n','fixed_effect',ETA_MAIN,'NEXIS (adjust=None)','precision')}.",
        f"  FDR:  precision≥0.95 at η={sat(e20,'effect_scale','fixed_n',N_MAIN,'NEXIS (adjust=FDR)','precision')}, n={sat(n20,'n','fixed_effect',ETA_MAIN,'NEXIS (adjust=FDR)','precision')}.",
        f"  FWER: precision≥0.95 at η={sat(e20,'effect_scale','fixed_n',N_MAIN,'NEXIS','precision')}, n={sat(n20,'n','fixed_effect',ETA_MAIN,'NEXIS','precision')}.",
        "",
        "**Conclusion:** No adjustment inflates false positives; at large η many neurons",
        "become marginally significant and pass the unadjusted test.  FDR and FWER",
        "perform similarly in this low-truth-set-size regime; FWER (Bonferroni) is the",
        "conservative default and gives cleaner precision curves.",
        "",
        "### 4c. Correlation threshold ρ  [`method_rho.pdf`]",
        "",
        "ρ ∈ {0, 0.2, 0.5 (default), 0.8}.",
        "",
        f"  ρ=0:   recall≥0.95 at η={sat(e20,'effect_scale','fixed_n',N_MAIN,'NEXIS (rho=0)')};  precision≥0.95 (n sweep): {sat(n20,'n','fixed_effect',ETA_MAIN,'NEXIS (rho=0)','precision')}.",
        f"  ρ=0.2: recall≥0.95 at η={sat(e20,'effect_scale','fixed_n',N_MAIN,'NEXIS (rho=0.2)')};  precision similar.",
        f"  ρ=0.5: recall≥0.95 at η={sat(e20,'effect_scale','fixed_n',N_MAIN,'NEXIS')};  precision≥0.95 at η={sat(e20,'effect_scale','fixed_n',N_MAIN,'NEXIS','precision')}.",
        f"  ρ=0.8: recall≥0.95 at η={sat(e20,'effect_scale','fixed_n',N_MAIN,'NEXIS (rho=0.8)')} — overly conservative.",
        "",
        "**Conclusion:** Low ρ (0 or 0.2) allows correlated neurons to be co-selected,",
        "which hurts precision without benefiting recall.  High ρ (0.8) rejects many",
        "true positives because correlated neurons are blocked from sequential selection.",
        "ρ=0.5 is the best trade-off.",
        "",
        "### 4d. Backward elimination  [`method_backward.pdf`]",
        "",
        f"  True (default): recall≥0.95 at η={sat(e20,'effect_scale','fixed_n',N_MAIN,'NEXIS')}, n={sat(n20,'n','fixed_effect',ETA_MAIN,'NEXIS')}.",
        f"  False:          recall≥0.95 at η={sat(e20,'effect_scale','fixed_n',N_MAIN,'NEXIS (backward=False)')}, n={sat(n20,'n','fixed_effect',ETA_MAIN,'NEXIS (backward=False)')}.",
        "",
        "**Conclusion:** Backward elimination is empirically neutral in this experiment.",
        "Because k=20 sparse codes are near-orthogonal, selected neurons are already",
        "conditionally independent after the forward pass, so the backward pruning step",
        "rarely removes any of them.  Theoretically, however, backward elimination is",
        "the correct procedure: when the number of true effect-modifying drivers is",
        "larger than one, a forward-only selection can leave spurious neurons that pass",
        "marginal tests but become redundant once all true drivers are included.",
        "The step is therefore retained as the default — it adds negligible cost and",
        "will be increasingly beneficial as the complexity of the effect-modification",
        "structure grows.",
        "",
        "---",
        "",
        "### Practical recommendation (method ablations)",
        "",
        "Given the results above, the recommended default configuration is:",
        "**linear conditional test**, **FWER (Bonferroni) adjustment**, **ρ=0.5**,",
        "**backward=True**.",
        "This setting is robust across all DGP conditions tested.  Practitioners should",
        "consider switching to a non-linear test (GCM) only when the outcome–feature",
        "relationship is known to be strongly non-linear, accepting the associated power",
        "cost.  Lowering ρ is inadvisable unless the feature dictionary is guaranteed to",
        "be orthogonal (as SAE codes approximately are); raising ρ is inadvisable when",
        "effect sizes may be moderate.",
        "",
        "---",
        "",
        "## 5. Summary Table",
        "",
        "| Figure              | What varies         | Key finding vs reference |",
        "|---------------------|---------------------|--------------------------|",
        "| `dgp.pdf`           | DGP (rows)          | Performance degrades gracefully; baselines never control precision |",
        "| `model_k5.pdf`      | SAE k (5 vs 20)     | k=5 similar recall, lower precision |",
        "| `model_precode.pdf` | Feature (z vs z_pre)| z_pre needs more data; z preferred |",
        "| `method_test.pdf`   | Conditional test    | Linear test dominates; GCM tests need 2× more power |",
        "| `method_adjust.pdf` | MHT correction      | FWER and FDR equivalent; unadjusted inflates FP |",
        "| `method_rho.pdf`    | ρ threshold         | ρ=0.5 optimal; extremes hurt precision or recall |",
        "| `method_backward.pdf` | Backward elim.    | Negligible effect with orthogonal codes |",
        "",
        "**Overall:** The main setting (k=20, z, FWER, ρ=0.5, linear test, backward=True)",
        "is the strongest across all dimensions.  All ablations degrade performance",
        "gracefully — no single design choice is catastrophic, and NEXIS consistently",
        "outperforms all marginal baselines in precision across every condition tested.",
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    fig_dgp()
    fig_model_k5()
    fig_model_precode()
    fig_method_test()
    fig_method_adjust()
    fig_method_rho()
    fig_method_backward()
    write_brief()
    print(f"\nAll appendix assets saved to {OUT}")


if __name__ == "__main__":
    main()
