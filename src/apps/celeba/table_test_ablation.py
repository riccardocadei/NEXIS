#!/usr/bin/env python3
"""
CATE-equivalence test ablation → Markdown + LaTeX tables.

Summarises the sweeps produced by run_experiment.py for the four (five, with the
PCM) instantiations of the test in Equation (8), on two DGPs:

  attr            τ linear in the binary attributes W1, W2 (the published benchmark)
  ortho_quadratic τ an orthogonalised quadratic of the two ground-truth coordinates,
                  built so Cov(τ, Z^j) = 0 in the population — a conditional-mean
                  alternative that no linear-functional test can see

Reported per test:
  n*  — smallest n on the grid whose mean recall reaches `thr` (η fixed)
  η*  — smallest effect size whose mean recall reaches `thr` (n fixed)
  Precision / Recall / IoU at the reference operating points, ± Monte-Carlo s.e.
  median wall-clock per run

Usage
-----
    python src/apps/celeba/table_test_ablation.py
    python src/apps/celeba/table_test_ablation.py --dgp ortho_quadratic
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent

TESTS: dict[str, str] = {
    "NEXIS":                       "linear (default)",
    "NEXIS (test=GCM: quadratic)": "GCM: quadratic",
    "NEXIS (test=GCM: lgbm)":      "GCM: lgbm",
    "NEXIS (test=PCM: quadratic)": "PCM: quadratic",
    "NEXIS (test=PCM: lgbm)":      "PCM: lgbm",
}
BASELINE = "Marginal Testing (FWER)"

#: ALL_METHODS label -> bench_tests.py label
_BENCH_KEY = {
    "NEXIS": "linear",
    "NEXIS (test=GCM: quadratic)": "GCM: quadratic",
    "NEXIS (test=GCM: lgbm)": "GCM: lgbm",
    "NEXIS (test=PCM: quadratic)": "PCM: quadratic",
    "NEXIS (test=PCM: lgbm)": "PCM: lgbm",
}

THR = 0.95
#: Metric the n*/eta* saturation thresholds are read on.  IoU rather than recall: a
#: recall threshold rewards a test that fires on everything, which is exactly the
#: failure mode the paper is about.  Under recall the linear test looks 4x more
#: sample-efficient than the assumption-lean ones at eta=5; under IoU that gap is
#: entirely false positives and disappears.
THR_METRIC = "iou"


def _mean_se(sub: pd.DataFrame, metric: str) -> tuple[float, float]:
    v = sub[metric].to_numpy(dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0


def _threshold(df: pd.DataFrame, xcol: str, fixed_col: str, fixed_val: float,
               method: str, thr: float = THR, metric: str = THR_METRIC):
    sub = df[(df[fixed_col] == fixed_val) & (df["method"] == method)]
    if sub.empty:
        return None
    m = sub.groupby(xcol)[metric].mean()
    hit = m[m >= thr]
    return None if hit.empty else hit.index.min()


def _fmt_thr(v) -> str:
    if v is None:
        return "—"
    return str(int(v)) if float(v) == int(v) else f"{v:g}"


def _at(df: pd.DataFrame, fixed_col: str, fixed_val, xcol: str, xval, method: str):
    return df[(df[fixed_col] == fixed_val) & (df[xcol] == xval) & (df["method"] == method)]


def build(df_n: pd.DataFrame, df_e: pd.DataFrame, methods: dict[str, str],
          n_points, e_points, ref_n: float, ref_e: float) -> pd.DataFrame:
    rows = []
    for key, label in methods.items():
        if key not in set(df_n["method"]) | set(df_e["method"]):
            continue
        r: dict[str, object] = {"test": label}
        for eta in n_points:
            r[f"n*(η={eta:g})"] = _fmt_thr(_threshold(df_n, "n", "fixed_effect", eta, key))
        for nn in e_points:
            r[f"η*(n={nn:g})"] = _fmt_thr(_threshold(df_e, "effect_scale", "fixed_n", nn, key))

        sub = _at(df_n, "fixed_effect", ref_e, "n", ref_n, key)
        if sub.empty:   # reference point may live in the effect sweep instead
            sub = _at(df_e, "fixed_n", ref_n, "effect_scale", ref_e, key)
        for metric, short in [("precision", "Prec"), ("recall", "Rec"), ("iou", "IoU")]:
            mu, se = _mean_se(sub, metric)
            r[short] = "—" if np.isnan(mu) else f"{mu:.2f}±{se:.2f}"
        # Cost at the reference point, NOT a sweep median: at small n the tests with
        # less power select nothing and terminate after one round, so a median over the
        # whole grid rewards weakness and makes the GCM look ~5x faster than linear.
        mu_t, _ = _mean_se(sub, "time_s")
        r["s/run"] = "—" if np.isnan(mu_t) else f"{mu_t:.1f}"
        rows.append(r)
    return pd.DataFrame(rows)


def to_markdown(tab: pd.DataFrame) -> str:
    cols = list(tab.columns)
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in tab.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def to_latex(tab: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(tab.columns)

    def _hdr(c: str) -> str:
        # "n*(η=2)" → "$n^\star(\eta{=}2)$";  "FWER (n=500)" → "FWER ($n{=}500$)"
        m = re.fullmatch(r"(n|η)\*\((n|η)=([0-9.]+)\)", c)
        if m:
            sym = {"n": "n", "η": r"\eta"}
            return (rf"${sym[m.group(1)]}^\star({sym[m.group(2)]}{{=}}{m.group(3)})$")
        m = re.fullmatch(r"FWER \(n=([0-9]+)\)", c)
        if m:
            return rf"FWER ($n{{=}}{m.group(1)}$)"
        return c.replace("±", r"$\pm$")

    hdr = [_hdr(c) for c in cols]
    lines = [r"\begin{table}[h]", r"\centering", r"\small",
             r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(hdr) + r" \\", r"\midrule"]
    for _, r in tab.iterrows():
        cells = [str(r[c]).replace("±", r"$\pm$").replace("—", r"---") for c in cols]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def build_paper_table(k: int = 20) -> str:
    """The single two-panel table to drop into the appendix test-ablation section.

    Panel A: the published DGP (τ linear in the binary attributes, sparse codes z) —
             shows that adding the PCM leaves the recommended default untouched.
    Panel B: the GCM-blind DGP (τ an orthogonalised quadratic of the two ground-truth
             coordinates, continuous z_pre) — shows the PCM is the only instantiation
             in the suite that is consistent there.
    """
    a_n, a_e = (pd.read_parquet(ROOT / f"results/celeba/experiment/k{k}/sae" / f)
                for f in ("n_sweep.parquet", "effect_sweep.parquet"))
    b_n, b_e = (pd.read_parquet(
        ROOT / f"results/celeba/experiment_ushape/k{k}/sae_precode" / f)
        for f in ("n_sweep.parquet", "effect_sweep.parquet"))

    def cell(df, metric, fixed_col, fixed_val, xcol, xval, key, fmt="{:.2f}"):
        sub = _at(df, fixed_col, fixed_val, xcol, xval, key)
        mu, _ = _mean_se(sub, metric)
        return "---" if np.isnan(mu) else fmt.format(mu)

    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\caption{\small \textbf{CATE-equivalence test ablation}, including the "
        r"modified Projected Covariance Measure. CelebA, $k=20$ SigLIP SAE, "
        r"$|\mathcal{S}^\star|=2$, $\alpha=0.05$, 50 seeds. "
        r"$n^\star$: smallest sample size (at $\eta=2$ in A, $\eta=5$ in B) reaching "
        r"mean IoU $\ge 0.95$; --- never on the grid. IoU at $n=2000$, $\eta=5$. \emph{A}: the DGP of Figure~\ref{fig:power_paradox_synth}, $\tau$ "
        r"linear in the binary attributes (sparse codes). The linear test remains the "
        r"best choice and the PCM matches the GCM. \emph{B}: $\tau$ an orthogonalised "
        r"quadratic of the two principal coordinates (pre-activations), calibrated so "
        r"that $\mathrm{Cov}(\tau, Z^j)=0$ in the population while $\E[\tau \mid Z^j]$ "
        r"still varies. The linear test and both GCM variants are inconsistent there by "
        r"construction and never recover a single coordinate; the PCM recovers "
        r"$\mathcal{S}^\star$ exactly.}",
        r"\label{tab:celeba:test_ablation}",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"& \multicolumn{2}{c}{\textbf{A.} $\tau$ linear in $W_1, W_2$}"
        r" & \multicolumn{2}{c}{\textbf{B.} $\tau$ quadratic $\perp Z^j$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"CATE-equivalence test & $n^\star$ & IoU & $n^\star$ & IoU \\",
        r"\midrule",
    ]
    for key, label in TESTS.items():
        row = [label.replace("linear (default)", "Linear (default)"),
               _fmt_thr(_threshold(a_n, "n", "fixed_effect", 2.0, key)),
               cell(a_n, "iou", "fixed_effect", 5.0, "n", 2000, key),
               _fmt_thr(_threshold(b_n, "n", "fixed_effect", 5.0, key)),
               cell(b_n, "iou", "fixed_effect", 5.0, "n", 2000, key)]
        row = [c.replace("—", "---") for c in row]
        if key == "NEXIS":
            row[0] = r"\textbf{" + row[0] + "}"
        lines.append(" & ".join(row) + r" \\")
        if key == "NEXIS (test=GCM: lgbm)":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def build_type1(k: int = 20) -> tuple[pd.DataFrame, str]:
    """Empirical FWER at effect_scale = 0: a run is an error iff it selects anything."""
    rows = []
    seen_feats = []
    for feat, flabel in [("sae", "z (codes)"), ("sae_precode", "z_pre (pre-act.)")]:
        path = (ROOT / "results/celeba/experiment_type1" / f"k{k}" / feat /
                "effect_sweep.parquet")
        if not path.exists():
            continue
        seen_feats.append(flabel)
        df = pd.read_parquet(path)
        df = df[df["effect_scale"] == 0]
        for key, label in {BASELINE: "marginal (FWER)", **TESTS}.items():
            sub = df[df["method"] == key]
            if sub.empty:
                continue
            r = {"features": flabel, "test": label}
            for nn in sorted(sub["fixed_n"].unique()):
                s = sub[sub["fixed_n"] == nn]
                err = (s["n_selected"].to_numpy() > 0).astype(float)
                se = err.std(ddof=1) / np.sqrt(err.size) if err.size > 1 else 0.0
                r[f"FWER (n={int(nn)})"] = f"{err.mean():.2f}±{se:.2f}"
            rows.append(r)
    caption = (f"Type-I error control of the CATE-equivalence tests on CelebA "
               f"($k={k}$, $\\eta=0$, so every selection is a false discovery). "
               f"Entries are the empirical family-wise error rate "
               f"$\\widehat{{\\mathbb{{P}}}}(\\widehat{{\\mathcal{{S}}}}_n \\neq \\emptyset)$ "
               f"over 50 seeds $\\pm$ Monte-Carlo standard error; the nominal level is "
               f"$\\alpha = 0.05$.")
    return pd.DataFrame(rows), caption


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dgp", choices=["attr", "ortho_quadratic", "type1", "paper"],
                   default="attr")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--feat", default=None,
                   help="sae | sae_precode (default: sae for attr, sae_precode otherwise)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if args.dgp == "paper":
        tex = build_paper_table(args.k)
        out = args.out or (ROOT / "results/celeba/appendix" / "table_test_ablation.tex")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(tex + "\n")
        print(tex)
        print(f"\nSaved → {out}")
        return

    if args.dgp == "type1":
        tab, caption = build_type1(args.k)
        md = to_markdown(tab)
        print("\n### Type-I error control (η = 0)\n")
        print(md)
        out = args.out or (ROOT / "results/celeba/appendix" / "table_test_type1.tex")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(to_latex(tab, caption, "tab:celeba:test_type1") + "\n")
        out.with_suffix(".md").write_text(md + "\n")
        print(f"\nSaved → {out}\n     → {out.with_suffix('.md')}")
        return

    feat = args.feat or ("sae" if args.dgp == "attr" else "sae_precode")
    root = ROOT / ("results/celeba/experiment" if args.dgp == "attr"
                   else "results/celeba/experiment_ushape") / f"k{args.k}" / feat
    df_n = pd.read_parquet(root / "n_sweep.parquet")
    df_e = pd.read_parquet(root / "effect_sweep.parquet")

    n_points = sorted(df_n["fixed_effect"].unique())
    e_points = sorted(df_e["fixed_n"].unique())
    ref_e, ref_n = max(n_points), max(e_points)

    methods = dict(TESTS)
    if BASELINE in set(df_n["method"]) | set(df_e["method"]):
        methods = {BASELINE: "marginal (FWER)", **methods}

    tab = build(df_n, df_e, methods, n_points, e_points, ref_n, ref_e)

    seeds = df_n.groupby("method")["seed"].nunique().max()
    m_dim = "9{,}216" if args.k == 20 else "13{,}824"
    dgp_txt = ("$\\tau$ linear in the binary attributes"
               if args.dgp == "attr"
               else "$\\tau$ an orthogonalised quadratic of the two ground-truth "
                    "coordinates ($\\mathrm{Cov}(\\tau, Z^j) = 0$ in the population)")
    caption = (f"CATE-equivalence test ablation on CelebA "
               f"($k={args.k}$ SAE {'codes' if feat == 'sae' else 'pre-activations'}, "
               f"$m={m_dim}$, $|\\mathcal{{S}}^\\star|=2$, {seeds} seeds). DGP: {dgp_txt}. "
               f"$n^\\star$ / $\\eta^\\star$ are the smallest grid values reaching mean "
               f"recall $\\geq {THR}$; --- means never on the grid. "
               f"Precision / Recall / IoU are reported at $n={ref_n:g}$, $\\eta={ref_e:g}$ "
               f"with Monte-Carlo standard errors.")
    label = f"tab:celeba:test_ablation_{args.dgp}"

    md = to_markdown(tab)
    tex = to_latex(tab, caption, label)
    print(f"\n### DGP = {args.dgp}   ({root.relative_to(ROOT)})   seeds={seeds}\n")
    print(md)

    out = args.out or (ROOT / "results/celeba/appendix" /
                       f"table_test_ablation_{args.dgp}.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex + "\n")
    out.with_suffix(".md").write_text(md + "\n")
    print(f"\nSaved → {out}\n     → {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
