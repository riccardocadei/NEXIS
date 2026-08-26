#!/usr/bin/env python3
"""
Agreement between two CelebA experiment runs that differ *only* in the SAE training corpus.

Setting
-------
Arm A ("paper")    : SAEs trained on the 19,867 valid-split images.
Arm B ("resample") : SAEs trained on 19,867 *different* images (CelebA train split,
                     identity-disjoint), same architecture, sparsity and schedule.
Both arms encode the same valid-split embeddings and are evaluated on the same design
grid with the same Monte Carlo seeds.  Because ``generate_celeba_rct`` seeds the image
draw, the treatment, and the outcome noise from ``seed`` alone — never from the features —
arm A and arm B see the *identical* dataset at every (grid point, seed) cell.  Every
difference reported here is therefore attributable to the dictionary alone, and can be
measured with paired statistics.

Metrics
-------
For a per-run metric m (precision, recall, IoU, exact recovery) at grid point g and
seed s, with paired difference d(g,s) = m_B(g,s) − m_A(g,s):

1. Paired mean difference  Δ = mean_{g,s} d      (pp), with a seed-clustered bootstrap CI.
2. Equivalence (TOST)      declared when the 90 % CI of Δ lies inside ±δ (default δ = 5 pp).
3. Curve discrepancy       MAD  = mean_g |D(g)|,  MaxAD = max_g |D(g)|, D(g) = mean_s d(g,s).
4. MC-normalised discrepancy  |t| = mean_g |D(g)| / SE_g, SE_g = sd_s d(g,s)/√S.
   This is the headline number: it expresses the disagreement in units of the Monte Carlo
   noise of the paper's own curves.  |t| ≲ 2 means the two dictionaries are statistically
   indistinguishable at that resolution; large |t| means a real dictionary effect (which
   may still be practically negligible — read it together with MAD).
5. Conclusion-level agreement
   - PAS (Performance Agreement Score): share of (method × grid point) cells with
     |D(g)| ≤ δ.
   - Sign-flip rate of the paper's key claim, precision(NEXIS) − precision(Marginal FWER):
     the share of grid points where the sign of that gap differs between arms.
   - Kendall τ between the method rankings (by precision, by recall) at each grid point.
   - Detection-threshold ratio: the smallest n (resp. η) at which mean recall reaches 0.9,
     by linear interpolation, as a ratio B / A.

Also reported per config: the ground-truth (principal) neuron of each attribute in each
arm and its best-threshold F1, plus the margin to the runner-up neuron — the dictionary-side
explanation for any performance gap.

Usage
-----
    python src/apps/celeba/sae_agreement.py --tag b1
    python src/apps/celeba/sae_agreement.py --tag b1 --delta 0.05 --no-figure
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

CONFIGS  = [("k20", "sae"), ("k20", "sae_precode"), ("k5", "sae"), ("k5", "sae_precode")]
METRICS  = ["recall", "precision", "iou", "exact"]
MAIN_METHOD, BASE_METHOD = "NEXIS", "Marginal Testing (FWER)"
N_BOOT   = 2000


# ── loading ───────────────────────────────────────────────────────────────────

def load_arm(base: Path, kdir: str, feature: str) -> dict:
    """Load one arm's sweeps + ground truth for one config."""
    d = base / kdir / feature
    out = {}
    for fname, sweep_col, fixed_col in [("effect_sweep.parquet", "effect_scale", "fixed_n"),
                                        ("n_sweep.parquet",      "n",            "fixed_effect")]:
        p = d / fname
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["exact"] = (df["iou"] >= 1.0).astype(float)
        out[sweep_col] = (df, fixed_col)
    gt_path = d / "ground_truth.json"
    out["gt"] = json.loads(gt_path.read_text()) if gt_path.exists() else None
    return out


# ── paired statistics ─────────────────────────────────────────────────────────

def _paired_matrix(a: pd.DataFrame, b: pd.DataFrame, sweep_col: str,
                   metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (D, seeds, grid): D[g, s] = m_B − m_A on the common (grid × seed) cells."""
    grid  = sorted(set(a[sweep_col]) & set(b[sweep_col]))
    seeds = sorted(set(a["seed"])    & set(b["seed"]))
    pa = a.pivot_table(index=sweep_col, columns="seed", values=metric, aggfunc="mean")
    pb = b.pivot_table(index=sweep_col, columns="seed", values=metric, aggfunc="mean")
    pa, pb = pa.loc[grid, seeds], pb.loc[grid, seeds]
    return pb.to_numpy() - pa.to_numpy(), np.array(seeds), np.array(grid)


def paired_stats(a: pd.DataFrame, b: pd.DataFrame, sweep_col: str, metric: str,
                 delta: float, rng: np.random.Generator) -> dict:
    """Paired agreement statistics for one (config, sweep, fixed value, method, metric)."""
    D, seeds, grid = _paired_matrix(a, b, sweep_col, metric)
    if D.size == 0:
        return {}
    S = D.shape[1]
    curve = D.mean(axis=1)                                  # D(g)
    se    = D.std(axis=1, ddof=1) / np.sqrt(S)              # SE of D(g)

    # seed-clustered bootstrap of the overall paired mean (keeps the g-structure intact)
    idx   = rng.integers(0, S, size=(N_BOOT, S))
    boots = D[:, idx].mean(axis=(0, 2))                     # (N_BOOT,)
    lo90, hi90 = np.percentile(boots, [5, 95])
    lo95, hi95 = np.percentile(boots, [2.5, 97.5])

    # SE floor: when both arms are near-deterministic at a grid point (e.g. precision 1.0
    # for every seed), the paired SE collapses and a raw t would explode on a difference of
    # a fraction of a point.  Floor it at 0.2 pp — below that the gap is not practically
    # meaningful anyway — so |t| stays interpretable as "MC standard errors of disagreement".
    se_floor = np.maximum(se, 0.002)
    t_abs = float(np.mean(np.abs(curve) / se_floor))

    return {
        "mean_diff":   float(curve.mean()),
        "ci95_lo":     float(lo95), "ci95_hi": float(hi95),
        "tost_equiv":  bool(lo90 > -delta and hi90 < delta),
        "mad":         float(np.abs(curve).mean()),
        "max_ad":      float(np.abs(curve).max()),
        "t_abs":       t_abs,
        "pas":         float(np.mean(np.abs(curve) <= delta)),
        "n_grid":      int(len(grid)),
        "n_seeds":     int(S),
    }


def threshold(df: pd.DataFrame, sweep_col: str, metric: str, target: float) -> float:
    """Smallest grid value where the seed-mean metric reaches `target` (linear interp)."""
    s = df.groupby(sweep_col)[metric].mean().sort_index()
    x, y = s.index.to_numpy(dtype=float), s.to_numpy()
    hit = np.where(y >= target)[0]
    if len(hit) == 0:
        return float("nan")
    i = hit[0]
    if i == 0:
        return float(x[0])
    x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x1)


def kendall_tau(ra: np.ndarray, rb: np.ndarray) -> float:
    """Kendall τ-b between two score vectors (small n → direct O(n²) count)."""
    n = len(ra)
    if n < 2:
        return float("nan")
    conc = disc = tie_a = tie_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = ra[i] - ra[j], rb[i] - rb[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                tie_a += 1
            elif db == 0:
                tie_b += 1
            elif np.sign(da) == np.sign(db):
                conc += 1
            else:
                disc += 1
    n0 = conc + disc + tie_a + tie_b
    den = np.sqrt((n0 - tie_a) * (n0 - tie_b))
    return float((conc - disc) / den) if den > 0 else float("nan")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag",          default="b1")
    p.add_argument("--results-root", type=Path, default="results/celeba")
    p.add_argument("--arm-a",        type=Path, default=None,
                   help="Arm A results dir (default: <results-root>/experiment)")
    p.add_argument("--delta",        type=float, default=0.05,
                   help="Equivalence margin in absolute metric units (default: 0.05 = 5 pp)")
    p.add_argument("--out-dir",      type=Path, default=None,
                   help="Where to write the report (default: <results-root>/agreement_<tag>)")
    p.add_argument("--no-figure",    action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = (ROOT / args.results_root if not args.results_root.is_absolute()
            else args.results_root)
    base_a = args.arm_a or (root / "experiment")
    base_b = root / f"experiment_resample_{args.tag}"
    out_dir = args.out_dir or (root / f"agreement_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    rows, dict_rows, concl_rows = [], [], []

    for kdir, feature in CONFIGS:
        A, B = load_arm(base_a, kdir, feature), load_arm(base_b, kdir, feature)
        if not A or not B or "effect_scale" not in A or "effect_scale" not in B:
            print(f"[skip] {kdir}/{feature}: missing results")
            continue

        # ── dictionary-side diagnostics ──────────────────────────────────────
        for arm, G in (("A", A["gt"]), ("B", B["gt"])):
            if G is None:
                continue
            for w in ("w1", "w2"):
                f1 = np.asarray(G.get(f"{w}_f1_scores_pre") or G[f"{w}_f1_scores"], dtype=float)
                order = np.argsort(-f1)
                dict_rows.append({
                    "config": f"{kdir}/{feature}", "arm": arm, "attr": G[f"{w}_attr"],
                    "principal_neuron": int(order[0]),
                    "f1_principal": float(f1[order[0]]),
                    "f1_runner_up":  float(f1[order[1]]),
                    "margin":        float(f1[order[0]] - f1[order[1]]),
                    "truth": G["truth"],
                })

        # ── sweep-level paired agreement ─────────────────────────────────────
        for sweep_col in ("effect_scale", "n"):
            if sweep_col not in A or sweep_col not in B:
                continue
            (dfa, fixed_col), (dfb, _) = A[sweep_col], B[sweep_col]
            fixed_vals = sorted(set(dfa[fixed_col]) & set(dfb[fixed_col]))
            methods    = sorted(set(dfa["method"])  & set(dfb["method"]))

            for fv, method in product(fixed_vals, methods):
                sa = dfa[(dfa[fixed_col] == fv) & (dfa["method"] == method)]
                sb = dfb[(dfb[fixed_col] == fv) & (dfb["method"] == method)]
                if sa.empty or sb.empty:
                    continue
                for metric in METRICS:
                    st = paired_stats(sa, sb, sweep_col, metric, args.delta, rng)
                    if st:
                        rows.append({"config": f"{kdir}/{feature}", "sweep": sweep_col,
                                     fixed_col: fv, "method": method, "metric": metric, **st})

            # ── conclusion-level agreement, per fixed value ──────────────────
            for fv in fixed_vals:
                fa = dfa[dfa[fixed_col] == fv]
                fb = dfb[dfb[fixed_col] == fv]
                grid = sorted(set(fa[sweep_col]) & set(fb[sweep_col]))

                # sign of the paper's headline precision gap, per grid point
                flips = 0
                for g in grid:
                    def gap(df):
                        sl = df[df[sweep_col] == g]
                        m = sl.groupby("method")["precision"].mean()
                        if MAIN_METHOD not in m or BASE_METHOD not in m:
                            return np.nan
                        return m[MAIN_METHOD] - m[BASE_METHOD]
                    ga, gb = gap(fa), gap(fb)
                    if np.isfinite(ga) and np.isfinite(gb) and np.sign(ga) != np.sign(gb):
                        flips += 1

                taus = {}
                for metric in ("precision", "recall"):
                    tt = []
                    for g in grid:
                        ma = fa[fa[sweep_col] == g].groupby("method")[metric].mean()
                        mb = fb[fb[sweep_col] == g].groupby("method")[metric].mean()
                        common = sorted(set(ma.index) & set(mb.index))
                        if len(common) >= 3:
                            tt.append(kendall_tau(ma[common].to_numpy(), mb[common].to_numpy()))
                    taus[metric] = float(np.nanmean(tt)) if tt else float("nan")

                nex_a = fa[fa["method"] == MAIN_METHOD]
                nex_b = fb[fb["method"] == MAIN_METHOD]
                thr_a = threshold(nex_a, sweep_col, "recall", 0.9)
                thr_b = threshold(nex_b, sweep_col, "recall", 0.9)
                concl_rows.append({
                    "config": f"{kdir}/{feature}", "sweep": sweep_col, fixed_col: fv,
                    "sign_flip_rate":    flips / len(grid) if grid else float("nan"),
                    "kendall_tau_prec":  taus["precision"],
                    "kendall_tau_recall": taus["recall"],
                    "recall90_A": thr_a, "recall90_B": thr_b,
                    "recall90_ratio_B_over_A": (thr_b / thr_a
                                                if np.isfinite(thr_a) and np.isfinite(thr_b)
                                                and thr_a else float("nan")),
                })
        print(f"[done] {kdir}/{feature}")

    if not rows:
        print("Nothing to compare — did the arm-B rerun finish and merge?")
        return 1

    df   = pd.DataFrame(rows)
    dd   = pd.DataFrame(dict_rows)
    dc   = pd.DataFrame(concl_rows)
    df.to_csv(out_dir / "agreement_by_cell.csv", index=False)
    dd.to_csv(out_dir / "dictionary_diagnostics.csv", index=False)
    dc.to_csv(out_dir / "conclusion_agreement.csv", index=False)

    # ── summary ──────────────────────────────────────────────────────────────
    lines: list[str] = []
    w = lines.append
    w(f"# SAE training-corpus resampling: performance agreement (tag = {args.tag})\n")
    w(f"Arm A = {base_a.relative_to(root.parent) if base_a.is_relative_to(root.parent) else base_a}, "
      f"arm B = {base_b.name}.  Equivalence margin δ = {args.delta:.3f}.")
    w(f"Paired over identical (grid point, seed) datasets; "
      f"{int(df.n_seeds.max())} seeds per cell.\n")

    w("## Headline, main setting (k=20, sparse codes z, NEXIS)\n")
    main = df[(df.config == "k20/sae") & (df.method == MAIN_METHOD)]
    w("| sweep | fixed | metric | Δ (B−A) | 95% CI | MAD | max | mean \\|t\\| | equiv |")
    w("|---|---|---|---|---|---|---|---|---|")
    for _, r in main.iterrows():
        fv = r.get("fixed_n") if np.isfinite(r.get("fixed_n", np.nan)) else r.get("fixed_effect")
        w(f"| {r['sweep']} | {fv:g} | {r['metric']} | {r['mean_diff']:+.3f} | "
          f"[{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}] | {r['mad']:.3f} | {r['max_ad']:.3f} | "
          f"{r['t_abs']:.1f} | {'yes' if r['tost_equiv'] else 'NO'} |")

    w("\n## All methods and configs, aggregated\n")
    agg = (df.groupby(["config", "metric"])
             .agg(mean_diff=("mean_diff", "mean"), mad=("mad", "mean"),
                  max_ad=("max_ad", "max"), t_abs=("t_abs", "mean"),
                  pas=("pas", "mean"), equiv_share=("tost_equiv", "mean"))
             .reset_index())
    w("| config | metric | mean Δ | mean MAD | worst \\|D\\| | mean \\|t\\| | PAS | TOST-equivalent cells |")
    w("|---|---|---|---|---|---|---|---|")
    for _, r in agg.iterrows():
        w(f"| {r['config']} | {r['metric']} | {r['mean_diff']:+.3f} | {r['mad']:.3f} | "
          f"{r['max_ad']:.3f} | {r['t_abs']:.1f} | {r['pas']:.2f} | {r['equiv_share']:.2f} |")

    w("\n## Conclusion-level agreement\n")
    w("| config | sweep | fixed | sign-flip rate | τ (precision) | τ (recall) | n*/η* A | B | ratio |")
    w("|---|---|---|---|---|---|---|---|---|")
    for _, r in dc.iterrows():
        fv = r.get("fixed_n") if np.isfinite(r.get("fixed_n", np.nan)) else r.get("fixed_effect")
        w(f"| {r['config']} | {r['sweep']} | {fv:g} | {r['sign_flip_rate']:.2f} | "
          f"{r['kendall_tau_prec']:.2f} | {r['kendall_tau_recall']:.2f} | "
          f"{r['recall90_A']:.0f} | {r['recall90_B']:.0f} | "
          f"{r['recall90_ratio_B_over_A']:.2f} |")

    w("\n## Dictionary diagnostics (why any gap exists)\n")
    w("| config | arm | attribute | principal neuron | F1 | runner-up F1 | margin |")
    w("|---|---|---|---|---|---|---|")
    for _, r in dd.iterrows():
        w(f"| {r['config']} | {r['arm']} | {r['attr']} | {r['principal_neuron']} | "
          f"{r['f1_principal']:.3f} | {r['f1_runner_up']:.3f} | {r['margin']:.3f} |")

    for label, col in (("largest practical gap (MAD)", "mad"),
                       ("largest statistical gap (|t|)", "t_abs")):
        worst = df.loc[df[col].idxmax()]
        w(f"\nWorst cell by {label}: {worst['config']} / {worst['method']} / "
          f"{worst['metric']} on the {worst['sweep']} sweep — "
          f"Δ = {worst['mean_diff']:+.3f}, MAD = {worst['mad']:.3f}, "
          f"max |D| = {worst['max_ad']:.3f}, mean |t| = {worst['t_abs']:.1f}.")
    w(f"\nTOST-equivalent cells overall: {df.tost_equiv.mean():.1%} of {len(df)} "
      f"(cell = config × sweep × fixed value × method × metric).")
    w(f"Cells with MAD ≤ δ: {(df['mad'] <= args.delta).mean():.1%}; "
      f"with max |D| ≤ δ: {(df['max_ad'] <= args.delta).mean():.1%}.")

    report = "\n".join(lines) + "\n"
    (out_dir / "REPORT.md").write_text(report)
    print(report)
    print(f"Wrote {out_dir}/REPORT.md, agreement_by_cell.csv, "
          f"conclusion_agreement.csv, dictionary_diagnostics.csv")

    if not args.no_figure:
        make_figure(base_a, base_b, out_dir, args.tag)
    return 0


def make_figure(base_a: Path, base_b: Path, out_dir: Path, tag: str) -> None:
    """Overlay arm A and arm B curves for NEXIS and the FWER baseline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("effect_scale", "fixed_n", 2000, r"Effect size $\eta$"),
              ("n",            "fixed_effect", 5.0, r"Sample size $n$")]
    fig, axes = plt.subplots(len(CONFIGS), 4, figsize=(17, 3.1 * len(CONFIGS)),
                             squeeze=False)
    for i, (kdir, feature) in enumerate(CONFIGS):
        A, B = load_arm(base_a, kdir, feature), load_arm(base_b, kdir, feature)
        col = 0
        for sweep_col, fixed_col, fv, xlabel in panels:
            for metric in ("recall", "precision"):
                ax = axes[i][col]; col += 1
                for arm, pack, style in (("A (paper)", A, "-o"), ("B (resample)", B, "--s")):
                    if sweep_col not in pack:
                        continue
                    dfx, fcol = pack[sweep_col]
                    for method, colour in ((MAIN_METHOD, "tab:blue"),
                                           (BASE_METHOD, "tab:orange")):
                        sl = dfx[(dfx[fcol] == fv) & (dfx["method"] == method)]
                        if sl.empty:
                            continue
                        s = sl.groupby(sweep_col)[metric].mean().sort_index()
                        ax.plot(s.index, s.values, style, color=colour, ms=3.5, lw=1.4,
                                alpha=0.95 if arm.startswith("A") else 0.75,
                                label=f"{method} — {arm}")
                if sweep_col == "n":
                    ax.set_xscale("log")
                ax.set_ylim(-0.03, 1.03)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(metric)
                ax.set_title(f"{kdir}/{feature} — {metric} "
                             f"({fixed_col.replace('fixed_', '')}={fv:g})", fontsize=9)
                ax.grid(alpha=0.25)
        axes[i][0].legend(fontsize=6, loc="lower right")
    fig.suptitle(f"CelebA experiments under SAE training-corpus resampling "
                 f"(solid = paper SAEs, dashed = resampled SAEs, tag {tag})", y=1.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"agreement_curves.{ext}", dpi=160, bbox_inches="tight")
    print(f"Wrote {out_dir}/agreement_curves.png")


if __name__ == "__main__":
    raise SystemExit(main())
