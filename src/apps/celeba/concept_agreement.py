#!/usr/bin/env python3
"""
Concept-level agreement between two SAE dictionaries that differ only in training corpus.

The sweep parquets record *how many* features NEXIS recovered and how many were correct,
but not *which* ones — and the neuron indices of two independently trained SAEs are not
comparable anyway.  This script re-runs the main design point while recording the selected
sets, then compares the two arms at the level of what the recovered features *mean*:

  Concept agreement
      Every recovered feature is described by its top-M activating images (over the same
      19,867 evaluation images), and labelled with the CelebA attribute those images share
      (the attribute maximising top-M purity, subject to a purity floor; CelebA's 40
      attributes act as the semantic vocabulary in place of a VLM caption).  Features are
      then matched across the two arms by top-M image-set Jaccard overlap (greedy, highest
      overlap first).  Concept agreement is the share of arm-B recoveries whose matched
      arm-A partner carries the same concept label — i.e. the proportion of recovered
      features that mean the same thing in both dictionaries.

  Matched-CATE correlation
      For each matched pair (j_A, j_B) and each seed, the per-unit CATE profile implied by
      that feature is fitted in each arm, tau_i = tau_T + beta_TZ * Z_ij from
      Y ~ 1 + T + Z_j + T*Z_j, and the two profiles are correlated across units.  The two
      arms see the *same* units (the RCT draw is seeded independently of the features), so
      this is a paired, unit-level comparison of the recovered heterogeneity map.
      Reported as the Fisher-z mean over pairs and seeds.

Outputs (in --out-dir):
    headline_table.md / .csv     the reporting table (precision, recall, IoU, both agreements)
    selections.csv               every selected feature, per arm / seed, with its concept label
    matches.csv                  matched feature pairs with Jaccard, labels and CATE correlation
    topk_<arm>_<neuron>.png      contact sheets of top-activating images (with --contact-sheets)

Usage
-----
    python src/apps/celeba/concept_agreement.py --tag b1
    python src/apps/celeba/concept_agreement.py --tag b1 --sae-top-k 5 --n 2000 --effect 5.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[3]

from apps.celeba.scm import build_buckets, generate_celeba_rct
from apps.celeba.experiment import compute_f1_scores
from method.nexis import nexis

# Attributes that define the DGP; excluded from nothing, but reported separately.
W1_ATTR, W2_ATTR = "Wearing_Hat", "Eyeglasses"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag",           default="b1", help="Resample tag of arm B")
    p.add_argument("--sae-top-k",     type=int, default=20)
    p.add_argument("--n",             type=int, default=2000, help="Sample size (main setting)")
    p.add_argument("--effect",        type=float, default=5.0, help="Effect scale η (main setting)")
    p.add_argument("--grid",          action="store_true",
                   help="Evaluate over the *whole* paper design grid (10 η values × "
                        "{n=500,2000} plus 11 n values × {η=2,5} = 42 cells) instead of the "
                        "single main design point, and pool every reported quantity over it.")
    p.add_argument("--n-seeds",       type=int, default=50)
    p.add_argument("--alpha",         type=float, default=0.05)
    p.add_argument("--max-steps",     type=int, default=10)
    p.add_argument("--top-m",         type=int, default=100,
                   help="Top-activating images per feature used for description/matching")
    p.add_argument("--purity-floor",  type=float, default=0.30,
                   help="Minimum share of top-M images carrying an attribute for it to be "
                        "accepted as the concept label (else 'unlabelled')")
    p.add_argument("--jaccard-floor", type=float, default=0.05,
                   help="Minimum top-M overlap for two features to be called a match")
    p.add_argument("--n-jobs",        type=int, default=-1)
    p.add_argument("--data-dir",      type=Path, default="data/celeba")
    p.add_argument("--results-root",  type=Path, default="results/celeba")
    p.add_argument("--out-dir",       type=Path, default=None)
    p.add_argument("--contact-sheets", action="store_true",
                   help="Write PNG contact sheets of top-activating images per recovered "
                        "feature (needs data/celeba/images.npy)")
    return p.parse_args()


# ── description / labelling ───────────────────────────────────────────────────

def top_indices(zpre: np.ndarray, j: int, m: int) -> np.ndarray:
    """Indices of the m images with the highest pre-activation on feature j."""
    col = zpre[:, j]
    idx = np.argpartition(-col, m)[:m]
    return idx[np.argsort(-col[idx])]


def concept_label(top_idx: np.ndarray, labels: pd.DataFrame, attrs: list[str],
                  prevalence: np.ndarray, purity_floor: float) -> tuple[str, float, float]:
    """Label a feature by the attribute its top-activating images share.

    Returns (attribute, purity, lift).  Purity is the share of the top-M images carrying
    the attribute; lift is purity / dataset prevalence.  The attribute maximising lift is
    chosen among those clearing the purity floor, so a very rare-but-pure concept beats a
    common one; falls back to 'unlabelled' when nothing clears the floor.
    """
    sub = labels.iloc[top_idx][attrs].to_numpy(dtype=float)   # (M, n_attr)
    purity = sub.mean(axis=0)
    lift = purity / np.maximum(prevalence, 1e-9)
    ok = purity >= purity_floor
    if not ok.any():
        return "unlabelled", float(purity.max()), 0.0
    cand = np.where(ok)[0]
    best = cand[np.argmax(lift[cand])]
    return attrs[best], float(purity[best]), float(lift[best])


def cate_profile(y: np.ndarray, t: np.ndarray, zj: np.ndarray) -> np.ndarray:
    """Per-unit CATE from the saturated single-feature interaction fit."""
    X = np.column_stack([np.ones_like(t), t, zj, t * zj])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta[1] + beta[3] * zj          # tau_i = tau_T + beta_TZ * Z_ij


def cate_slope(y: np.ndarray, t: np.ndarray, zj: np.ndarray) -> float:
    """The interaction slope beta_TZ of that fit.

    The CATE profile is affine in Z_j, so the correlation between two matched features'
    profiles is sign(beta_A * beta_B) * corr(Z_A, Z_B) over the sampled units — which is
    how the grid-wide version computes it without storing 4,200 profiles.
    """
    X = np.column_stack([np.ones_like(t), t, zj, t * zj])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[3])


def design_cells() -> list[tuple[int, float, str]]:
    """The paper's full CelebA design grid as (n, effect_scale, sweep) cells."""
    effect_grid = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    n_grid      = [50, 100, 200, 350, 500, 750, 1000, 2000, 3500, 5000, 10000]
    cells  = [(n, e, "effect") for n in (500, 2000) for e in effect_grid]
    cells += [(n, e, "n")      for e in (2.0, 5.0)  for n in n_grid]
    return cells


# ── one Monte Carlo replication ───────────────────────────────────────────────

def run_seed(seed: int, feats: dict[str, np.ndarray], labels_df: pd.DataFrame,
             buckets, truth: dict[str, list[int]], n: int, effect: float,
             alpha: float, max_steps: int, cell: str = "main") -> dict:
    """Run NEXIS on both arms for one seed; the RCT draw is identical across arms."""
    out: dict = {"seed": seed, "cell": cell, "n": n, "effect": effect, "arms": {}}
    ref = None
    for arm, Z in feats.items():
        try:
            data = generate_celeba_rct(
                n=n, features=Z, labels_df=labels_df, buckets=buckets,
                effect_scale=effect, seed=seed, w1_attr=W1_ATTR, w2_attr=W2_ATTR,
            )
        except ValueError:
            return None            # (W1, W2) bucket exhausted at this n — as in run_sweep
        # Identical draw across arms (seeded from labels + seed only) — verify once.
        key = (data.T.sum(), float(data.Y.sum()), int(data.image_indices.sum()))
        if ref is None:
            ref = key
        elif key != ref:
            raise RuntimeError(f"arms disagree on the sampled dataset at seed {seed}")

        out["image_indices"] = data.image_indices
        res = nexis(y=data.Y, t=data.T, z=data.Z, alpha=alpha, max_rounds=max_steps)
        sel = [int(j) for j in res.selected]
        tset = set(truth[arm])
        tp = len(set(sel) & tset)
        out["arms"][arm] = {
            "selected":  sel,
            "tp": tp, "fp": len(sel) - tp, "n_selected": len(sel),
            "recall":    tp / len(tset) if tset else 1.0,
            "precision": tp / len(sel) if sel else (1.0 if not tset else 0.0),
            "iou":       tp / len(set(sel) | tset) if (sel or tset) else 1.0,
            # CATE profiles are affine in Z_j, so the slope + the sampled rows are all that
            # the matched-profile correlation needs (see cate_slope).
            "slope": {j: cate_slope(data.Y, data.T, data.Z[:, j]) for j in sel},
        }
    return out


def main() -> int:
    args = parse_args()
    data_dir = ROOT / args.data_dir
    root     = ROOT / args.results_root
    out_dir  = args.out_dir or (root / f"agreement_{args.tag}" / f"concept_k{args.sae_top_k}")
    out_dir  = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    K = args.sae_top_k

    labels_df = pd.read_parquet(data_dir / "labels.parquet")
    attrs = [c for c in labels_df.columns
             if labels_df[c].dropna().isin([0, 1]).all() and labels_df[c].nunique() == 2]
    prevalence = labels_df[attrs].mean().to_numpy(dtype=float)

    paths = {
        "A": (data_dir / "embeddings" / f"sae_k{K}.npy",
              data_dir / "embeddings" / f"sae_precode_k{K}.npy",
              root / "experiment" / f"k{K}" / "sae" / "ground_truth.json"),
        "B": (ROOT / f"data/celeba_resample_{args.tag}/eval/embeddings/sae_k{K}.npy",
              ROOT / f"data/celeba_resample_{args.tag}/eval/embeddings/sae_precode_k{K}.npy",
              root / f"experiment_resample_{args.tag}" / f"k{K}" / "sae" / "ground_truth.json"),
    }
    feats, zpre, truth = {}, {}, {}
    for arm, (zp, pp, gtp) in paths.items():
        feats[arm] = np.load(zp)
        zpre[arm]  = np.load(pp)
        if gtp.exists():
            truth[arm] = json.loads(gtp.read_text())["truth"]
        else:   # fall back to recomputing S* from the F1 spectra
            f1a = compute_f1_scores(zpre[arm], labels_df[W1_ATTR].values.astype(float))
            f1b = compute_f1_scores(zpre[arm], labels_df[W2_ATTR].values.astype(float))
            truth[arm] = sorted({int(f1a.argmax()), int(f1b.argmax())})
        print(f"arm {arm}: z={feats[arm].shape}  S*={truth[arm]}")

    buckets = build_buckets(labels_df, W1_ATTR, W2_ATTR)

    if args.grid:
        cells = design_cells()
        scope = (f"the whole design grid ({len(cells)} cells × {args.n_seeds} seeds "
                 f"= {len(cells) * args.n_seeds} paired runs per arm)")
        tasks = [(s, n, e, f"{sw}_n{n}_e{e:g}") for (n, e, sw) in cells
                 for s in range(args.n_seeds)]
    else:
        scope = f"the main design point (n={args.n}, η={args.effect}, {args.n_seeds} seeds)"
        tasks = [(s, args.n, args.effect, "main") for s in range(args.n_seeds)]

    print(f"\nRunning NEXIS on both arms over {scope} …", flush=True)
    runs = Parallel(n_jobs=args.n_jobs, prefer="threads")(
        delayed(run_seed)(s, feats, labels_df, buckets, truth,
                          n, e, args.alpha, args.max_steps, cell)
        for s, n, e, cell in tasks
    )
    runs = [r for r in runs if r is not None]

    # ── per-arm headline metrics ─────────────────────────────────────────────
    perf_rows = []
    for r in runs:
        for arm, d in r["arms"].items():
            perf_rows.append({"arm": arm, "seed": r["seed"],
                              **{k: d[k] for k in ("precision", "recall", "iou",
                                                   "tp", "fp", "n_selected")}})
    perf = pd.DataFrame(perf_rows)
    summary = perf.groupby("arm")[["precision", "recall", "iou", "n_selected"]].mean()
    # Standard errors cluster on seed: the design cells are fixed, the 50 seeds are the
    # independent replications.  In single-point mode this is the plain across-seed SEM.
    seed_means = perf.groupby(["arm", "seed"])[["precision", "recall", "iou"]].mean()
    sem = seed_means.groupby("arm").sem()

    # ── describe every selected feature ──────────────────────────────────────
    sel_counts = {arm: {} for arm in feats}
    for r in runs:
        for arm, d in r["arms"].items():
            for j in d["selected"]:
                sel_counts[arm][j] = sel_counts[arm].get(j, 0) + 1

    desc: dict[tuple[str, int], dict] = {}
    for arm in feats:
        for j in sel_counts[arm]:
            ti = top_indices(zpre[arm], j, args.top_m)
            lab, purity, lift = concept_label(ti, labels_df, attrs, prevalence,
                                              args.purity_floor)
            desc[(arm, j)] = {"arm": arm, "neuron": j, "top_idx": ti,
                              "concept": lab, "purity": purity, "lift": lift,
                              "n_runs_selected": sel_counts[arm][j],
                              "is_truth": j in truth[arm]}
    sel_df = pd.DataFrame([{k: v for k, v in d.items() if k != "top_idx"}
                           for d in desc.values()]).sort_values(
        ["arm", "n_runs_selected"], ascending=[True, False])
    sel_df.to_csv(out_dir / "selections.csv", index=False)

    # ── concept-level scoring ────────────────────────────────────────────────
    # Index-level precision punishes a dictionary that splits an injected concept over two
    # coordinates: both are genuinely about the concept, but S* names only the single
    # F1-argmax coordinate, so the second counts as a false positive.  Scoring the same
    # selections against the concept *set* {Wearing_Hat, Eyeglasses} removes that
    # index-convention artefact and is comparable across dictionaries.
    C_TRUE = {W1_ATTR, W2_ATTR}
    concept_rows = []
    for r in runs:
        for arm, d in r["arms"].items():
            labs = [desc[(arm, j)]["concept"] for j in d["selected"]]
            hits = [l for l in labs if l in C_TRUE]
            cset = set(labs)
            concept_rows.append({
                "arm": arm, "seed": r["seed"],
                "concept_precision": len(hits) / len(labs) if labs else 0.0,
                "concept_recall":    len(cset & C_TRUE) / len(C_TRUE),
                "concept_iou":       len(cset & C_TRUE) / len(cset | C_TRUE) if cset else 0.0,
            })
    cperf = pd.DataFrame(concept_rows)
    ccols = ["concept_precision", "concept_recall", "concept_iou"]
    csummary = cperf.groupby("arm")[ccols].mean()
    csem     = cperf.groupby(["arm", "seed"])[ccols].mean().groupby("arm").sem()
    cperf.to_csv(out_dir / "per_seed_concept_performance.csv", index=False)

    # ── match B features to A features by top-M overlap ──────────────────────
    a_keys = [k for k in desc if k[0] == "A"]
    b_keys = [k for k in desc if k[0] == "B"]
    pairs = []
    for bk in b_keys:
        sb = set(desc[bk]["top_idx"].tolist())
        for ak in a_keys:
            sa = set(desc[ak]["top_idx"].tolist())
            inter = len(sa & sb)
            if inter:
                pairs.append((inter / len(sa | sb), ak, bk))
    pairs.sort(reverse=True)
    used_a, used_b, matches = set(), set(), []
    for jac, ak, bk in pairs:
        if jac < args.jaccard_floor or ak in used_a or bk in used_b:
            continue
        used_a.add(ak); used_b.add(bk)
        matches.append({"neuron_A": ak[1], "neuron_B": bk[1], "jaccard": jac,
                        "concept_A": desc[ak]["concept"], "concept_B": desc[bk]["concept"],
                        "same_concept": desc[ak]["concept"] == desc[bk]["concept"]
                                        and desc[ak]["concept"] != "unlabelled",
                        "is_truth_A": desc[ak]["is_truth"], "is_truth_B": desc[bk]["is_truth"],
                        "runs_A": desc[ak]["n_runs_selected"],
                        "runs_B": desc[bk]["n_runs_selected"]})

    # ── matched-CATE correlation ─────────────────────────────────────────────
    # corr(tau_A, tau_B) over the sampled units, for every run where both members of the
    # pair were recovered; averaged in Fisher-z.
    for mt in matches:
        zs = []
        for r in runs:
            sa = r["arms"]["A"]["slope"].get(mt["neuron_A"])
            sb = r["arms"]["B"]["slope"].get(mt["neuron_B"])
            if sa is None or sb is None or sa == 0.0 or sb == 0.0:
                continue
            idx = r["image_indices"]
            za = feats["A"][idx, mt["neuron_A"]]
            zb = feats["B"][idx, mt["neuron_B"]]
            if za.std() < 1e-12 or zb.std() < 1e-12:
                continue
            rho = float(np.corrcoef(za, zb)[0, 1]) * np.sign(sa * sb)
            zs.append(np.arctanh(np.clip(rho, -0.999999, 0.999999)))
        mt["n_runs_both"] = len(zs)
        mt["cate_corr"] = float(np.tanh(np.mean(zs))) if zs else float("nan")
    match_df = pd.DataFrame(matches)
    match_df.to_csv(out_dir / "matches.csv", index=False)

    # Weight by how often the arm-B feature was actually recovered, so a feature selected
    # in 50/50 runs counts more than an incidental one-off false positive.
    if len(match_df):
        wB = match_df["runs_B"].to_numpy(dtype=float)
        concept_agreement_w = float(np.average(match_df["same_concept"].to_numpy(dtype=float),
                                               weights=wB))
        concept_agreement_u = float(match_df["same_concept"].mean())
        matched_share = float(wB.sum() / sum(sel_counts["B"].values()))
        cc = match_df["cate_corr"].to_numpy(dtype=float)
        ok = np.isfinite(cc)
        cate_corr = (float(np.tanh(np.average(np.arctanh(np.clip(cc[ok], -0.999999, 0.999999)),
                                              weights=wB[ok]))) if ok.any() else float("nan"))
        # restricted to the true modifiers, the quantity the table is really about
        tmask = match_df["is_truth_B"].to_numpy(dtype=bool) & ok
        cate_corr_truth = (float(np.tanh(np.average(
            np.arctanh(np.clip(cc[tmask], -0.999999, 0.999999)), weights=wB[tmask])))
            if tmask.any() else float("nan"))
        concept_agreement_truth = (float(np.average(
            match_df.loc[tmask, "same_concept"].to_numpy(dtype=float), weights=wB[tmask]))
            if tmask.any() else float("nan"))
    else:
        concept_agreement_w = concept_agreement_u = matched_share = float("nan")
        cate_corr = cate_corr_truth = concept_agreement_truth = float("nan")

    # ── report ───────────────────────────────────────────────────────────────
    lines = []
    w = lines.append
    w(f"# CelebA under SAE training-corpus resampling "
      f"(k={K}, sparse codes z, NEXIS)\n")
    w(f"Pooled over {scope}; ± is a seed-clustered standard error.\n")
    w("| SAE training | Precision | Recall | IoU | Concept agreement | Matched-CATE correlation |")
    w("|---|---:|---:|---:|---:|---:|")
    lab = {"A": "Original training sample", "B": "Independent training sample"}
    for arm in ("A", "B"):
        s, e = summary.loc[arm], sem.loc[arm]
        extra = ("— | — |" if arm == "A"
                 else f"{concept_agreement_w:.2f} | {cate_corr:.2f} |")
        w(f"| {lab[arm]} | {s['precision']:.3f} ± {e['precision']:.3f} "
          f"| {s['recall']:.3f} ± {e['recall']:.3f} "
          f"| {s['iou']:.3f} ± {e['iou']:.3f} | {extra}")
    w("")
    w("Same selections scored against the concept set {Wearing_Hat, Eyeglasses} instead of "
      "the F1-argmax neuron indices (removes the split-coordinate artefact):\n")
    w("| SAE training | Precision (concept) | Recall (concept) | IoU (concept) |")
    w("|---|---:|---:|---:|")
    for arm in ("A", "B"):
        s, e = csummary.loc[arm], csem.loc[arm]
        w(f"| {lab[arm]} | {s['concept_precision']:.3f} ± {e['concept_precision']:.3f} "
          f"| {s['concept_recall']:.3f} ± {e['concept_recall']:.3f} "
          f"| {s['concept_iou']:.3f} ± {e['concept_iou']:.3f} |")
    w("")
    w(f"- Concept agreement, unweighted over matched pairs: {concept_agreement_u:.2f} "
      f"({int(match_df['same_concept'].sum()) if len(match_df) else 0}/{len(match_df)} pairs); "
      f"restricted to true modifiers: {concept_agreement_truth:.2f}.")
    w(f"- Share of arm-B recoveries that found a match in arm A "
      f"(weighted by recovery frequency): {matched_share:.2f}.")
    w(f"- Matched-CATE correlation restricted to true modifiers: {cate_corr_truth:.2f}.")
    w(f"- Mean features selected per run: A {summary.loc['A','n_selected']:.2f}, "
      f"B {summary.loc['B','n_selected']:.2f}.  |S*| = "
      f"{len(truth['A'])} (A), {len(truth['B'])} (B).")
    w("")
    w("## Matched features\n")
    w("| A neuron | B neuron | Jaccard(top-M) | concept A | concept B | same | CATE corr | runs A | runs B |")
    w("|---|---|---:|---|---|---|---:|---:|---:|")
    for _, r in match_df.sort_values("runs_B", ascending=False).iterrows():
        w(f"| {r['neuron_A']} | {r['neuron_B']} | {r['jaccard']:.2f} | {r['concept_A']} | "
          f"{r['concept_B']} | {'yes' if r['same_concept'] else 'no'} | "
          f"{r['cate_corr']:.2f} | {r['runs_A']} | {r['runs_B']} |")

    unmatched = [d for (arm, j), d in desc.items()
                 if arm == "B" and (arm, j) not in used_b]
    if unmatched:
        w("\n## Arm-B recoveries with no arm-A counterpart\n")
        w("| B neuron | concept | top-M purity | lift | runs selected | in S*_B |")
        w("|---|---|---:|---:|---:|---|")
        for d in sorted(unmatched, key=lambda d: -d["n_runs_selected"]):
            w(f"| {d['neuron']} | {d['concept']} | {d['purity']:.2f} | {d['lift']:.1f} | "
              f"{d['n_runs_selected']} | {'yes' if d['is_truth'] else 'no'} |")

    (out_dir / "headline_table.md").write_text("\n".join(lines) + "\n")
    summary.assign(sem_precision=sem["precision"], sem_recall=sem["recall"],
                   sem_iou=sem["iou"],
                   concept_agreement=[np.nan, concept_agreement_w][:len(summary)],
                   ).to_csv(out_dir / "headline_table.csv")
    perf.to_csv(out_dir / "per_seed_performance.csv", index=False)
    print("\n".join(lines))
    print(f"\nWrote {out_dir}/headline_table.md (+ selections.csv, matches.csv)")

    if args.contact_sheets:
        write_contact_sheets(desc, data_dir, out_dir)
    return 0


def write_contact_sheets(desc, data_dir: Path, out_dir: Path, n_show: int = 16) -> None:
    """Save a strip of top-activating thumbnails per recovered feature (for VLM / eyeball)."""
    img_path = data_dir / "images.npy"
    if not img_path.exists():
        print(f"[skip contact sheets] {img_path} not found")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    imgs = np.load(img_path, mmap_mode="r")
    for (arm, j), d in desc.items():
        if d["n_runs_selected"] < 2:
            continue
        idx = d["top_idx"][:n_show]
        fig, axes = plt.subplots(1, len(idx), figsize=(len(idx) * 0.9, 1.35))
        for ax, i in zip(np.atleast_1d(axes), idx):
            ax.imshow(np.asarray(imgs[i])); ax.axis("off")
        fig.suptitle(f"arm {arm} · neuron {j} · {d['concept']} "
                     f"(purity {d['purity']:.2f}, lift {d['lift']:.1f}, "
                     f"{d['n_runs_selected']} runs)", fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"topk_{arm}_{j}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote contact sheets → {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
