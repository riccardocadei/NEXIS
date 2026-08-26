#!/usr/bin/env python3
"""
Stage 3 — Run the CelebA semi-synthetic experiment.

Reads:   data/celeba/embeddings/sae_k{K}.npy        (N, H) — sparse post-topk codes (default)
      or data/celeba/embeddings/sae_precode_k{K}.npy (N, H) — continuous pre-activations (--precode)
      or data/celeba/embeddings/{backbone}.npy        (N, D) — raw embeddings (--raw)
         data/celeba/labels.parquet                   CelebA attribute labels

Writes:  results/celeba/experiment/raw/ground_truth.json            (raw, K-independent)
         results/celeba/experiment/k{K}/sae/ground_truth.json       (SAE sparse codes)
         results/celeba/experiment/k{K}/sae_precode/ground_truth.json
         (and corresponding effect_sweep.parquet / n_sweep.parquet files)

With --backbone dinov2 (or any non-SigLIP backbone) both the input file names and
the results tree are tagged, e.g. data/celeba/embeddings/sae_dinov2_k20.npy and
results/celeba/experiment/dinov2/k20/sae/.

Feature modes for SAE (--precode controls NEXIS regression features):
  default  — sparse post-topk codes (sae.npy): features are ~orthogonal by design;
             NEXIS ≈ Bonferroni because conditioning on orthogonal features adds no power.
  --precode — continuous pre-activations (sae_precode.npy): features are dense and
             correlated; NEXIS conditioning removes false positives that Bonferroni misses.

Two sweeps are produced (mimicking Fig. 5 of the ECI paper):
  effect_scale_sweep — fix n, vary heterogeneity strength (0 → type-I, >0 → power)
  n_sweep            — fix effect_scale, vary sample size

Methods compared
  NEXIS               — sequential conditional testing (Bonferroni-gated)
  Marginal           — marginal interaction test, no correction
  Marginal (Bonf.)   — marginal interaction test, global Bonferroni

Usage
-----
    python src/apps/celeba/run_experiment.py                          # SAE sparse codes, k=5
    python src/apps/celeba/run_experiment.py --sae-top-k 20          # SAE sparse codes, k=20
    python src/apps/celeba/run_experiment.py --precode               # SAE pre-activations, k=5
    python src/apps/celeba/run_experiment.py --precode --sae-top-k 20
    python src/apps/celeba/run_experiment.py --raw                   # raw SigLIP embeddings
    python src/apps/celeba/run_experiment.py --w1-attr Eyeglasses --w2-attr Wearing_Hat
    python src/apps/celeba/run_experiment.py --n-seeds 20 --max-steps 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent

from apps.celeba.scm import build_buckets
from apps.celeba.experiment import (
    find_ground_truth_neurons_multi,
    find_ground_truth_neurons, compute_f1_scores, run_sweep,
    ALL_METHODS,
)
from apps.celeba.backbones import (
    BACKBONES, DEFAULT_BACKBONE, get_backbone,
    embed_path as _embed_path, code_path as _code_path,
    precode_path as _precode_path, experiment_dir as _experiment_dir,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir",     type=Path, default="data/celeba")
    p.add_argument("--out-dir",      type=Path, default="results/celeba/experiment")
    p.add_argument("--backbone",     default=DEFAULT_BACKBONE, choices=sorted(BACKBONES),
                   help=f"Frozen backbone behind the SAE (default: {DEFAULT_BACKBONE}). "
                        "Non-default backbones write to out-dir/{backbone}/…")
    p.add_argument("--raw",          action="store_true",
                   help="Use raw backbone embeddings instead of SAE features. "
                        "Results go to out-dir/raw/ (default: out-dir/sae/)")
    p.add_argument("--precode",      action="store_true",
                   help="Use continuous SAE pre-activations (sae_precode_k{K}.npy) for NEXIS "
                        "regression instead of sparse post-topk codes (sae_k{K}.npy). "
                        "Results go to out-dir/k{K}/sae_precode/. "
                        "Pre-activations are dense and correlated, so NEXIS conditioning "
                        "suppresses false positives that Bonferroni misses. "
                        "Ignored when --raw is set.")
    p.add_argument("--sae-top-k",   type=int, default=5,
                   help="Top-k sparsity used when training the SAE (default: 5). "
                        "Selects which sae_k{K}.npy / sae_precode_k{K}.npy files to load "
                        "and organises results under out-dir/k{K}/. Ignored when --raw is set.")
    # Attributes
    p.add_argument("--w1-attr",      default="Wearing_Hat",
                   help="CelebA attribute for W1 (default: Wearing_Hat, prevalence≈5%%)")
    p.add_argument("--w2-attr",      default="Eyeglasses",
                   help="CelebA attribute for W2 (default: Eyeglasses, prevalence≈7%%)")
    p.add_argument("--w-attrs",      nargs='+', default=None, metavar="ATTR",
                   help="Explicit list of r ground-truth modifier attributes, "
                        "overriding --w1-attr/--w2-attr. Use to enlarge the truth set: "
                        "e.g. --w-attrs Wearing_Hat Eyeglasses Sideburns for |S*|=3. "
                        "Each attribute must be carried by one dominant SAE coordinate "
                        "(the run logs the F1 gap to the runner-up) and the all-ones "
                        "joint cell must hold enough images for the sampler.")
    p.add_argument("--gammas",       type=float, nargs='+', default=None,
                   help="Length-r T x W_k interaction coefficients. Defaults to "
                        "--gamma-w1/--gamma-w2 at r=2 and to alternating +1/-1 beyond.")
    p.add_argument("--betas",        type=float, nargs='+', default=None,
                   help="Length-r main effects of W_k on Y. Defaults to 0.3/-0.2 at "
                        "r=2 and to an alternating 0.3/-0.2 pattern beyond.")
    p.add_argument("--top-k",        type=int, default=1,
                   help="Top-k neurons per attribute as ground truth (default: 1)")
    p.add_argument("--gt-json",       type=Path, default=None,
                   help="Reuse the ground-truth neurons / F1 spectra from an existing "
                        "ground_truth.json (same dictionary, same labels) instead of "
                        "recomputing them. Only affects runtime, not results.")
    # Experiment design
    p.add_argument("--n-seeds",      type=int, default=50)
    p.add_argument("--seed-offset",  type=int, default=0,
                   help="First Monte Carlo seed (default: 0). Seeds run over "
                        "[seed-offset, seed-offset + n-seeds). Use to shard one sweep "
                        "over disjoint seed blocks in parallel jobs, then concatenate.")
    p.add_argument("--alpha",        type=float, default=0.05)
    p.add_argument("--max-steps",    type=int, default=5,
                   help="Max NEXIS selection steps (default: 5)")
    p.add_argument("--methods",      nargs='+', default=None,
                   help=f"Methods to run (default: all). Choices: {ALL_METHODS}")
    p.add_argument("--gcm-splits",   type=int, default=3,
                   help="Cross-fit folds for GCM methods (default: 3, faster than 5)")
    # Fixed values for each sweep (multiple values → one row per value in plots)
    p.add_argument("--fixed-n",      type=int,   nargs='+', default=[500, 2000],
                   help="n values used in effect-size sweep (default: 500 2000)")
    p.add_argument("--fixed-effect", type=float, nargs='+', default=[1.0, 3.0],
                   help="effect_scale values used in n sweep (default: 1.0 3.0)")
    p.add_argument("--effect-grid",  type=float, nargs='+', default=None,
                   help="Override the effect-size grid (default: 1 … 10). "
                        "Pass 0 to include the type-I-error control point.")
    p.add_argument("--n-grid",       type=int,   nargs='+', default=None,
                   help="Override the sample-size grid (default: 50 … 10000).")
    # SCM parameters (override defaults)
    p.add_argument("--tau0",         type=float, default=0.5)
    p.add_argument("--gamma-w1",     type=float, default=1.0)
    p.add_argument("--gamma-w2",     type=float, default=-1.0)
    p.add_argument("--noise-sd",     type=float, default=1.0)
    p.add_argument("--effect-form",  choices=["attr", "ortho_quadratic"], default="attr",
                   help="Shape of the treatment-effect modification. 'attr' (default) "
                        "makes τ linear in the binary attributes W1/W2. "
                        "'ortho_quadratic' makes τ a U-shape in the two ground-truth "
                        "coordinates, constructed to have exactly zero population "
                        "covariance with Z^j — a conditional-mean alternative that the "
                        "linear test and the GCM are blind to by construction. Only "
                        "meaningful with --precode or --raw (continuous features).")
    p.add_argument("--sweep",        choices=["effect", "n", "both"], default="both",
                   help="Which sweep to run (default: both)")
    p.add_argument("--force",        action="store_true")
    p.add_argument("--merge",        action="store_true",
                   help="Merge new method results into existing parquet files instead of "
                        "overwriting. Rows for the methods being run are replaced; all "
                        "other existing rows are kept. Implies --force.")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = (ROOT / args.data_dir
                if not args.data_dir.is_absolute() else args.data_dir)
    base_out = (ROOT / args.out_dir
                if not args.out_dir.is_absolute() else args.out_dir)

    # Non-default backbones get their own results subtree; SigLIP keeps the flat layout
    spec     = get_backbone(args.backbone)
    base_out = _experiment_dir(base_out, spec.key)

    # Sub-directory depends on feature type and SAE top-k so all runs coexist
    if args.raw:
        out_dir = base_out / "raw"
    elif args.precode:
        out_dir = base_out / f"k{args.sae_top_k}" / "sae_precode"
    else:
        out_dir = base_out / f"k{args.sae_top_k}" / "sae"
    out_dir.mkdir(parents=True, exist_ok=True)

    effect_path = out_dir / "effect_sweep.parquet"
    n_path      = out_dir / "n_sweep.parquet"
    if args.merge:
        args.force = True  # merge implies force
    if args.sweep in ("effect", "both") and effect_path.exists() and not args.force:
        print("Effect sweep results already exist. Use --force to rerun.")
        if args.sweep == "effect":
            return
    if args.sweep in ("n", "both") and n_path.exists() and not args.force:
        print("N sweep results already exist. Use --force to rerun.")
        if args.sweep == "n":
            return

    def _merge_parquet(path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
        """Keep existing rows for methods not in new_df; replace rows that are."""
        if not path.exists():
            return new_df
        old = pd.read_parquet(path)
        keep = old[~old["method"].isin(new_df["method"].unique())]
        return pd.concat([keep, new_df], ignore_index=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    k = args.sae_top_k
    if args.raw:
        reg_file      = _embed_path(data_dir, spec.key)
        precode_file  = None
        feat_label    = f"raw {spec.label} embeddings"
    elif args.precode:
        reg_file      = _precode_path(data_dir, spec.key, k)
        precode_file  = _precode_path(data_dir, spec.key, k)
        feat_label    = f"{spec.label} SAE pre-activations (z_pre, continuous, k={k})"
    else:
        reg_file      = _code_path(data_dir, spec.key, k)
        precode_file  = _precode_path(data_dir, spec.key, k)
        feat_label    = f"{spec.label} SAE sparse codes (z, post-topk, k={k})"

    print(f"Loading {feat_label} from {reg_file} …")
    labels_df = pd.read_parquet(data_dir / "labels.parquet")

    # Features used for NEXIS regression
    features = np.load(reg_file)
    print(f"  NEXIS regression features: {features.shape}  "
          f"sparsity={(features == 0).mean():.3f}")

    # F1 ground-truth evaluation always uses z_pre (continuous pre-activations):
    # best-threshold sweep gives higher, cleaner F1 (~0.8 vs ~0.01 for sparse codes).
    if precode_file is not None and precode_file.exists():
        precode_features = np.load(precode_file)
        print(f"  GT F1 features (z_pre, continuous): {precode_features.shape}")
    else:
        precode_features = None

    print(f"Labels: {labels_df.shape}")

    # Resolve the modifier attribute list once: --w-attrs wins, otherwise the
    # legacy (--w1-attr, --w2-attr) pair.  r = len(w_attrs) drives everything
    # downstream (ground truth, buckets, gammas, |S*|).
    w_attrs = list(args.w_attrs) if args.w_attrs else [args.w1_attr, args.w2_attr]
    if len(w_attrs) < 2:
        raise ValueError("--w-attrs needs at least two attributes")
    if len(set(w_attrs)) != len(w_attrs):
        raise ValueError(f"--w-attrs contains duplicates: {w_attrs}")
    for attr in w_attrs:
        if attr not in labels_df.columns:
            raise ValueError(
                f"Attribute '{attr}' not found in labels. "
                f"Available: {list(labels_df.columns)}"
            )
    for name, vals in (("--gammas", args.gammas), ("--betas", args.betas)):
        if vals is not None and len(vals) != len(w_attrs):
            raise ValueError(f"{name} has {len(vals)} values but r={len(w_attrs)}")
    print(f"Ground-truth modifiers (r={len(w_attrs)}): {w_attrs}")

    # ── Ground truth ──────────────────────────────────────────────────────────
    # GT neuron identification always uses precode (continuous scores give cleaner F1).
    # --gt-json reuses a ground_truth.json produced by an earlier identical run: the
    # F1 spectra depend only on (features, labels), so sharded jobs over the same
    # dictionary can skip recomputing them (6 full 9216-neuron threshold sweeps).
    if args.gt_json is not None:
        gt_path = (ROOT / args.gt_json if not args.gt_json.is_absolute() else args.gt_json)
        print(f"\nReusing ground truth from {gt_path}")
        with open(gt_path) as f:
            gt = json.load(f)
        neurons_per_attr = gt.get(
            "neurons_per_attr", [gt["w1_neurons"], gt["w2_neurons"]])
        w1_neurons, w2_neurons = neurons_per_attr[0], neurons_per_attr[1]
        truth = sorted(set().union(*(set(v) for v in neurons_per_attr)))
        for a, v in zip(gt.get("w_attrs", [args.w1_attr, args.w2_attr]),
                        neurons_per_attr):
            print(f"  {a:20s}: {v}")
        print(f"  Truth set  : {truth}  (size={len(truth)})")
        # Copy into out_dir only when it is a different file: --gt-json commonly
        # points at out_dir/ground_truth.json itself, and rewriting it there races
        # with sibling jobs sharing the same results tree.
        dest = out_dir / "ground_truth.json"
        if gt_path.resolve() != dest.resolve():
            with open(dest, "w") as f:
                json.dump(gt, f, indent=2)
    else:
        gt_features = precode_features if precode_features is not None else features
        print(f"\nFinding ground truth neurons  "
              f"(attrs={w_attrs}, top_k={args.top_k})…")
        neurons_per_attr, f1_per_attr = find_ground_truth_neurons_multi(
            gt_features, labels_df, w_attrs, top_k=args.top_k)
        w1_neurons, w2_neurons = neurons_per_attr[0], neurons_per_attr[1]
        truth = sorted(set().union(*(set(v) for v in neurons_per_attr)))
        # The F1 gap to the runner-up is the empirical Principal Alignment check:
        # a small gap means the concept is smeared over several coordinates and the
        # argmax target is arbitrary, so precision penalises target misspecification
        # rather than the algorithm.  Log it so a bad attribute choice is visible.
        for a, v, f in zip(w_attrs, neurons_per_attr, f1_per_attr):
            order = np.argsort(f)[::-1]
            print(f"  {a:20s}: j*={v[-1]:5d}  F1={f[order[0]]:.3f}  "
                  f"runner-up={f[order[1]]:.3f}  gap={f[order[0]] - f[order[1]]:+.3f}")
        if len(truth) < sum(len(v) for v in neurons_per_attr):
            print(f"  WARNING: attributes share a principal coordinate — "
                  f"|truth|={len(truth)} < r*top_k="
                  f"{sum(len(v) for v in neurons_per_attr)}")
        print(f"  Truth set  : {truth}  (size={len(truth)})")

        # F1 spectra cached for plot_importance.
        # "reg" scores use regression features (z for code run, z_pre for precode/raw).
        # "precode" scores use continuous z_pre when available (always cleanest).
        f1_w1_z = compute_f1_scores(features, labels_df[w_attrs[0]].values.astype(float))
        f1_w2_z = compute_f1_scores(features, labels_df[w_attrs[1]].values.astype(float))
        if precode_features is not None:
            f1_w1_zpre = compute_f1_scores(precode_features, labels_df[w_attrs[0]].values.astype(float))
            f1_w2_zpre = compute_f1_scores(precode_features, labels_df[w_attrs[1]].values.astype(float))
        else:
            f1_w1_zpre, f1_w2_zpre = f1_w1_z, f1_w2_z

        with open(out_dir / "ground_truth.json", "w") as f:
            json.dump({
                "backbone": spec.key,
                "feature_type": "raw" if args.raw else ("sae_precode" if args.precode else "sae"),
                "w1_attr": w_attrs[0],    "w2_attr": w_attrs[1],
                "w_attrs": w_attrs,
                "top_k": args.top_k,
                "w1_neurons": w1_neurons,  "w2_neurons": w2_neurons,
                "neurons_per_attr": [list(map(int, v)) for v in neurons_per_attr],
                "truth": truth,
                "w1_f1_scores":      f1_w1_z.tolist(),
                "w2_f1_scores":      f1_w2_z.tolist(),
                "w1_f1_scores_pre":  f1_w1_zpre.tolist(),
                "w2_f1_scores_pre":  f1_w2_zpre.tolist(),
            }, f, indent=2)

    # ── Bucket sizes ──────────────────────────────────────────────────────────
    buckets = build_buckets(labels_df, w_attrs=w_attrs)
    print("\nBucket sizes:")
    # The sampler draws attributes independently and then needs a real image in the
    # matching joint cell, so the binding constraint is min_cell(count / cell prob),
    # not the smallest cell.  Report it: it is the largest n this attribute set can
    # support, and it collapses fast as r grows.
    p_w = np.array([float(labels_df[a].mean()) for a in w_attrs])
    n_cap = np.inf
    for k, v in sorted(buckets.items()):
        q = float(np.prod(np.where(np.array(k) == 1, p_w, 1.0 - p_w)))
        cap = len(v) / q if q > 0 else np.inf
        n_cap = min(n_cap, cap)
        cell = ", ".join(f"{a}={b}" for a, b in zip(w_attrs, k))
        print(f"  {cell}: {len(v):,} images   (supports n <= {cap:,.0f})")
    print(f"  → largest sample size this attribute set supports: n <= {n_cap:,.0f}")

    scm_kwargs = dict(
        w_attrs=w_attrs,
        tau_0=args.tau0,
        gamma_w1=args.gamma_w1,
        gamma_w2=args.gamma_w2,
        noise_sd=args.noise_sd,
        effect_form=args.effect_form,
    )
    if args.gammas is not None:
        scm_kwargs["gammas"] = args.gammas
    if args.betas is not None:
        scm_kwargs["betas"] = args.betas
    if args.effect_form != "attr":
        # τ is driven by the ground-truth coordinates themselves, so S* is exact
        # by construction rather than inherited from the attribute alignment.
        scm_kwargs["modifier_cols"] = [int(w1_neurons[-1]), int(w2_neurons[-1])]
        print(f"\nEffect form: {args.effect_form}  "
              f"modifier_cols={scm_kwargs['modifier_cols']}")

    methods = args.methods  # None → all methods

    # ── Effect-size sweep (one sub-sweep per fixed n) ─────────────────────────
    if args.sweep in ("effect", "both"):
        effect_grid = args.effect_grid or [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        dfs_effect = []
        for fixed_n in args.fixed_n:
            print(f"\n=== Effect-size sweep  n={fixed_n}  seeds={args.n_seeds} ===")
            df = run_sweep(
                features, labels_df, buckets, truth,
                sweep_param="effect_scale",
                param_grid=effect_grid,
                fixed_n=fixed_n,
                n_seeds=args.n_seeds,
                seed_offset=args.seed_offset,
                alpha=args.alpha,
                max_rounds=args.max_steps,
                methods=methods,
                gcm_splits=args.gcm_splits,
                **scm_kwargs,
            )
            dfs_effect.append(df)
        df_effect = pd.concat(dfs_effect, ignore_index=True)
        if args.merge:
            df_effect = _merge_parquet(effect_path, df_effect)
        df_effect.to_parquet(effect_path, index=False)
        print(f"Effect sweep: {len(df_effect)} rows  →  {effect_path}")

    # ── Sample-size sweep (one sub-sweep per fixed effect) ────────────────────
    if args.sweep in ("n", "both"):
        n_grid = args.n_grid or [50, 100, 200, 350, 500, 750, 1000, 2000, 3500, 5000, 10000]
        dfs_n = []
        for fixed_effect in args.fixed_effect:
            print(f"\n=== Sample-size sweep  effect={fixed_effect}  seeds={args.n_seeds} ===")
            df = run_sweep(
                features, labels_df, buckets, truth,
                sweep_param="n",
                param_grid=n_grid,
                fixed_effect=fixed_effect,
                n_seeds=args.n_seeds,
                seed_offset=args.seed_offset,
                alpha=args.alpha,
                max_rounds=args.max_steps,
                methods=methods,
                gcm_splits=args.gcm_splits,
                **scm_kwargs,
            )
            dfs_n.append(df)
        df_n = pd.concat(dfs_n, ignore_index=True)
        if args.merge:
            df_n = _merge_parquet(n_path, df_n)
        df_n.to_parquet(n_path, index=False)
        print(f"N sweep:      {len(df_n)} rows  →  {n_path}")

    print(f"\nDone ({feat_label}).  Run notebooks/celeba_semisynthetic.ipynb to visualise.")


if __name__ == "__main__":
    main()
