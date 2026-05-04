#!/usr/bin/env python3
"""
Stage 3 — Run the CelebA semi-synthetic experiment.

Reads:   data/celeba/embeddings/sae_k{K}.npy        (N, H) — sparse post-topk codes (default)
      or data/celeba/embeddings/sae_precode_k{K}.npy (N, H) — continuous pre-activations (--precode)
      or data/celeba/embeddings/siglip.npy            (N, D) — raw SigLIP (--raw)
         data/celeba/labels.parquet                   CelebA attribute labels

Writes:  results/celeba/experiment/raw/ground_truth.json            (raw, K-independent)
         results/celeba/experiment/k{K}/sae/ground_truth.json       (SAE sparse codes)
         results/celeba/experiment/k{K}/sae_precode/ground_truth.json
         (and corresponding effect_sweep.parquet / n_sweep.parquet files)

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
    find_ground_truth_neurons, compute_f1_scores, run_sweep,
    ALL_METHODS,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir",     type=Path, default="data/celeba")
    p.add_argument("--out-dir",      type=Path, default="results/celeba/experiment")
    p.add_argument("--raw",          action="store_true",
                   help="Use raw SigLIP embeddings instead of SAE features. "
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
    p.add_argument("--top-k",        type=int, default=1,
                   help="Top-k neurons per attribute as ground truth (default: 1)")
    # Experiment design
    p.add_argument("--n-seeds",      type=int, default=50)
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
    # SCM parameters (override defaults)
    p.add_argument("--tau0",         type=float, default=0.5)
    p.add_argument("--gamma-w1",     type=float, default=1.0)
    p.add_argument("--gamma-w2",     type=float, default=-1.0)
    p.add_argument("--noise-sd",     type=float, default=1.0)
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
    if effect_path.exists() and n_path.exists() and not args.force:
        print("Results already exist. Use --force to rerun.")
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
        reg_file      = data_dir / "embeddings" / "siglip.npy"
        precode_file  = None
        feat_label    = "raw SigLIP embeddings"
    elif args.precode:
        reg_file      = data_dir / "embeddings" / f"sae_precode_k{k}.npy"
        precode_file  = data_dir / "embeddings" / f"sae_precode_k{k}.npy"
        feat_label    = f"SAE pre-activations (z_pre, continuous, k={k})"
    else:
        reg_file      = data_dir / "embeddings" / f"sae_k{k}.npy"
        precode_file  = data_dir / "embeddings" / f"sae_precode_k{k}.npy"
        feat_label    = f"SAE sparse codes (z, post-topk, k={k})"

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

    for attr in [args.w1_attr, args.w2_attr]:
        if attr not in labels_df.columns:
            raise ValueError(
                f"Attribute '{attr}' not found in labels. "
                f"Available: {list(labels_df.columns)}"
            )

    # ── Ground truth ──────────────────────────────────────────────────────────
    # GT neuron identification always uses precode (continuous scores give cleaner F1).
    gt_features = precode_features if precode_features is not None else features
    print(f"\nFinding ground truth neurons  "
          f"(W1={args.w1_attr}, W2={args.w2_attr}, top_k={args.top_k})…")
    w1_neurons, w2_neurons = find_ground_truth_neurons(
        gt_features, labels_df,
        w1_attr=args.w1_attr,
        w2_attr=args.w2_attr,
        top_k=args.top_k,
    )
    truth = sorted(set(w1_neurons) | set(w2_neurons))
    print(f"  W1 neurons : {w1_neurons}")
    print(f"  W2 neurons : {w2_neurons}")
    print(f"  Truth set  : {truth}  (size={len(truth)})")

    # F1 spectra cached for plot_importance.
    # "reg" scores use regression features (z for code run, z_pre for precode/raw).
    # "precode" scores use continuous z_pre when available (always cleanest).
    f1_w1_z = compute_f1_scores(features, labels_df[args.w1_attr].values.astype(float))
    f1_w2_z = compute_f1_scores(features, labels_df[args.w2_attr].values.astype(float))
    if precode_features is not None:
        f1_w1_zpre = compute_f1_scores(precode_features, labels_df[args.w1_attr].values.astype(float))
        f1_w2_zpre = compute_f1_scores(precode_features, labels_df[args.w2_attr].values.astype(float))
    else:
        f1_w1_zpre, f1_w2_zpre = f1_w1_z, f1_w2_z

    with open(out_dir / "ground_truth.json", "w") as f:
        json.dump({
            "feature_type": "raw" if args.raw else ("sae_precode" if args.precode else "sae"),
            "w1_attr": args.w1_attr,   "w2_attr": args.w2_attr,
            "top_k": args.top_k,
            "w1_neurons": w1_neurons,  "w2_neurons": w2_neurons,
            "truth": truth,
            "w1_f1_scores":      f1_w1_z.tolist(),
            "w2_f1_scores":      f1_w2_z.tolist(),
            "w1_f1_scores_pre":  f1_w1_zpre.tolist(),
            "w2_f1_scores_pre":  f1_w2_zpre.tolist(),
        }, f, indent=2)

    # ── Bucket sizes ──────────────────────────────────────────────────────────
    buckets = build_buckets(labels_df, args.w1_attr, args.w2_attr)
    print("\nBucket sizes:")
    for k, v in sorted(buckets.items()):
        print(f"  {args.w1_attr}={k[0]}, {args.w2_attr}={k[1]}: {len(v):,} images")

    scm_kwargs = dict(
        w1_attr=args.w1_attr,
        w2_attr=args.w2_attr,
        tau_0=args.tau0,
        gamma_w1=args.gamma_w1,
        gamma_w2=args.gamma_w2,
        noise_sd=args.noise_sd,
    )

    methods = args.methods  # None → all methods

    # ── Effect-size sweep (one sub-sweep per fixed n) ─────────────────────────
    effect_grid = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    dfs_effect = []
    for fixed_n in args.fixed_n:
        print(f"\n=== Effect-size sweep  n={fixed_n}  seeds={args.n_seeds} ===")
        df = run_sweep(
            features, labels_df, buckets, truth,
            sweep_param="effect_scale",
            param_grid=effect_grid,
            fixed_n=fixed_n,
            n_seeds=args.n_seeds,
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
    n_grid = [50, 100, 250, 500, 1000, 2000, 5000, 10000]
    dfs_n = []
    for fixed_effect in args.fixed_effect:
        print(f"\n=== Sample-size sweep  effect={fixed_effect}  seeds={args.n_seeds} ===")
        df = run_sweep(
            features, labels_df, buckets, truth,
            sweep_param="n",
            param_grid=n_grid,
            fixed_effect=fixed_effect,
            n_seeds=args.n_seeds,
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
