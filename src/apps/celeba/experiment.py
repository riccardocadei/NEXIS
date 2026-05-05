"""
CelebA semi-synthetic experiment runner.

Ground truth
------------
SAE neurons are ranked by F1 score (using Zj > 0 as a binary predictor)
against W1 and W2 computed over all CelebA images.  The single best neuron
per attribute (top-1 by default) forms the truth set used to evaluate
recall, precision, and IoU for each selection method.

Sweeps
------
  effect_scale_sweep : fix n, vary effect_scale ∈ [0, …, 3]
                       effect_scale = 0 ↔ type-I-error control check
  n_sweep            : fix effect_scale, vary n ∈ [100, …, 2000]

Each sweep repeats over n_seeds random draws to average out sampling noise.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from method.nexis import nexis, marginal_select, iou_score
from apps.celeba.scm import CelebAData, build_buckets, generate_celeba_rct


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

#: Canonical ordered list of all methods used in experiments.
#: Each NEXIS variant is characterised by the quadruplet
#: (test, adjust, rho, backward); "NEXIS" uses the defaults
#: (linear, FWER, 0.5, True) and variants name only what changed.
ALL_METHODS: List[str] = [
    # Baselines
    "Marginal Testing",
    "Marginal Testing (FWER)",
    "Marginal Testing (FDR)",
    # NEXIS default
    "NEXIS",
    # test ablation
    "NEXIS (test=GCM: quadratic)",
    "NEXIS (test=GCM: lgbm)",
    # adjust ablation
    "NEXIS (adjust=None)",
    "NEXIS (adjust=FDR)",
    # rho ablation
    "NEXIS (rho=0)",
    "NEXIS (rho=0.2)",
    "NEXIS (rho=0.8)",
    # backward ablation
    "NEXIS (backward=False)",
]

#: Fast-only subset (skips GCM lgbm, ~3× slower) for compute-tight runs.
FAST_METHODS: List[str] = [m for m in ALL_METHODS if m != "NEXIS (test=GCM: lgbm)"]


def evaluate_methods_on_dataset(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    truth: Sequence[int],
    alpha: float = 0.05,
    max_rounds: Optional[int] = None,
    methods: Optional[List[str]] = None,
    gcm_splits: int = 3,
) -> Dict[str, Dict[str, float]]:
    """Run all (or a subset of) selection methods and return per-method metrics.

    Each entry in the returned dict includes: iou, recall, precision, tp, fp,
    n_selected, time_s.

    methods: subset of ALL_METHODS to run (default: all).
    gcm_splits: cross-fit folds for GCM methods (3 is faster than 5, minimal loss).
    """
    if methods is None:
        methods = ALL_METHODS

    truth_set = set(int(x) for x in truth)
    n_truth = len(truth_set)

    def _metrics(selected: Sequence[int], t_s: float) -> Dict[str, float]:
        sel_set = set(int(x) for x in selected)
        tp = float(len(sel_set & truth_set))
        fp = float(len(sel_set - truth_set))
        n_sel = float(len(sel_set))
        recall = tp / n_truth if n_truth > 0 else 1.0
        precision = (tp / n_sel) if n_sel > 0 else (1.0 if n_truth == 0 else 0.0)
        return {
            "iou": iou_score(sel_set, truth_set),
            "n_selected": n_sel,
            "tp": tp,
            "fp": fp,
            "recall": float(recall),
            "precision": float(precision),
            "time_s": t_s,
        }

    out: Dict[str, Dict[str, float]] = {}

    def _run(name: str, fn):
        if name not in methods:
            return
        t0 = time.perf_counter()
        res = fn()
        out[name] = _metrics(res.selected, time.perf_counter() - t0)

    _run("Marginal Testing",
         lambda: marginal_select(y=y, t=t, z=z, alpha=alpha, adjust=None))
    _run("Marginal Testing (FWER)",
         lambda: marginal_select(y=y, t=t, z=z, alpha=alpha, adjust="FWER"))
    _run("Marginal Testing (FDR)",
         lambda: marginal_select(y=y, t=t, z=z, alpha=alpha, adjust="FDR"))
    # NEXIS default: test=linear, adjust=FWER, rho=0.5, backward=True
    _run("NEXIS",
         lambda: nexis(y=y, t=t, z=z, alpha=alpha, max_rounds=max_rounds))
    # test ablation
    _run("NEXIS (test=GCM: quadratic)",
         lambda: nexis(y=y, t=t, z=z, alpha=alpha, max_rounds=max_rounds,
                      test="GCM: quadratic", n_splits=gcm_splits))
    _run("NEXIS (test=GCM: lgbm)",
         lambda: nexis(y=y, t=t, z=z, alpha=alpha, max_rounds=max_rounds,
                      test="GCM: lgbm", n_splits=gcm_splits))
    # adjust ablation
    _run("NEXIS (adjust=None)",
         lambda: nexis(y=y, t=t, z=z, alpha=alpha, max_rounds=max_rounds,
                      adjust=None))
    _run("NEXIS (adjust=FDR)",
         lambda: nexis(y=y, t=t, z=z, alpha=alpha, max_rounds=max_rounds,
                      adjust="FDR"))
    # rho ablation
    _run("NEXIS (rho=0)",
         lambda: nexis(y=y, t=t, z=z, alpha=alpha, max_rounds=max_rounds,
                      rho=0))
    _run("NEXIS (rho=0.8)",
         lambda: nexis(y=y, t=t, z=z, alpha=alpha, max_rounds=max_rounds,
                      rho=0.8))
    _run("NEXIS (rho=0.2)",
         lambda: nexis(y=y, t=t, z=z, alpha=alpha, max_rounds=max_rounds,
                      rho=0.2))
    # backward ablation
    _run("NEXIS (backward=False)",
         lambda: nexis(y=y, t=t, z=z, alpha=alpha, max_rounds=max_rounds,
                      backward=False))

    return out


# ---------------------------------------------------------------------------
# Ground truth identification
# ---------------------------------------------------------------------------

def compute_f1_scores(
    features: np.ndarray,
    labels: np.ndarray,
    chunk_size: int = 2048,
) -> np.ndarray:
    """
    Best-threshold F1 per neuron by sweeping all possible thresholds on the
    continuous pre-activation scores (z_pre).  Equivalent to the ECI paper's
    ``prf_best_on_scores_batched``.

    For each neuron j, sorts images by descending z_pre_j and finds the threshold
    position that maximises F1 for predicting the binary ``labels``.  This gives
    much higher F1 (~0.8) for principally-aligned concept neurons than the fixed
    ``z_j > 0`` rule (which yields ~0.01 because only ≈5 neurons fire per image).

    Args:
        features:   (N, hidden_dim) continuous pre-activation matrix (z_pre).
        labels:     (N,) binary {0, 1} attribute vector.
        chunk_size: Number of neurons to process per chunk (memory control).

    Returns:
        f1: (hidden_dim,) array of best-threshold F1 scores.
    """
    N, D = features.shape
    y = labels.astype(np.int32)          # (N,)
    total_pos = int(y.sum())
    if total_pos == 0 or total_pos == N:
        return np.zeros(D, dtype=np.float32)

    best_f1 = np.zeros(D, dtype=np.float32)

    for start in range(0, D, chunk_size):
        end = min(start + chunk_size, D)
        S = features[:, start:end].astype(np.float32)  # (N, chunk)

        # Sort descending for each neuron
        order = np.argsort(-S, axis=0)                 # (N, chunk)
        y_sorted = y[order]                             # (N, chunk)

        TP = np.cumsum(y_sorted,     axis=0, dtype=np.int32)
        FP = np.cumsum(1 - y_sorted, axis=0, dtype=np.int32)
        FN = total_pos - TP

        with np.errstate(divide="ignore", invalid="ignore"):
            P = np.where(TP + FP > 0, TP / (TP + FP).astype(np.float32), 0.0)
            R = np.where(TP + FN > 0, TP / (TP + FN).astype(np.float32), 0.0)
            F = np.where(P + R > 0,   2 * P * R / (P + R),                0.0)

        best_f1[start:end] = F.max(axis=0).astype(np.float32)

    return best_f1


def find_ground_truth_neurons(
    features: np.ndarray,
    labels_df: pd.DataFrame,
    w1_attr: str,
    w2_attr: str,
    top_k: int = 1,
) -> Tuple[List[int], List[int]]:
    """
    Identify the top-k SAE neurons most predictive of W1 and W2 by F1 score.

    The binary prediction for neuron j is (Zj > 0).  F1 is computed over
    ALL CelebA images for a stable estimate.

    Args:
        features:  (N_celeba, hidden_dim) SAE feature matrix.
        labels_df: CelebA attribute table aligned with features rows.
        w1_attr:   Attribute name for W1.
        w2_attr:   Attribute name for W2.
        top_k:     Number of neurons per attribute to return (default: 1).

    Returns:
        (w1_neurons, w2_neurons) — index lists into the hidden_dim dimension.
    """
    W1 = labels_df[w1_attr].values.astype(float)
    W2 = labels_df[w2_attr].values.astype(float)

    f1_w1 = compute_f1_scores(features, W1)
    f1_w2 = compute_f1_scores(features, W2)

    w1_neurons: List[int] = np.argsort(f1_w1)[-top_k:].tolist()
    w2_neurons: List[int] = np.argsort(f1_w2)[-top_k:].tolist()

    return w1_neurons, w2_neurons


# ---------------------------------------------------------------------------
# Single-run evaluation
# ---------------------------------------------------------------------------

def run_one(
    features: np.ndarray,
    labels_df: pd.DataFrame,
    buckets: Dict,
    truth: List[int],
    n: int,
    effect_scale: float,
    seed: int,
    alpha: float = 0.05,
    max_rounds: int = 5,
    methods: Optional[List[str]] = None,
    gcm_splits: int = 3,
    **scm_kwargs: Any,
) -> Dict[str, Dict[str, float]]:
    """Draw one RCT sample and evaluate all (or a subset of) selection methods."""
    data = generate_celeba_rct(
        n=n,
        features=features,
        labels_df=labels_df,
        buckets=buckets,
        effect_scale=effect_scale,
        seed=seed,
        **scm_kwargs,
    )
    return evaluate_methods_on_dataset(
        y=data.Y,
        t=data.T,
        z=data.Z,
        truth=truth,
        alpha=alpha,
        max_rounds=max_rounds,
        methods=methods,
        gcm_splits=gcm_splits,
    )


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def run_sweep(
    features: np.ndarray,
    labels_df: pd.DataFrame,
    buckets: Dict,
    truth: List[int],
    sweep_param: str,
    param_grid: List[float],
    fixed_n: Optional[int] = None,
    fixed_effect: Optional[float] = None,
    n_seeds: int = 10,
    alpha: float = 0.05,
    max_rounds: int = 5,
    methods: Optional[List[str]] = None,
    gcm_splits: int = 3,
    n_jobs: int = -1,
    verbose: bool = True,
    **scm_kwargs: Any,
) -> pd.DataFrame:
    """Run a 1-D parameter sweep (effect_scale or n) and collect metrics.

    Args:
        sweep_param:   "effect_scale" or "n".
        param_grid:    Values to sweep over.
        fixed_n:       n used when sweep_param == "effect_scale".
        fixed_effect:  effect_scale used when sweep_param == "n".
        n_seeds:       Monte Carlo replications per grid point.
        methods:       Subset of ALL_METHODS to run (default: all).
        gcm_splits:    Cross-fit folds for GCM methods.

    Returns:
        Long-format DataFrame with columns:
          [sweep_param, seed, method, iou, recall, precision, tp, fp, n_selected, time_s]
    """
    def _one_task(param_val, seed):
        n_loc        = int(param_val) if sweep_param == "n" else fixed_n
        effect_loc   = param_val if sweep_param == "effect_scale" else fixed_effect
        try:
            return param_val, seed, run_one(
                features, labels_df, buckets, truth,
                n=n_loc, effect_scale=effect_loc, seed=seed,
                alpha=alpha, max_rounds=max_rounds,
                methods=methods, gcm_splits=gcm_splits,
                **scm_kwargs,
            )
        except ValueError:
            return param_val, seed, None  # bucket exhausted

    tasks = [(pv, s) for pv in param_grid for s in range(n_seeds)]
    if verbose:
        print(f"  {len(tasks)} tasks  "
              f"({len(param_grid)} {sweep_param} values × {n_seeds} seeds) …", flush=True)

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_one_task)(pv, s) for pv, s in tasks
    )

    rows = []
    for param_val, seed, metrics in results:
        if metrics is None:
            if verbose:
                print(f"    SKIP seed={seed}: bucket exhausted")
            continue
        for method, m in metrics.items():
            rows.append({
                sweep_param: param_val,
                "seed": seed,
                "method": method,
                **m,
            })

    df = pd.DataFrame(rows)
    if sweep_param == "effect_scale" and fixed_n is not None:
        df["fixed_n"] = fixed_n
    elif sweep_param == "n" and fixed_effect is not None:
        df["fixed_effect"] = fixed_effect
    return df
