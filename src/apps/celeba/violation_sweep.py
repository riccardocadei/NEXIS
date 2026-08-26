"""
Controlled violation of Principal Alignment (Assumption 3) on CelebA.

Rebuttal analysis.  Assumption 3 asks that each latent direct modifier admit a
single coordinate that screens it off from the rest of the dictionary.  It does
*not* ask for monosemanticity, and it is not violated by a coordinate encoding
several concepts, nor by many coordinates being correlated with the modifier:
those leave the principal a sufficient statistic.  The one failure mode that
does violate it is *concept splitting* (a.k.a. feature absorption): the
modifier's signal is spread over several coordinates, none of which screens off
the others.

This script builds that failure mode explicitly, with a dial, and measures what
survives.  For a split fraction eps, a fixed per-image indicator
U_i ~ Bernoulli(eps) redistributes the principal coordinate of W1:

    Z'[:, j1   ] = Z[:, j1] * (1 - U)        # principal keeps a (1-eps) share
    Z'[:, j_new] = Z[:, j1] * U              # a fresh coordinate takes the rest

At eps = 0 Assumption 3 holds and the target is {j1, j2}.  For eps > 0 no single
coordinate encodes W1: the minimal sufficient set becomes {j1, j_new, j2} and
the one-to-one map of Assumption 3 does not exist.  At eps = 0.5 the concept is
split in half and the notion of a "principal" coordinate for W1 is void.

The split is a property of the *dictionary*: U is drawn once and held fixed
across experiment seeds, sample sizes, and effect sizes.

Reported per (eps, method)
--------------------------
  * precision / recall / IoU against S*_orig  = {j1, j2}
      -> measures loss of the one-to-one characterisation (minimality)
  * precision / recall / IoU against S*_split = {j1, j_new, j2}
      -> measures whether the procedure finds the coordinates that jointly
         encode the modifier
  * suff_r2 : population R^2 of the linear projection of the true tau on the
    selected coordinates, against the reference achievable on S*_split
      -> measures loss of *sufficiency*, the property the causal claim rests on
  * eps_hat : measured leakage ratio of the perturbed dictionary (the scale-free
    quantity of the rho-gate), so recovery can be read against a quantity the
    analyst can estimate rather than against the nominal eps

Usage
-----
python src/apps/celeba/violation_sweep.py                       # default sweep
python src/apps/celeba/violation_sweep.py --n-seeds 50 --fixed-n 2000 10000
python src/apps/celeba/violation_sweep.py --split-both
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent.parent.parent

from method.nexis import nexis, marginal_select, iou_score
from apps.celeba.scm import build_buckets, generate_celeba_rct
from apps.celeba.experiment import compute_f1_scores
from apps.celeba.principal_alignment import leakage_spectrum
from apps.celeba.backbones import (
    BACKBONES, DEFAULT_BACKBONE, get_backbone,
    code_path as _code_path, precode_path as _precode_path,
)


METHODS: List[str] = [
    "NEXIS",                    # default: rho = 0.5
    "NEXIS (rho=0)",            # vanilla Algorithm 1, no spectral gap
    "Marginal Testing (FWER)",
]


# ── dictionary perturbation ───────────────────────────────────────────────────

def split_principal(
    Z: np.ndarray,
    principal: int,
    eps: float,
    seed: int = 0,
    U: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Split ``principal`` into itself plus one appended coordinate.

    A fraction ``eps`` of the images carrying the concept have their activation
    moved to the new coordinate.  Returns (Z_perturbed, new_index, U).
    """
    n = Z.shape[0]
    if U is None:
        U = (np.random.default_rng(seed).random(n) < eps).astype(np.float64)
    col = Z[:, principal].astype(np.float32)
    Zp = np.concatenate([Z, (col * U).astype(np.float32)[:, None]], axis=1)
    Zp[:, principal] = col * (1.0 - U)
    return Zp, Zp.shape[1] - 1, U


# ── sufficiency of a selected set ─────────────────────────────────────────────

def population_r2(Z: np.ndarray, tau: np.ndarray, S: Sequence[int]) -> float:
    """R^2 of the population linear projection of tau on Z^S (intercept included)."""
    S = [int(j) for j in S]
    if not S:
        return 0.0
    A = np.column_stack([np.ones(len(Z)), Z[:, S].astype(np.float64)])
    coef, *_ = np.linalg.lstsq(A, tau, rcond=None)
    resid = tau - A @ coef
    var = float(tau.var())
    return float(1.0 - resid.var() / var) if var > 0 else 0.0


# ── one run ───────────────────────────────────────────────────────────────────

def _metrics(sel: Sequence[int], truth: Sequence[int]) -> Dict[str, float]:
    s, t = set(int(x) for x in sel), set(int(x) for x in truth)
    tp = len(s & t)
    return {
        "recall": tp / len(t) if t else 1.0,
        "precision": (tp / len(s)) if s else (1.0 if not t else 0.0),
        "iou": iou_score(s, t),
    }


def run_one(
    Zp: np.ndarray,
    labels_df: pd.DataFrame,
    buckets: Dict,
    n: int,
    effect_scale: float,
    seed: int,
    alpha: float,
    max_rounds: int,
    scm_kwargs: Dict,
) -> List[Dict]:
    """Run every method on one draw and return the selections only.

    The sufficiency R^2 is deliberately *not* computed here: it needs the full
    dictionary, and at n = 10^4 the conditional test already holds several
    n x m float64 buffers per worker.  The parent computes it from the returned
    selections instead.
    """
    d = generate_celeba_rct(
        n=n, features=Zp, labels_df=labels_df, buckets=buckets,
        effect_scale=effect_scale, seed=seed, **scm_kwargs,
    )
    runners = {
        "NEXIS": lambda: nexis(y=d.Y, t=d.T, z=d.Z, alpha=alpha,
                               max_rounds=max_rounds, rho=0.5),
        "NEXIS (rho=0)": lambda: nexis(y=d.Y, t=d.T, z=d.Z, alpha=alpha,
                                       max_rounds=max_rounds, rho=0.0),
        "Marginal Testing (FWER)": lambda: marginal_select(
            y=d.Y, t=d.T, z=d.Z, alpha=alpha, adjust="FWER"),
    }
    rows = []
    for name, fn in runners.items():
        sel = sorted(int(x) for x in fn().selected)
        rows.append({"method": name, "n": n, "effect_scale": effect_scale,
                     "seed": seed, "n_selected": len(sel), "selected": sel})
    del d
    return rows


def score_rows(
    rows: List[Dict],
    Zp: np.ndarray,
    tau_pop: np.ndarray,
    truth_orig: Sequence[int],
    truth_split: Sequence[int],
    ref_r2: float,
) -> List[Dict]:
    """Attach recovery metrics and sufficiency R^2 to raw selections (parent side).

    R^2 is cached on the selected set, since many seeds return the same set.
    """
    cache: Dict[tuple, float] = {}
    for r in rows:
        sel = r["selected"]
        for tag, truth in [("orig", truth_orig), ("split", truth_split)]:
            for k, v in _metrics(sel, truth).items():
                r[f"{k}_{tag}"] = v
        key = tuple(sel)
        if key not in cache:
            cache[key] = population_r2(Zp, tau_pop, sel)
        r["suff_r2"] = cache[key]
        r["suff_ratio"] = cache[key] / ref_r2 if ref_r2 > 0 else np.nan
    return rows


# ── sweep ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default="data/celeba")
    p.add_argument("--out-dir", type=Path,
                   default="results/celeba/experiment/violation")
    p.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=sorted(BACKBONES))
    p.add_argument("--sae-top-k", type=int, default=20)
    p.add_argument("--precode", action="store_true")
    p.add_argument("--eps-grid", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    p.add_argument("--fixed-n", type=int, nargs="+", default=[2000, 10000])
    p.add_argument("--fixed-effect", type=float, nargs="+", default=[5.0])
    p.add_argument("--n-seeds", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--split-both", action="store_true",
                   help="Split the principal of both modifiers (default: W1 only, "
                        "so the W2 principal acts as an internal control).")
    p.add_argument("--w1-attr", default="Wearing_Hat")
    p.add_argument("--w2-attr", default="Eyeglasses")
    p.add_argument("--tau0", type=float, default=0.5)
    p.add_argument("--gamma-w1", type=float, default=1.0)
    p.add_argument("--gamma-w2", type=float, default=-1.0)
    p.add_argument("--noise-sd", type=float, default=1.0)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--split-seed", type=int, default=12345)
    p.add_argument("--tag", default="",
                   help="Suffix for the output files, so sample-size arms run as "
                        "separate jobs without clobbering each other. Merge with "
                        "pandas.concat over violation_sweep*.parquet.")
    args = p.parse_args()

    data_dir = ROOT / args.data_dir if not args.data_dir.is_absolute() else args.data_dir
    out_dir = ROOT / args.out_dir if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = get_backbone(args.backbone)
    reg_file = (_precode_path if args.precode else _code_path)(
        data_dir, spec.key, args.sae_top_k)
    pre_file = _precode_path(data_dir, spec.key, args.sae_top_k)

    Z = np.load(reg_file)
    Zpre = np.load(pre_file) if pre_file.exists() else Z
    labels_df = pd.read_parquet(data_dir / "labels.parquet")
    W1 = labels_df[args.w1_attr].values.astype(np.float64)
    W2 = labels_df[args.w2_attr].values.astype(np.float64)
    print(f"dictionary {Z.shape}  ({spec.label}, k={args.sae_top_k}, "
          f"{'z_pre' if args.precode else 'z'})")

    j1 = int(np.argmax(compute_f1_scores(Zpre, W1)))
    j2 = int(np.argmax(compute_f1_scores(Zpre, W2)))
    print(f"principals: {args.w1_attr} -> {j1}   {args.w2_attr} -> {j2}")

    tau_pop = args.gamma_w1 * W1 + args.gamma_w2 * W2
    buckets = build_buckets(labels_df, args.w1_attr, args.w2_attr)
    scm_kwargs = dict(w1_attr=args.w1_attr, w2_attr=args.w2_attr,
                      tau_0=args.tau0, gamma_w1=args.gamma_w1,
                      gamma_w2=args.gamma_w2, noise_sd=args.noise_sd)

    rows: List[Dict] = []
    diag: List[Dict] = []
    for eps in args.eps_grid:
        Zp, jn1, _ = split_principal(Z, j1, eps, seed=args.split_seed)
        split_new = [jn1]
        if args.split_both:
            Zp, jn2, _ = split_principal(Zp, j2, eps, seed=args.split_seed + 1)
            split_new.append(jn2)

        truth_orig = sorted([j1, j2])
        truth_split = sorted([j1, j2] + split_new)

        # measured leakage of the perturbed dictionary w.r.t. the original S*
        lk = leakage_spectrum(Zp, tau_pop, truth_orig)
        ref_r2 = population_r2(Zp, tau_pop, truth_split)
        r2_orig = population_r2(Zp, tau_pop, truth_orig)
        d = {"eps": eps, "eps_hat": lk["eps_hat"], "argmax_leak": lk["argmax_leak"],
             "leak_is_split_coord": bool(lk["argmax_leak"] in split_new),
             "r2_S_split": ref_r2, "r2_S_orig": r2_orig,
             "truth_orig": truth_orig, "truth_split": truth_split}
        diag.append(d)
        print(f"\neps={eps:.2f}  eps_hat={lk['eps_hat']:.3f}  "
              f"(max leak at {lk['argmax_leak']}, is split coord: "
              f"{d['leak_is_split_coord']})  "
              f"R2(S_split)={ref_r2:.4f}  R2(S_orig)={r2_orig:.4f}")

        for n in args.fixed_n:
            for es in args.fixed_effect:
                out = Parallel(n_jobs=args.n_jobs, verbose=0)(
                    delayed(run_one)(
                        Zp, labels_df, buckets, n, es, s,
                        args.alpha, args.max_steps, scm_kwargs,
                    ) for s in range(args.n_seeds)
                )
                flat = [r | {"eps": eps, "eps_hat": lk["eps_hat"]}
                        for batch in out for r in batch]
                flat = score_rows(flat, Zp, tau_pop, truth_orig, truth_split, ref_r2)
                rows.extend(flat)
                df = pd.DataFrame(flat)
                agg = df.groupby("method")[
                    ["recall_orig", "precision_orig", "iou_orig",
                     "recall_split", "precision_split", "iou_split",
                     "suff_ratio", "n_selected"]].mean()
                print(f"  n={n} eta={es}")
                print(agg.round(3).to_string().replace("\n", "\n    "))

    suffix = f"_{args.tag}" if args.tag else ""
    df = pd.DataFrame(rows)
    df["selected"] = df["selected"].apply(json.dumps)
    path = out_dir / f"violation_sweep{suffix}.parquet"
    df.to_parquet(path, index=False)
    diag_path = out_dir / f"violation_diagnostics{suffix}.json"
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)
    print(f"\nwrote {path}  ({len(df)} rows)")
    print(f"wrote {diag_path}")


if __name__ == "__main__":
    main()
