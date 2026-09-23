#!/usr/bin/env python3
"""Does the INTERLEAVED backward step of NEXIS help on top of the terminal backward step?

Two arms are run on the same simulated CelebA RCT (one nexis call each):

  fb    forward + backward            nexis(backward=False, terminal_filter=True)
  fib   forward + interleaved + bwd   nexis(backward=True,  terminal_filter=True)

and two more are read off their metadata at no extra cost:

  f     forward only                  = S~ of fb   (metadata["terminal_candidates"])
  fi    forward + interleaved         = S~ of fib

Terminology (paper): "backward step" = the terminal subset-robust filter (keep j iff
p_j(A) <= alpha/m for every A subset of S~ minus j); "interleaved backward step" =
nexis(backward=True): after each forward admission drop j if p_j(S minus j) > alpha/|S|.

DGP (src/apps/celeba/scm.py, extended):
  T ~ Bern(1/2);  W in {0,1}^r CelebA attributes;  X = CelebA image with those
  attributes;  Z = dictionary code of X;
  Y = sum_k beta_k W_k + T (tau_0 + eta sum_k gamma_k W_k) + N(0, 1)
  sampler "indep"   : scm.generate_celeba_rct (attributes independent at their
                      marginal prevalence, image drawn from the joint cell)
  sampler "natural" : n images drawn uniformly without replacement from CelebA, W read
                      off their labels (natural attribute correlations; no joint-cell
                      constraint, n <= 19,867)
  S* = the best-threshold-F1 coordinate of each modifier (the paper's rule).

Constructed "mixing" coordinate (--mix, optional, clearly a constructed setting):
  z_mix = sum_k sign(gamma_k) * std(Z^{p_k}) * w_k  +  noise * N(0, 1)
  appended as an extra column, with p_k the principal coordinates of the modifiers
  listed in --mix-of and std() standardising with the full-CelebA column moments.  It
  is a noisy superposition of principal coordinates, so it is conditionally
  redundant once they are all in S, but it can have the largest marginal interaction
  and enter first.  It is never in S*.

Usage
  interleaved_backward.py --tag T --dict sae_k20 --attrs Wearing_Hat Eyeglasses Sideburns
      [--gammas 1 -1 1] [--sampler indep|natural] [--mix-of 0 1 --mix-w 1 1 --mix-noise 0.5]
      [--rho 0.5] --sweep effect|n --fixed X --grid a,b,c [--seeds 20]
Writes results/celeba/interleaved_backward/<tag>/<sweep>_<fixed>.csv (one row per run).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from apps.celeba.scm import build_buckets, generate_celeba_rct  # noqa: E402
from apps.celeba.experiment import compute_f1_scores  # noqa: E402
from method.nexis import nexis  # noqa: E402

OUT_ROOT = ROOT / "results/celeba/interleaved_backward"
ALIGN = OUT_ROOT / "align"
ALPHA, MAX_STEPS, MAX_CERT = 0.05, 10, 14

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--dict", default="sae_k20")
ap.add_argument("--attrs", nargs="+", required=True)
ap.add_argument("--gammas", nargs="+", type=float, default=None)
ap.add_argument("--sampler", choices=["indep", "natural"], default="indep")
ap.add_argument("--mix-of", nargs="*", type=int, default=[],
                help="indices (into --attrs) of the modifiers whose principal "
                     "coordinates are superposed into the constructed mixing column")
ap.add_argument("--mix-w", nargs="*", type=float, default=None)
ap.add_argument("--mix-noise", type=float, default=0.0)
ap.add_argument("--rho", type=float, default=0.5)
ap.add_argument("--sweep", choices=["effect", "n"], required=True)
ap.add_argument("--fixed", type=float, required=True)
ap.add_argument("--grid", required=True, help="comma-separated sweep values")
ap.add_argument("--seeds", type=int, default=20)
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()

out_dir = OUT_ROOT / args.tag
out_path = out_dir / f"{args.sweep}_{args.fixed:g}.csv"
if out_path.exists() and not args.overwrite:
    print(f"exists, skipping: {out_path}")
    sys.exit(0)
out_dir.mkdir(parents=True, exist_ok=True)

# ── data ─────────────────────────────────────────────────────────────────────
features = np.load(ROOT / "data/celeba/embeddings" / f"{args.dict}.npy")
labels_df = pd.read_parquet(ROOT / "data/celeba/labels.parquet")
attrs = list(args.attrs)
r = len(attrs)
gammas = (args.gammas if args.gammas is not None
          else [(1.0 if k % 2 == 0 else -1.0) for k in range(r)])
betas = [(0.3 if k % 2 == 0 else -0.2) for k in range(r)]
assert len(gammas) == r

# ground truth: best-threshold-F1 coordinate per modifier
f1_path = ALIGN / f"f1_{args.dict}.npz"
if f1_path.exists():
    npz = np.load(f1_path, allow_pickle=True)
    a_all = list(npz["attrs"])
    f1s = [npz["f1"][a_all.index(a)] for a in attrs]
else:
    f1s = [compute_f1_scores(features, labels_df[a].values.astype(float)) for a in attrs]
principal = [int(np.argmax(f)) for f in f1s]
gaps = [float(np.sort(f)[-1] - np.sort(f)[-2]) for f in f1s]
TRUTH = sorted(set(principal))
M0 = features.shape[1]

mix = len(args.mix_of) > 0
if mix:
    mw = args.mix_w if args.mix_w else [1.0] * len(args.mix_of)
    mix_cols = [principal[k] for k in args.mix_of]
    mix_mu = features[:, mix_cols].mean(axis=0)
    mix_sd = features[:, mix_cols].std(axis=0)
    mix_coef = np.array([np.sign(gammas[k]) * w for k, w in zip(args.mix_of, mw)]) / mix_sd
M = M0 + int(mix)

meta = {"tag": args.tag, "dict": args.dict, "attrs": attrs, "gammas": gammas,
        "betas": betas, "sampler": args.sampler, "principal": principal,
        "principal_f1": [float(f.max()) for f in f1s], "f1_gap": gaps,
        "truth": TRUTH, "mix_of": args.mix_of,
        "mix_w": (args.mix_w if mix else None), "mix_noise": args.mix_noise,
        "mix_col": (M0 if mix else None), "rho": args.rho, "alpha": ALPHA,
        "max_steps": MAX_STEPS, "m": M}
(out_dir / "setting.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta), flush=True)

buckets = build_buckets(labels_df, w_attrs=attrs) if args.sampler == "indep" else None
L = labels_df[attrs].values.astype(np.float64)


def simulate(n, effect, seed):
    if args.sampler == "indep":
        d = generate_celeba_rct(n=n, features=features, labels_df=labels_df,
                                buckets=buckets, w_attrs=attrs, betas=betas,
                                gammas=gammas, effect_scale=effect, tau_0=0.5,
                                noise_sd=1.0, seed=seed)
        y, t, Z, idx = d.Y, d.T, d.Z, d.image_indices
        rng = np.random.default_rng([seed, 7])
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(labels_df), size=n, replace=False)
        W = L[idx]
        t = rng.binomial(1, 0.5, size=n).astype(np.float64)
        tau = 0.5 + effect * (W * np.asarray(gammas)).sum(axis=1)
        y = (W * np.asarray(betas)).sum(axis=1) + tau * t + rng.normal(0, 1, size=n)
        Z = features[idx].astype(np.float64)
    if mix:
        zm = ((features[idx][:, mix_cols] - mix_mu) * mix_coef).sum(axis=1)
        zm = zm + args.mix_noise * rng.normal(0, 1, size=n)
        Z = np.column_stack([Z, zm])
    return y, t, Z


def score(sel):
    ss, ts = set(int(x) for x in sel), set(TRUTH)
    tp, k = len(ss & ts), len(ss)
    den = k + len(ts) - tp
    return {"n_sel": k, "tp": tp, "fp": k - tp, "recall": tp / len(ts),
            "precision": (tp / k) if k else 0.0, "iou": (tp / den) if den else 0.0,
            "has_mix": int(mix and M0 in ss)}


def run_arm(y, t, Z, backward):
    t0 = time.perf_counter()
    try:
        res = nexis(y=y, t=t, z=Z, alpha=ALPHA, max_rounds=MAX_STEPS, rho=args.rho,
                    backward=backward, terminal_filter=True,
                    terminal_max_size=MAX_CERT)
        md = res.metadata
        return dict(S_pre=[int(j) for j in md["terminal_candidates"]],
                    S_fin=[int(j) for j in res.selected],
                    term_tests=int(md["terminal_tests"]),
                    int_tests=int(md["backward_tests"]),
                    int_removed=int(md["backward_removed"]),
                    removed_ids=[int(b["j"]) for b in md["backward_log"]
                                 if not b["retained"]],
                    rounds=int(md["rounds"]), sec=time.perf_counter() - t0,
                    uncert=False)
    except ValueError as e:
        if "terminal_max_size" not in str(e):
            raise
        res = nexis(y=y, t=t, z=Z, alpha=ALPHA, max_rounds=MAX_STEPS, rho=args.rho,
                    backward=backward)
        md = res.metadata
        S = [int(j) for j in res.selected]
        return dict(S_pre=S, S_fin=S, term_tests=0, int_tests=int(md["backward_tests"]),
                    int_removed=int(md["backward_removed"]),
                    removed_ids=[int(b["j"]) for b in md["backward_log"]
                                 if not b["retained"]],
                    rounds=int(md["rounds"]), sec=time.perf_counter() - t0, uncert=True)


def one(param, seed):
    n = int(param) if args.sweep == "n" else int(args.fixed)
    effect = float(param) if args.sweep == "effect" else float(args.fixed)
    try:
        y, t, Z = simulate(n, effect, seed)
    except ValueError as e:
        print(f"skip n={n} eta={effect} seed={seed}: {e}", flush=True)
        return None
    row = {"tag": args.tag, "sweep": args.sweep, "fixed": args.fixed, "param": param,
           "seed": seed, "n": n, "effect_scale": effect}
    A = run_arm(y, t, Z, backward=False)
    B = run_arm(y, t, Z, backward=True)
    for name, S in (("f", A["S_pre"]), ("fb", A["S_fin"]),
                    ("fi", B["S_pre"]), ("fib", B["S_fin"])):
        row.update({f"{name}_{k}": v for k, v in score(S).items()})
        row[f"{name}_sel"] = " ".join(map(str, S))
    row.update({"fb_term_tests": A["term_tests"], "fib_term_tests": B["term_tests"],
                "fib_int_tests": B["int_tests"], "fib_int_removed": B["int_removed"],
                "fib_removed_ids": " ".join(map(str, B["removed_ids"])),
                "fib_removed_truth": sum(j in TRUTH for j in B["removed_ids"]),
                "fb_rounds": A["rounds"], "fib_rounds": B["rounds"],
                "fb_sec": A["sec"], "fib_sec": B["sec"],
                "fb_uncert": A["uncert"], "fib_uncert": B["uncert"],
                "first_in": A["S_pre"][0] if A["S_pre"] else -1,
                "differ": int(set(A["S_fin"]) != set(B["S_fin"]))})
    return row


grid = [float(g) for g in args.grid.split(",")]
tasks = [(p, s) for p in grid for s in range(args.seeds)]
print(f"{len(tasks)} runs, truth={TRUTH} m={M}", flush=True)
nj = int(os.environ.get("NJOBS", "8"))
rows = Parallel(n_jobs=nj, prefer="threads", verbose=5)(delayed(one)(*tk) for tk in tasks)
df = pd.DataFrame([r_ for r_ in rows if r_ is not None])
tmp = out_path.with_suffix(".csv.tmp")
df.to_csv(tmp, index=False)
tmp.replace(out_path)
print(f"wrote {len(df)} rows -> {out_path}")
