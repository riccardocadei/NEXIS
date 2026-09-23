#!/usr/bin/env python3
"""Randomization-calibrated gating for NEXIS.

Idea: effect modification is a T x Z interaction and T is RANDOMIZED, so the null
distribution of the whole search can be simulated by permuting T.  Instead of paying
a Bonferroni/combinatorial price for adaptivity, calibrate the gate so that the
probability of ANY false selection under the permuted-T null is <= alpha:

    t_cal(seed) = alpha-quantile of  min_j p_j(T_perm)   over B permutations

Then run NEXIS with that single gate (adjust=None, alpha=t_cal), which makes both the
forward and backward gates equal to t_cal.

Why this should fix BOTH failure modes at once:
  - leverage FPs (2-nonzero sparse codes) produce absurdly small p-values under the
    permuted null too, so t_cal automatically becomes strict enough to exclude them;
  - genuine modifiers sit at 1e-20..1e-60 and are unaffected;
  - no faithfulness assumption, no subset enumeration, no combinatorial multiplicity.

Limitation (stated, not hidden): this calibrates the FIRST-step gate and reuses it for
later steps; a fully rigorous version recalibrates at each step conditional on the
current S.  Also, permuting T leaves the original interaction variance in Y, making
the calibration mildly conservative.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path("/nfs/scistore19/locatgrp/rcadei/NEXIS")
sys.path.insert(0, str(ROOT / "src"))
from apps.celeba.scm import build_buckets, generate_celeba_rct
from method.nexis import nexis, conditional_interaction_pvalues

ALPHA, MAX_STEPS, N_SEEDS, B = 0.05, 10, 50, 200
CELLS = [("k5/sae", 200, 5.0), ("k5/sae", 350, 5.0), ("k5/sae", 500, 5.0),
         ("k20/sae", 200, 5.0), ("k20/sae", 500, 3.0), ("k20/sae", 500, 4.0),
         ("k20/sae", 500, 5.0), ("k20/sae", 750, 2.0),
         ("k20/sae", 500, 10.0), ("k5/sae", 500, 10.0)]

_c = {}
def load(tree):
    if tree not in _c:
        kdir, ftype = tree.split("/"); k = int(kdir[1:])
        fn = f"sae_k{k}.npy" if ftype == "sae" else f"sae_precode_k{k}.npy"
        gt = json.load(open(ROOT / f"results/celeba/experiment/{tree}/ground_truth.json"))
        _c[tree] = (np.load(ROOT / "data/celeba/embeddings" / fn), sorted(set(gt["truth"])))
    return _c[tree]

labels_df = pd.read_parquet(ROOT / "data/celeba/labels.parquet")
buckets = build_buckets(labels_df, "Wearing_Hat", "Eyeglasses")
scm = dict(w1_attr="Wearing_Hat", w2_attr="Eyeglasses",
           tau_0=0.5, gamma_w1=1.0, gamma_w2=-1.0, noise_sd=1.0)


def metrics(sel, truth):
    ss, ts = set(int(x) for x in sel), set(truth)
    tp, k = len(ss & ts), len(ss)
    return dict(tp=tp, k=k, recall=tp / len(ts),
                precision=(tp / k) if k else 0.0,
                iou=tp / (k + len(ts) - tp) if (k + len(ts) - tp) else 0.0)


def one(tree, n, effect, seed):
    feats, TRUTH = load(tree)
    m = feats.shape[1]
    try:
        d = generate_celeba_rct(n=n, features=feats, labels_df=labels_df, buckets=buckets,
                                effect_scale=effect, seed=seed, **scm)
    except ValueError:
        return None
    y, t, Z = d.Y, d.T, d.Z
    rng = np.random.default_rng(10_000 + seed)
    mins = np.empty(B)
    for b in range(B):
        tp_ = rng.permutation(t)
        pv = conditional_interaction_pvalues(y=y, t=tp_, z=Z, S=[])
        mins[b] = pv.min()
    t_cal = float(np.quantile(mins, ALPHA))

    base = list(nexis(y=y, t=t, z=Z, alpha=ALPHA, max_rounds=MAX_STEPS).selected)
    cal = list(nexis(y=y, t=t, z=Z, alpha=t_cal, adjust=None, max_rounds=MAX_STEPS).selected)

    # Opt2 and Opt2-nonempty on the ORIGINAL candidate set, for side-by-side
    def robust(S, nonempty):
        keep = []
        for j in S:
            others = [s for s in S if s != j]
            best = 0.0
            for r in range(1 if nonempty else 0, len(others) + 1):
                for A in combinations(others, r):
                    best = max(best, float(conditional_interaction_pvalues(
                        y=y, t=t, z=Z, S=list(A), candidates=[j])[j]))
            if best <= ALPHA / m:
                keep.append(j)
        return keep

    out = {"tree": tree, "n": n, "effect": effect, "seed": seed,
           "t_cal": t_cal, "alpha_over_m": ALPHA / m,
           "null_min_p_median": float(np.median(mins))}
    for name, sel in [("base", base), ("opt2", robust(base, False)),
                      ("opt2_ne", robust(base, True)), ("randcal", cal)]:
        for kk, vv in metrics(sel, TRUTH).items():
            out[f"{name}_{kk}"] = vv
    return out


tasks = [(tr, n, e, s) for (tr, n, e) in CELLS for s in range(N_SEEDS)]
print(f"{len(tasks)} runs (B={B} permutations each) …", flush=True)
res = Parallel(n_jobs=32, prefer="threads", verbose=5)(delayed(one)(*t) for t in tasks)
df = pd.DataFrame([r for r in res if r])
df.to_csv("rand_gate.csv", index=False)
print(f"wrote {len(df)} rows")
