#!/usr/bin/env python3
"""Full CelebA paper grid with 4 variants.

  base        NEXIS as published
  opt2        NEXIS + subset-robust certification (all subsets incl. empty, alpha/m)
  scr         NEXIS on a support-screened dictionary
  scr_opt2    screened dictionary + subset-robust certification (alpha/m_screened)

Screen: effective support per treatment arm, ESS(z) = (sum|z|)^2 / sum z^2, must be
>= MIN_ARM in BOTH arms.  For sparse top-k codes ESS = number of nonzeros; for dense
pre-activations it generalises to "how many observations actually carry the column",
so the same rule applies to sae and sae_precode.  Uses only Z and T, never Y.

Usage: run_all4.py <tree> <sweep> <fixed>
"""
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path("/nfs/scistore19/locatgrp/rcadei/NEXIS")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from apps.celeba.scm import build_buckets, generate_celeba_rct
from method.nexis import nexis, conditional_interaction_pvalues

TREE, SWEEP, FIXED = sys.argv[1], sys.argv[2], float(sys.argv[3])
ALPHA, MAX_STEPS, N_SEEDS, MIN_ARM = 0.05, 10, 50, 3
EFFECT_GRID = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
N_GRID = [50, 100, 200, 350, 500, 750, 1000, 2000, 3500, 5000, 10000]

kdir, ftype = TREE.split("/")
k = int(kdir[1:])
fname = f"sae_k{k}.npy" if ftype == "sae" else f"sae_precode_k{k}.npy"
features = np.load(ROOT / "data/celeba/embeddings" / fname)
labels_df = pd.read_parquet(ROOT / "data/celeba/labels.parquet")
gt = json.load(open(ROOT / f"results/celeba/experiment/{TREE}/ground_truth.json"))
TRUTH = sorted(set(gt["truth"]))
buckets = build_buckets(labels_df, "Wearing_Hat", "Eyeglasses")
scm = dict(w1_attr="Wearing_Hat", w2_attr="Eyeglasses",
           tau_0=0.5, gamma_w1=1.0, gamma_w2=-1.0, noise_sd=1.0)
M = features.shape[1]
print(f"{TREE} {fname} m={M} truth={TRUTH} sweep={SWEEP} fixed={FIXED:g}", flush=True)


def eff_support(Zarm):
    """Kish effective support per column: (sum|z|)^2 / sum z^2."""
    s1 = np.abs(Zarm).sum(axis=0)
    s2 = (Zarm ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ess = np.where(s2 > 0, s1 ** 2 / s2, 0.0)
    return ess


def certify(y, t, Z, S, m_eff):
    thr = ALPHA / m_eff
    keep = []
    for j in S:
        others = [s for s in S if s != j]
        worst = 0.0
        for r in range(len(others) + 1):
            for A in combinations(others, r):
                worst = max(worst, float(conditional_interaction_pvalues(
                    y=y, t=t, z=Z, S=list(A), candidates=[j])[j]))
        if worst <= thr:
            keep.append(j)
    return keep


def score(sel):
    ss, ts = set(int(x) for x in sel), set(TRUTH)
    tp, kk = len(ss & ts), len(ss)
    den = kk + len(ts) - tp
    return {"n_selected": kk, "tp": tp, "fp": len(ss - ts),
            "recall": tp / len(ts), "precision": (tp / kk) if kk else 0.0,
            "iou": (tp / den) if den else 0.0}


def one(param, seed):
    n = int(param) if SWEEP == "n" else int(FIXED)
    effect = float(param) if SWEEP == "effect" else FIXED
    try:
        d = generate_celeba_rct(n=n, features=features, labels_df=labels_df,
                                buckets=buckets, effect_scale=effect, seed=seed, **scm)
    except ValueError:
        return []
    y, t, Z = d.Y, d.T, d.Z
    S = list(nexis(y=y, t=t, z=Z, alpha=ALPHA, max_rounds=MAX_STEPS).selected)
    o2 = certify(y, t, Z, S, M)

    cols = np.where((eff_support(Z[t == 1]) >= MIN_ARM)
                    & (eff_support(Z[t == 0]) >= MIN_ARM))[0]
    if len(cols):
        Zs = Z[:, cols]
        Ss = list(nexis(y=y, t=t, z=Zs, alpha=ALPHA, max_rounds=MAX_STEPS).selected)
        s_sel = [int(cols[j]) for j in Ss]
        s_o2 = [int(cols[j]) for j in certify(y, t, Zs, Ss, len(cols))]
    else:
        s_sel, s_o2 = [], []

    tag = {"tree": TREE, "sweep": SWEEP, "fixed": FIXED, "param": param, "seed": seed,
           "n": n, "effect_scale": effect, "m": M, "m_scr": len(cols),
           "truth_kept": len(set(TRUTH) & set(cols.tolist()))}
    return [{**tag, "variant": v, **score(s)} for v, s in
            [("base", S), ("opt2", o2), ("scr", s_sel), ("scr_opt2", s_o2)]]


grid = EFFECT_GRID if SWEEP == "effect" else N_GRID
tasks = [(p, s) for p in grid for s in range(N_SEEDS)]
print(f"{len(tasks)} runs …", flush=True)
nj = int(os.environ.get("NJOBS", "32"))
out = Parallel(n_jobs=nj, prefer="threads", verbose=5)(delayed(one)(*t) for t in tasks)
df = pd.DataFrame([r for rows in out for r in rows])
tag = f"{TREE.replace('/', '_')}_{SWEEP}_{FIXED:g}"
df.to_csv(HERE / f"all4_{tag}.csv", index=False)
print(f"wrote {len(df)} rows -> all4_{tag}.csv")
