#!/usr/bin/env python3
"""Opt2 on a support-screened dictionary.

The false positives Opt2 was implicitly suppressing are a TEST-VALIDITY failure, not a
multiplicity failure: a sparse code with 2 in-sample nonzeros (often 1 per arm) makes the
classical interaction t-test meaningless, so its p=1e-9 is an artifact.  Screen those
columns out FIRST -- using only Z and T, never Y, so it is not data snooping -- and then
run NEXIS + Opt2 over ALL subsets including the empty set.

Two compounding benefits:
  1. nothing is left for Opt2 to compensate for, so its power cost buys nothing it needs;
  2. m drops from ~9216 to ~200, so the IUT threshold alpha/m LOOSENS by ~45x, which is
     exactly what the low-power marginal component needed.

Screen: a column is kept iff it fires on >= MIN_ARM units in EACH treatment arm.

Variants: base / opt2 on the full dictionary, vs base / opt2 after screening.
"""
import json
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

ALPHA, MAX_STEPS, N_SEEDS = 0.05, 10, 50
MIN_ARM = 5
CELLS = [("k5/sae", 200, 5.0), ("k5/sae", 350, 5.0), ("k5/sae", 500, 5.0),
         ("k20/sae", 200, 5.0), ("k20/sae", 500, 3.0), ("k20/sae", 500, 4.0),
         ("k20/sae", 500, 5.0), ("k20/sae", 750, 2.0),
         ("k20/sae", 500, 10.0), ("k5/sae", 500, 10.0)]

_c = {}


def load(tree):
    if tree not in _c:
        kdir, ftype = tree.split("/")
        k = int(kdir[1:])
        fn = f"sae_k{k}.npy" if ftype == "sae" else f"sae_precode_k{k}.npy"
        gt = json.load(open(ROOT / f"results/celeba/experiment/{tree}/ground_truth.json"))
        _c[tree] = (np.load(ROOT / "data/celeba/embeddings" / fn), sorted(set(gt["truth"])))
    return _c[tree]


labels_df = pd.read_parquet(ROOT / "data/celeba/labels.parquet")
buckets = build_buckets(labels_df, "Wearing_Hat", "Eyeglasses")
scm = dict(w1_attr="Wearing_Hat", w2_attr="Eyeglasses",
           tau_0=0.5, gamma_w1=1.0, gamma_w2=-1.0, noise_sd=1.0)


def opt2_certify(y, t, Z, S, m_eff):
    """max over ALL A subseteq S\\{j} (empty set included) <= alpha/m_eff."""
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


def score(sel, truth):
    ss, ts = set(int(x) for x in sel), set(truth)
    tp, k = len(ss & ts), len(ss)
    den = k + len(ts) - tp
    return dict(tp=tp, k=k, recall=tp / len(ts),
                precision=(tp / k) if k else 0.0, iou=(tp / den) if den else 0.0)


def one(tree, n, effect, seed):
    feats, TRUTH = load(tree)
    m = feats.shape[1]
    try:
        d = generate_celeba_rct(n=n, features=feats, labels_df=labels_df,
                                buckets=buckets, effect_scale=effect, seed=seed, **scm)
    except ValueError:
        return None
    y, t, Z = d.Y, d.T, d.Z

    nz = (Z != 0)
    keep_cols = ((nz[t == 1].sum(axis=0) >= MIN_ARM) & (nz[t == 0].sum(axis=0) >= MIN_ARM))
    cols = np.where(keep_cols)[0]
    m_scr = len(cols)
    truth_kept = len(set(TRUTH) & set(cols.tolist()))

    S_full = list(nexis(y=y, t=t, z=Z, alpha=ALPHA, max_rounds=MAX_STEPS).selected)
    o2_full = opt2_certify(y, t, Z, S_full, m)

    row = {"tree": tree, "n": n, "effect": effect, "seed": seed, "m": m,
           "m_screened": m_scr, "truth_kept": truth_kept}
    if m_scr > 0:
        Zs = Z[:, cols]
        Ss = list(nexis(y=y, t=t, z=Zs, alpha=ALPHA, max_rounds=MAX_STEPS).selected)
        o2s = opt2_certify(y, t, Zs, Ss, m_scr)
        S_scr = [int(cols[j]) for j in Ss]
        o2_scr = [int(cols[j]) for j in o2s]
    else:
        S_scr, o2_scr = [], []

    for name, sel in [("base", S_full), ("opt2", o2_full),
                      ("screen", S_scr), ("screen_opt2", o2_scr)]:
        for k, v in score(sel, TRUTH).items():
            row[f"{name}_{k}"] = v
    return row


tasks = [(tr, n, e, s) for (tr, n, e) in CELLS for s in range(N_SEEDS)]
print(f"{len(tasks)} runs (MIN_ARM={MIN_ARM}) …", flush=True)
res = Parallel(n_jobs=32, prefer="threads", verbose=5)(delayed(one)(*t) for t in tasks)
df = pd.DataFrame([r for r in res if r])
df.to_csv(HERE / "opt2_screen.csv", index=False)
print(f"wrote {len(df)} rows")
