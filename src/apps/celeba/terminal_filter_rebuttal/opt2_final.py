#!/usr/bin/env python3
"""Synthesis: Opt2 over ALL subsets (empty set included), with the two failure modes
handled where they actually live.

  (i)  power loss  = sigma^2 inflation from omitting co-modifiers
                   -> pool the error variance across the family (full-model sigma^2)
  (ii) leverage FPs = the t-test is invalid on near-singleton sparse columns
                   -> screen columns with < MIN_ARM nonzeros in EITHER arm, using only
                      Z and T (never Y), BEFORE the search.  MIN_ARM=3 is the minimum
                      support for a two-arm interaction contrast to mean anything; the
                      earlier MIN_ARM=5 was too blunt and dropped true neurons at n=200.

Screening also shrinks m (~9216 -> ~200), which LOOSENS alpha/m by ~45x -- helping the
low-efficiency members of the subset family that caused the recall loss.

Variants:
  base            NEXIS as published
  opt2            Opt2, all subsets, standard sigma^2, alpha/m          (as specified)
  scr_opt2        screen -> NEXIS -> Opt2, standard sigma^2, alpha/m_scr
  scr_opt2_pool   screen -> NEXIS -> Opt2, pooled sigma^2,   alpha/m_scr
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

ROOT = Path("/nfs/scistore19/locatgrp/rcadei/NEXIS")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from apps.celeba.scm import build_buckets, generate_celeba_rct
from method.nexis import nexis, conditional_interaction_pvalues

ALPHA, MAX_STEPS, N_SEEDS, MIN_ARM = 0.05, 10, 50, 3
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


def _resid(D, M):
    if D.shape[1] == 0:
        return M
    coef, *_ = np.linalg.lstsq(D, M, rcond=None)
    return M - D @ coef


def _design(t, Z, A):
    cols = [np.ones(len(t)), t]
    cols += [Z[:, k] for k in A]
    cols += [t * Z[:, k] for k in A]
    return np.column_stack(cols)


def p_interaction(y, t, Z, j, A, sigma2=None, dof_override=None):
    n = len(y)
    D = _design(t, Z, A)
    y_t = _resid(D, y)
    Zj = Z[:, j]
    Zt = _resid(D, Zj.reshape(-1, 1))[:, 0]
    Xt = _resid(D, (t * Zj).reshape(-1, 1))[:, 0]
    zz, xx, zx = Zt @ Zt, Xt @ Xt, Zt @ Xt
    det = zz * xx - zx * zx
    dof = n - (D.shape[1] + 2)
    if det <= 1e-12 or dof <= 0:
        return 1.0
    b_x = (zz * (Xt @ y_t) - zx * (Zt @ y_t)) / det
    b_z = (xx * (Zt @ y_t) - zx * (Xt @ y_t)) / det
    e = y_t - Zt * b_z - Xt * b_x
    s2 = (max(e @ e, 0.0) / dof) if sigma2 is None else sigma2
    var = s2 * (zz / det)
    if not np.isfinite(var) or var <= 0:
        return 1.0
    d = dof if dof_override is None else dof_override
    return float(2.0 * stats.t.sf(abs(b_x / np.sqrt(var)), df=d))


def full_sigma2(y, t, Z, S):
    D = _design(t, Z, S)
    r = _resid(D, y)
    dof = len(y) - D.shape[1]
    return (max(r @ r, 0.0) / dof if dof > 0 else np.nan), dof


def certify(y, t, Z, S, m_eff, pooled):
    thr = ALPHA / m_eff
    s2, dof = full_sigma2(y, t, Z, S) if pooled else (None, None)
    keep = []
    for j in S:
        others = [s for s in S if s != j]
        worst = 0.0
        for r in range(len(others) + 1):
            for A in combinations(others, r):
                worst = max(worst, p_interaction(y, t, Z, j, list(A),
                                                sigma2=s2, dof_override=dof))
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
    S_full = list(nexis(y=y, t=t, z=Z, alpha=ALPHA, max_rounds=MAX_STEPS).selected)
    o2 = certify(y, t, Z, S_full, m, pooled=False)

    nz = (Z != 0)
    cols = np.where((nz[t == 1].sum(axis=0) >= MIN_ARM)
                    & (nz[t == 0].sum(axis=0) >= MIN_ARM))[0]
    row = {"tree": tree, "n": n, "effect": effect, "seed": seed, "m": m,
           "m_scr": len(cols), "truth_kept": len(set(TRUTH) & set(cols.tolist()))}
    if len(cols):
        Zs = Z[:, cols]
        Ss = list(nexis(y=y, t=t, z=Zs, alpha=ALPHA, max_rounds=MAX_STEPS).selected)
        a = [int(cols[j]) for j in certify(y, t, Zs, Ss, len(cols), pooled=False)]
        b = [int(cols[j]) for j in certify(y, t, Zs, Ss, len(cols), pooled=True)]
    else:
        a, b = [], []
    for name, sel in [("base", S_full), ("opt2", o2), ("scr_opt2", a),
                      ("scr_opt2_pool", b)]:
        for k, v in score(sel, TRUTH).items():
            row[f"{name}_{k}"] = v
    return row


tasks = [(tr, n, e, s) for (tr, n, e) in CELLS for s in range(N_SEEDS)]
print(f"{len(tasks)} runs (MIN_ARM={MIN_ARM}) …", flush=True)
res = Parallel(n_jobs=32, prefer="threads", verbose=5)(delayed(one)(*t) for t in tasks)
df = pd.DataFrame([r for r in res if r])
df.to_csv(HERE / "opt2_final.csv", index=False)
print(f"wrote {len(df)} rows")
