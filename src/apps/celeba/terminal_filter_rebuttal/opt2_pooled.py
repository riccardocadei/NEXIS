#!/usr/bin/env python3
"""Opt2 with a pooled (full-model) error variance.

DIAGNOSIS.  In this DGP W1 and W2 are drawn independently and cor(z_5348, z_5537)=0.04,
so there is NO cancellation: gamma_j(A) is essentially the same for every A.  What
differs across A is the ERROR VARIANCE: omitting a co-modifier leaves its -eta*T*W2
term in the residual, inflating sigma^2 and deflating every t-statistic.  Opt2 takes the
max p-value over A, i.e. it is dominated by the LEAST EFFICIENT member of the family,
and then judges it at a threshold calibrated for the most efficient one.

FIX.  Keep the intersection-union test over ALL subsets (including the empty set), but
give every component test the SAME, efficient error variance: sigma^2 estimated from the
richest model (A = S_cand\\{j} plus j).  Only the design-based factor zz/det then varies
across A, not the noise estimate.

VALIDITY under the paper's faithfulness assumption.  The union null is true only when j
is non-adjacent, and faithfulness then forces gamma_j(A)=0 for EVERY A -- including the
full set, where the pooled sigma^2 is the correct variance.  So the component test that
matters under H_0 is correctly calibrated.  (Without faithfulness, i.e. if gamma_j(0)=0
while gamma_j(full)!=0, the pooled variance would be anti-conservative for the A=0
component -- that is exactly the configuration faithfulness excludes.)

Berger (1982): an IUT rejecting when max_A p_j(A) <= alpha has level alpha; the 2^|S|-1
subsets cost NOTHING.  So alpha/m stays as-is (alpha for the IUT, 1/m for the m
coordinates).

Variants compared:
  base          NEXIS as published
  opt2          max over ALL A of the standard p-value           <= alpha/m
  opt2_ne       max over NON-EMPTY A (the earlier hack)          <= alpha/m
  opt2_pooled   max over ALL A, pooled full-model sigma^2        <= alpha/m
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

ALPHA, MAX_STEPS, N_SEEDS = 0.05, 10, 50
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
    """Residualize columns of M against design D (least squares)."""
    if D.shape[1] == 0:
        return M
    coef, *_ = np.linalg.lstsq(D, M, rcond=None)
    return M - D @ coef


def _design(t, Z, A):
    cols = [np.ones(len(t)), t]
    for k in A:
        cols.append(Z[:, k])
    for k in A:
        cols.append(t * Z[:, k])
    return np.column_stack(cols)


def interaction_fit(y, t, Z, j, A, sigma2=None, dof_override=None):
    """gamma_hat_j(A), its SE and p-value.  If sigma2 given, use it (pooled) instead of
    the sub-model residual variance."""
    n = len(y)
    D = _design(t, Z, A)
    y_t = _resid(D, y)
    Zj = Z[:, j]
    Zt = _resid(D, Zj.reshape(-1, 1))[:, 0]
    Xt = _resid(D, (t * Zj).reshape(-1, 1))[:, 0]
    zz, xx, zx = Zt @ Zt, Xt @ Xt, Zt @ Xt
    zy, xy = Zt @ y_t, Xt @ y_t
    det = zz * xx - zx * zx
    p_full = D.shape[1] + 2
    dof = n - p_full
    if det <= 1e-12 or dof <= 0:
        return dict(gamma=0.0, se=np.inf, p=1.0, sigma2=np.nan, t=0.0)
    b_x = (zz * xy - zx * zy) / det
    b_z = (xx * zy - zx * xy) / det
    e = y_t - Zt * b_z - Xt * b_x
    s2_sub = max((e @ e), 0.0) / dof
    s2 = s2_sub if sigma2 is None else sigma2
    d = dof if dof_override is None else dof_override
    var = s2 * (zz / det)
    if not np.isfinite(var) or var <= 0:
        return dict(gamma=b_x, se=np.inf, p=1.0, sigma2=s2_sub, t=0.0)
    tstat = b_x / np.sqrt(var)
    return dict(gamma=b_x, se=np.sqrt(var), sigma2=s2_sub, t=tstat,
                p=float(2.0 * stats.t.sf(abs(tstat), df=d)))


def full_model_sigma2(y, t, Z, S):
    """sigma^2 from the richest model: [1, T, Z_S, T*Z_S] with S = full candidate set."""
    n = len(y)
    D = _design(t, Z, S)
    r = _resid(D, y)
    dof = n - D.shape[1]
    return (max(r @ r, 0.0) / dof if dof > 0 else np.nan), dof


def one(tree, n, effect, seed):
    feats, TRUTH = load(tree)
    m = feats.shape[1]
    try:
        d = generate_celeba_rct(n=n, features=feats, labels_df=labels_df,
                                buckets=buckets, effect_scale=effect, seed=seed, **scm)
    except ValueError:
        return None, []
    y, t, Z = d.Y, d.T, d.Z
    S = list(nexis(y=y, t=t, z=Z, alpha=ALPHA, max_rounds=MAX_STEPS).selected)
    thr = ALPHA / m
    s2_full, dof_full = full_model_sigma2(y, t, Z, S)

    keep = {"base": list(S), "opt2": [], "opt2_ne": [], "opt2_pooled": []}
    detail = []
    for j in S:
        others = [s for s in S if s != j]
        worst = {"opt2": (-1.0, None), "opt2_ne": (-1.0, None), "opt2_pooled": (-1.0, None)}
        for r in range(len(others) + 1):
            for A in combinations(others, r):
                st = interaction_fit(y, t, Z, j, list(A))
                pl = interaction_fit(y, t, Z, j, list(A), sigma2=s2_full,
                                     dof_override=dof_full)
                if st["p"] > worst["opt2"][0]:
                    worst["opt2"] = (st["p"], A)
                if r > 0 and st["p"] > worst["opt2_ne"][0]:
                    worst["opt2_ne"] = (st["p"], A)
                if pl["p"] > worst["opt2_pooled"][0]:
                    worst["opt2_pooled"] = (pl["p"], A)
                if r in (0, len(others)):   # record empty-set and full-set diagnostics
                    detail.append({"tree": tree, "n": n, "effect": effect, "seed": seed,
                                   "j": j, "is_truth": j in TRUTH, "n_cand": len(S),
                                   "subset": "empty" if r == 0 else "full",
                                   "gamma": st["gamma"], "se": st["se"], "t": st["t"],
                                   "sigma2_sub": st["sigma2"], "sigma2_full": s2_full,
                                   "p_std": st["p"], "p_pooled": pl["p"]})
        for k in ("opt2", "opt2_ne", "opt2_pooled"):
            pv = worst[k][0]
            if pv >= 0 and pv <= thr:
                keep[k].append(j)
            elif pv < 0:            # no subset of that kind (|S|=1 and non-empty family)
                keep[k].append(j)

    row = {"tree": tree, "n": n, "effect": effect, "seed": seed, "m": m}
    ts = set(TRUTH)
    for k, sel in keep.items():
        ss = set(sel)
        tp, kk = len(ss & ts), len(ss)
        den = kk + len(ts) - tp
        row[f"{k}_tp"] = tp
        row[f"{k}_k"] = kk
        row[f"{k}_recall"] = tp / len(ts)
        row[f"{k}_precision"] = (tp / kk) if kk else 0.0
        row[f"{k}_iou"] = (tp / den) if den else 0.0
    return row, detail


tasks = [(tr, n, e, s) for (tr, n, e) in CELLS for s in range(N_SEEDS)]
print(f"{len(tasks)} runs …", flush=True)
res = Parallel(n_jobs=32, prefer="threads", verbose=5)(delayed(one)(*t) for t in tasks)
rows = [r for r, _ in res if r]
det = [x for _, dd in res for x in dd]
pd.DataFrame(rows).to_csv(HERE / "opt2_pooled.csv", index=False)
pd.DataFrame(det).to_csv(HERE / "opt2_pooled_detail.csv", index=False)
print(f"wrote {len(rows)} metric rows, {len(det)} diagnostic rows")
