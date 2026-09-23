#!/usr/bin/env python3
"""Population screen: which natural CelebA modifier sets could activate the interleaved
backward step of NEXIS?

The whole CelebA table (19,867 images) is treated as the population (reweighted to the
independent-attribute sampler with --sampler indep).  For a modifier set W_1..W_r with
coefficients gamma, tau = sum_k gamma_k W_k, and for a conditioning set S the linear
interaction test's t-statistic scales as sqrt(n) times

    t_pop(j | S) = Cov(Z^j_res, tau) / SD(Z^j_res),   Z^j_res = Z^j residualised on [1, Z^S]

(noise-free population analogue; used only to rank and screen settings).  The screen
runs the NEXIS forward path on t_pop with rho off until every principal coordinate
(best-threshold-F1 coordinate per modifier) is in S or max steps is reached, and
reports

  early       non-principal coordinates admitted before the last principal
  early_ratio for each: t_pop(x | S minus x) / t_pop at its admission, with S the
              final path set.  Small ratio = x became redundant once the principals
              were in, i.e. what the interleaved backward step would remove.
  rho_ok      whether the path reaches all principals with rho = 0.5
  term_ratio  min over principals p of  min_A t_pop(p | A) (A over subsets of the path
              set without p)  /  min_A over subsets of the principals only: the recall
              handicap the early coordinates impose on the terminal step.

Usage: interleaved_backward_screen.py <dict> <r> [--sampler natural|indep]
Writes results/celeba/interleaved_backward/screen/<dict>_r<r>_<sampler>.csv
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser()
ap.add_argument("dict")
ap.add_argument("r", type=int)
ap.add_argument("--sampler", default="natural", choices=["natural", "indep"])
ap.add_argument("--min-gap", type=float, default=0.08)
ap.add_argument("--max-prev", type=float, default=0.5)
ap.add_argument("--equal-only", action="store_true", help="only |gamma_k| = 1")
args = ap.parse_args()

OUT = ROOT / "results/celeba/interleaved_backward/screen"
OUT.mkdir(parents=True, exist_ok=True)
out_path = OUT / (f"{args.dict}_r{args.r}_{args.sampler}"
                  + ("_eq" if args.equal_only else "") + ".csv")

lab = pd.read_parquet(ROOT / "data/celeba/labels.parquet")
npz = np.load(ROOT / f"results/celeba/interleaved_backward/align/f1_{args.dict}.npz",
              allow_pickle=True)
A_ALL, F1 = list(npz["attrs"]), npz["f1"]
Z = np.load(ROOT / "data/celeba/embeddings" / f"{args.dict}.npy").astype(np.float64)
N, M = Z.shape
Zc = Z - Z.mean(axis=0)            # centred once; weights applied per config
zz = (Zc ** 2).sum(axis=0)

# candidate attributes: a dominant coordinate, not too common, distinct principals
cands, seen = [], set()
for i in np.argsort([-(np.sort(f)[-1] - np.sort(f)[-2]) for f in F1]):
    a = A_ALL[i]
    f = F1[i]
    gap = float(np.sort(f)[-1] - np.sort(f)[-2])
    top = int(np.argmax(f))
    if gap < args.min_gap or lab[a].mean() > args.max_prev or top in seen:
        continue
    seen.add(top)
    cands.append((a, top, float(f.max()), gap))
print(f"{args.dict}: {len(cands)} candidate attributes: "
      f"{[(a, t, round(g, 3)) for a, t, _, g in cands]}", flush=True)
PRINC = {a: t for a, t, _, _ in cands}


def weights(attrs):
    if args.sampler == "natural":
        return None
    W = lab[attrs].values.astype(int)
    p = W.mean(axis=0)
    codes = W @ (1 << np.arange(W.shape[1]))
    cnt = np.bincount(codes, minlength=1 << W.shape[1])
    pind = np.prod(np.where(W == 1, p, 1 - p), axis=1)
    return pind / (cnt[codes] / len(codes))


def t_pop(tau, w, S, cand=None):
    """Vectorised t_pop(j | S) for all j (or cand)."""
    sw = np.ones(N) if w is None else np.sqrt(w / w.mean())
    D = np.column_stack([np.ones(N)] + [Zc[:, s] for s in S]) * sw[:, None]
    Q, _ = np.linalg.qr(D)
    tt = tau * sw
    tt = tt - Q @ (Q.T @ tt)
    X = Zc if cand is None else Zc[:, cand]
    if w is None:
        QX = Q.T @ X
        c = X.T @ tt - QX.T @ (Q.T @ tt)
        v = (zz if cand is None else zz[cand]) - (QX ** 2).sum(axis=0)
    else:
        Xw = X * sw[:, None]
        QX = Q.T @ Xw
        c = Xw.T @ tt
        v = (Xw ** 2).sum(axis=0) - (QX ** 2).sum(axis=0)
    return np.where(v > 1e-9 * N, c / np.sqrt(np.maximum(v, 1e-12)), 0.0) / np.sqrt(N)


def screen(attrs, gam):
    P = [PRINC[a] for a in attrs]
    w = weights(attrs)
    tau = lab[attrs].values.astype(float) @ np.asarray(gam, float)
    tau = tau - (tau.mean() if w is None else np.average(tau, weights=w))
    S, ts, rho_ok = [], [], True
    for _ in range(len(attrs) + 3):
        t = np.abs(t_pop(tau, w, S))
        t[S] = 0.0
        j = int(np.argmax(t))
        if ts and t[j] < 0.5 * min(ts) and not set(P) <= set(S):
            rho_ok = False
        S.append(j)
        ts.append(float(t[j]))
        if set(P) <= set(S):
            break
    reached = set(P) <= set(S)
    last_p = max(S.index(p) for p in P if p in S) if any(p in S for p in P) else -1
    early = [x for x in S[:last_p] if x not in P]
    early_ratio = []
    for x in early:
        tx = abs(float(t_pop(tau, w, [s for s in S if s != x], [x])[0]))
        early_ratio.append(tx / ts[S.index(x)])
    term_ratio = np.nan
    if early and reached:
        def worst(p, pool):
            others = [s for s in pool if s != p]
            return min(abs(float(t_pop(tau, w, list(A_), [p])[0]))
                       for k in range(len(others) + 1)
                       for A_ in itertools.combinations(others, k))
        term_ratio = min(worst(p, S) / max(worst(p, P), 1e-12) for p in P)
    return {"attrs": " ".join(attrs), "gammas": " ".join(f"{g:g}" for g in gam),
            "principals": " ".join(map(str, P)), "path": " ".join(map(str, S)),
            "path_t": " ".join(f"{x:.3f}" for x in ts), "reached": reached,
            "rho_ok": rho_ok, "early": " ".join(map(str, early)),
            "n_early": len(early),
            "early_ratio": " ".join(f"{x:.2f}" for x in early_ratio),
            "min_early_ratio": min(early_ratio) if early_ratio else np.nan,
            "term_ratio": term_ratio}


def patterns(r):
    """Sign patterns up to a global flip, x magnitude patterns (equal / one x2)."""
    out = []
    for signs in itertools.product((1, -1), repeat=r - 1):
        s = (1,) + signs
        out.append(tuple(float(x) for x in s))
        if args.equal_only:
            continue
        for k in range(r):
            out.append(tuple(float(x) * (2.0 if i == k else 1.0) for i, x in enumerate(s)))
    return out


if __name__ == "__main__":
    names = [a for a, *_ in cands]
    tasks = [(list(c), g) for c in itertools.combinations(names, args.r)
             for g in patterns(args.r)]
    print(f"{len(tasks)} configs", flush=True)
    nj = int(os.environ.get("NJOBS", "8"))
    rows = Parallel(n_jobs=nj, prefer="threads", verbose=2)(
        delayed(screen)(a, g) for a, g in tasks)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    hit = df[(df.n_early > 0) & df.reached]
    print(f"wrote {len(df)} -> {out_path}; with early non-principal: {len(hit)}; "
          f"of which rho_ok: {int(hit.rho_ok.sum())}", flush=True)
