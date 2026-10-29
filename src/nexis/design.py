"""Level-aware clustered CATE test for NEXIS on multilevel data.

The CATE test inside NEXIS is the linear T x Z_j interaction t-test.  When treatment is
assigned to a unit coarser than the row, or the candidates vary at coarser levels than
the row, treating rows as independent overstates the evidence.  This module replaces
the test with a level-aware clustered test (a drop-in `pvalue_fn` for nexis()); the
design-based checks run post hoc on the final sets are in nexis.checks.

Correction rule
---------------
For candidate j, cluster the T x Z_j score at L_j = the finest partition that is
coarser than both

  (a) the unit at which treatment was assigned (`d["assign"]`),
  (b) the coarsest level at which Z_j varies (detected from the data, coarse to fine).

L_j is the join of the two partitions (connected components of the bipartite graph
between them), so it stays well defined when they are not nested.  The test is CR1S
with a t(G_j - 1) reference, or HC1 with t(n - p) when L_j is the row itself.  Two
further parts of the rule:

  * Blocking.  When assignment was blocked with unequal treated shares, the block
    fixed effects enter the nuisance design (`d["block"]`).
  * Support.  A clustered t-test on T x Z_j is identified by the clusters that hold
    each (side of Z_j) x (arm) cell.  For a candidate with a mass point (its modal
    value holds >= 10% of the rows: binary or sparse; or, optionally, one declared
    binary, count or sparse), the support is the smallest number of distinct L_j
    clusters over the four cells (Z_j = mode, Z_j != mode) x (T = 1, T = 0); rows
    count as clusters under HC1.  A candidate with support below
    `min_support` is not testable at L_j (the few-treated-clusters failure, MacKinnon
    & Webb 2017) and is removed from the pool before the search; candidates without a
    mass point (continuous) have no support requirement.  The gate uses only Z, T and
    the clusters, never Y, so Bonferroni over the remaining pool stays valid.
    Candidates with a missing value are not testable either.

The same test runs in the forward step, the interleaved backward step and the terminal
subset enumeration.

A pool `d` is a dict with keys y, t, z (n x m), names, levels (level name -> partition
codes, one per row), order (level names, coarse to fine), assign (the level name of the
assignment unit) and, for the randomization test, block (a level name).  The application
pools (src/apps/{uganda,ghana}/pool.py) carry a few more keys for their reports;
nexis.discover builds one from a Dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse as sp
from scipy import stats
from scipy.sparse.csgraph import connected_components

from .pvalues import conditional_interaction_pvalues

MIN_SUPPORT = 5


# ── Partitions ────────────────────────────────────────────────────────────────

def codes(x) -> np.ndarray:
    """Integer codes 0..G-1 of the labels in x (compared as strings)."""
    return pd.factorize(pd.Series(np.asarray(x)).astype(str))[0]


def join(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Finest partition coarser than both a and b (connected components)."""
    a, b = codes(a), codes(b)
    na = a.max() + 1
    N = na + b.max() + 1
    A = sp.coo_matrix((np.ones(len(a)), (a, b + na)), shape=(N, N))
    _, lab = connected_components(A, directed=False)
    return codes(lab[a])


def is_coarser(a: np.ndarray, b: np.ndarray) -> bool:
    """True when every cell of b sits inside one cell of a."""
    return bool((pd.Series(a).groupby(b).nunique() == 1).all())


def variation_level(col: np.ndarray, levels: dict, order: list, tol=1e-10) -> str:
    """Coarsest level within whose cells `col` is constant (order is coarse → fine)."""
    scale = np.nanstd(col) or 1.0
    for name in order:
        s = pd.Series(col).groupby(levels[name]).transform("std").fillna(0.0).max()
        if s / scale < tol:
            return name
    return order[-1]


# ── Level-aware clustered interaction test ───────────────────────────────────

class LevelAwareTest:
    """Linear T x Z_j interaction t-test with a per-candidate cluster partition.

    Drop-in `pvalue_fn` for nexis().  Nuisance design D = [1, T, FE, Z_S, T*Z_S]
    (collinear columns dropped by pivoted QR, which matters when block FE absorb a
    main effect); per candidate a 2-regressor FWL fit on [Z_j, T*Z_j], or 1-regressor
    on T*Z_j when FE absorb Z_j.  `keys[j]` names the partition of candidate j in
    `parts`; a partition of None means HC1.  With fe=None and one partition for every
    candidate this reproduces conditional_interaction_pvalues(cluster=...) exactly.
    """

    def __init__(self, keys, parts: dict, fe: np.ndarray | None = None):
        self.keys = np.asarray(keys, dtype=object)
        self.fe = fe
        self.ind = {}
        for k, g in parts.items():
            if g is None:
                self.ind[k] = None
            else:
                c = codes(g)
                self.ind[k] = sp.csr_matrix((np.ones(len(c)), (np.arange(len(c)), c)))

    def subset(self, pool):
        """The same test on the columns `pool` of the candidate matrix."""
        new = object.__new__(LevelAwareTest)
        new.keys, new.fe, new.ind = self.keys[list(pool)], self.fe, self.ind
        return new

    def __call__(self, y, t, z, S, candidates, return_tstats=False):
        n, m = z.shape
        S = sorted(set(int(k) for k in (S or [])))
        cand = np.array([int(j) for j in candidates if int(j) not in S], dtype=int)
        pv, ts = np.ones(m), np.zeros(m)
        if cand.size == 0:
            return (pv, ts) if return_tstats else pv
        cols = [np.ones(n), t]
        if self.fe is not None:
            cols += list(self.fe.T)
        cols += [z[:, k] for k in S] + [t * z[:, k] for k in S]
        D = np.column_stack(cols)
        if not np.isfinite(D).all():          # a NaN column in S: nothing is testable
            return (pv, ts) if return_tstats else pv
        Q, R, _ = scipy.linalg.qr(D, mode="economic", pivoting=True)
        d = np.abs(np.diag(R))
        Q = Q[:, : int((d > 1e-10 * d[0]).sum())]
        res = lambda V_: V_ - Q @ (Q.T @ V_)          # noqa: E731
        yt = res(y)
        Zc = z[:, cand]
        Zt, Xt = res(Zc), res(t[:, None] * Zc)
        zz, xx, zx = (Zt * Zt).sum(0), (Xt * Xt).sum(0), (Zt * Xt).sum(0)
        zy, xy = (Zt * yt[:, None]).sum(0), (Xt * yt[:, None]).sum(0)
        # FE absorb Z_j (e.g. a region dummy is a sum of block dummies)
        scale = ((Zc - np.nanmean(Zc, 0)) ** 2).sum(0)
        one = zz <= 1e-10 * np.maximum(scale, 1e-300)
        det = zz * xx - zx * zx
        two = ~one & (det > 1e-12)
        ok1 = one & (xx > 1e-12)
        bx, bz = np.zeros(cand.size), np.zeros(cand.size)
        bx[two] = (zz[two] * xy[two] - zx[two] * zy[two]) / det[two]
        bz[two] = (xx[two] * zy[two] - zx[two] * xy[two]) / det[two]
        bx[ok1] = xy[ok1] / xx[ok1]
        e = yt[:, None] - Zt * bz[None, :] - Xt * bx[None, :]
        p_full = Q.shape[1] + np.where(one, 1, 2)
        # sandwich pieces, per candidate; the partition depends on the candidate
        zzee, xxee, zxee = (np.full(cand.size, np.nan) for _ in range(3))
        G = np.full(cand.size, np.nan)
        for key in set(self.keys[cand]):
            sel = np.where(self.keys[cand] == key)[0]
            C = self.ind[key]
            if C is None:
                e2 = e[:, sel] ** 2
                zzee[sel] = (Zt[:, sel] ** 2 * e2).sum(0)
                xxee[sel] = (Xt[:, sel] ** 2 * e2).sum(0)
                zxee[sel] = (Zt[:, sel] * Xt[:, sel] * e2).sum(0)
            else:
                Ze = np.asarray(C.T @ (Zt[:, sel] * e[:, sel]))
                Xe = np.asarray(C.T @ (Xt[:, sel] * e[:, sel]))
                zzee[sel], xxee[sel], zxee[sel] = (Ze ** 2).sum(0), (Xe ** 2).sum(0), (Ze * Xe).sum(0)
                G[sel] = C.shape[1]
        hc = np.isnan(G)
        fac = np.where(hc, n / (n - p_full), (G / (G - 1)) * ((n - 1) / (n - p_full)))
        var = np.full(cand.size, np.nan)
        var[two] = (zz[two] ** 2 * xxee[two] - 2 * zz[two] * zx[two] * zxee[two]
                    + zx[two] ** 2 * zzee[two]) / det[two] ** 2
        var[ok1] = xxee[ok1] / xx[ok1] ** 2
        var *= fac
        df = np.where(hc, n - p_full, G - 1)
        good = (two | ok1) & np.isfinite(var) & (var > 0)
        tt = np.zeros(cand.size)
        tt[good] = bx[good] / np.sqrt(var[good])
        p = np.ones(cand.size)
        p[good] = 2 * stats.t.sf(np.abs(tt[good]), df=df[good])
        pv[cand] = np.clip(np.nan_to_num(p, nan=1.0), 0, 1)
        ts[cand] = tt
        return (pv, ts) if return_tstats else pv


def plain_test(cluster: np.ndarray | None = None, hc1: bool = False, m: int | None = None):
    """One clustering (or HC1) for every candidate, no FE.

    Clustered / HC1 variants go through LevelAwareTest, which reproduces
    conditional_interaction_pvalues(cluster=..., hc1=...) and is much faster than its
    per-cluster Python loop, which matters for the terminal subset enumeration.
    Homoskedastic OLS calls the core test directly."""
    if cluster is not None or hc1:
        return LevelAwareTest(["c"] * m, {"c": None if cluster is None else cluster})

    def fn(y, t, z, S, candidates, return_tstats=False):
        return conditional_interaction_pvalues(y=y, t=t, z=z, S=S, candidates=candidates,
                                               return_tstats=return_tstats)
    return fn


# ── The correction rule ───────────────────────────────────────────────────────

def block_fe(block) -> np.ndarray:
    """Dummies of the blocks in `block`, the first block dropped (n x (G - 1))."""
    c = codes(block)
    return np.column_stack([(c == b).astype(float) for b in range(1, c.max() + 1)])


def build_rule(d: dict, min_support: int = MIN_SUPPORT, mass_point=None):
    """Per-candidate level, cluster partition L_j, G_j and support at L_j.

    Uses d["t"], d["z"], d["names"], d["levels"], d["order"] and d["assign"].  The
    support of a candidate with a mass point is the smallest number of distinct L_j
    clusters (rows under HC1) over the four cells (Z_j = mode, Z_j != mode) x
    (T = 1, T = 0); it is None for a candidate without a mass point.  A candidate is
    `testable` when it has no missing value and its support is None or >= min_support.

    mass_point: optional booleans, one per candidate; True gives the candidate the
        support rule even when its data show no mass point (nexis.discover passes the
        columns declared binary, count or sparse).  None (the default) uses the data
        rule alone.

    Returns (keys, parts, info): the partition key of every candidate, the partitions
    by key (None = HC1), and per candidate a dict with its level, cluster, G, support
    and testable."""
    lv, t, z = d["levels"], d["t"], d["z"]
    assign = lv[d["assign"]]
    finest = d["order"][-1]
    parts, keys, info = {}, [], []
    for j, name in enumerate(d["names"]):
        lvl = variation_level(z[:, j], lv, d["order"])
        a = lv[lvl]
        if is_coarser(a, assign):
            key = lvl if lvl != finest else d["assign"]
        elif is_coarser(assign, a):
            key = d["assign"]
        else:
            key = f"{lvl}+{d['assign']}"
        if key == finest:                       # assignment at the row: HC1
            parts[key] = None
        elif key not in parts:
            parts[key] = join(a, assign) if "+" in key else lv[key]
        g = parts[key]
        G = int(len(np.unique(g))) if g is not None else len(t)
        # support: clusters in each (minority side of Z_j) x (arm) cell
        col = z[:, j]
        fin = np.isfinite(col)
        vals, cnt = np.unique(col[fin], return_counts=True)
        mode = vals[cnt.argmax()]
        support = None
        if cnt.max() >= 0.1 * fin.sum() or (mass_point is not None and mass_point[j]):
            gg = g if g is not None else np.arange(len(t))
            side = fin & (col != mode)
            other = fin & (col == mode)
            cells_ = [side & (t == 1), side & (t == 0), other & (t == 1), other & (t == 0)]
            support = int(min(len(np.unique(gg[c])) for c in cells_))
        testable = bool(fin.all()) and (support is None or support >= min_support)
        keys.append(key)
        info.append(dict(name=name, level=lvl, cluster=key, G=G, support=support,
                         testable=testable))
    return keys, parts, info
