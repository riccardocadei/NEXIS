"""Level-aware clustered CATE test for NEXIS on multilevel data, and post-hoc checks.

The CATE test inside NEXIS is the linear T x Z_j interaction t-test.  When treatment is
assigned to a unit coarser than the row, or the candidates vary at coarser levels than
the row, treating rows as independent overstates the evidence.  This module replaces
the test with a level-aware clustered test (a drop-in `pvalue_fn` for nexis()) and
provides the design-based checks run post hoc on the final sets.

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
    each (side of Z_j) x (arm) cell.  A candidate with a mass point (binary or sparse)
    whose minority side has fewer than `min_support` clusters in either arm is not
    testable at L_j (the few-treated-clusters failure, MacKinnon & Webb 2017) and is
    removed from the pool before the search.  The gate uses only Z, T and the
    clusters, never Y, so Bonferroni over the remaining pool stays valid.

The same test runs in the forward step, the interleaved backward step and the terminal
subset enumeration.  Post hoc: randomization inference replaying the lottery of the
assignment units within blocks, and a restricted wild cluster bootstrap-t.

A pool `d` is a dict with keys y, t, z (n x m), names, published, levels (name ->
partition codes), order (level names, coarse to fine), assign, block (a level name or
None), community.  The application pools are built in src/apps/{uganda,ghana}/pool.py.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse as sp
from scipy import stats
from scipy.sparse.csgraph import connected_components

from .core import nexis, conditional_interaction_pvalues

ALPHA, RHO, MIN_SUPPORT = 0.05, 0.5, 5
PUBLISHED_CFG = dict(backward=True, terminal_filter=False)
NEW_DEFAULT = dict(backward=False, terminal_filter=True)
VARIANTS = {"published NEXIS": PUBLISHED_CFG, "new default": NEW_DEFAULT}


# ── Partitions ────────────────────────────────────────────────────────────────

def codes(x) -> np.ndarray:
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


def plain_test(cluster=None, hc1=False, m=None):
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

def block_fe(block):
    c = codes(block)
    return np.column_stack([(c == b).astype(float) for b in range(1, c.max() + 1)])


def build_rule(d, min_support=MIN_SUPPORT):
    """Per-candidate level, cluster partition L_j, G_j and support at L_j.

    Returns (keys, parts, info): the partition key of every candidate, the partitions
    by key (None = HC1), and per candidate a dict with its level, cluster, G, support
    and whether it passes the support gate (`testable`)."""
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
        if cnt.max() >= 0.1 * fin.sum():        # mass point: binary or sparse
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


# ── Reporting helpers ─────────────────────────────────────────────────────────

def cells(d, keys, parts, j):
    """Clusters at L_j in each (active = off the mode / inactive) x (treated / control) cell.
    For a column without a mass point (continuous), only the clusters per arm."""
    col, t = d["z"][:, j], d["t"]
    g = parts[keys[j]]
    g = np.arange(len(t)) if g is None else g
    cnt = lambda msk: int(len(np.unique(g[msk])))       # noqa: E731
    vals, c = np.unique(col[np.isfinite(col)], return_counts=True)
    out = dict(cluster=keys[j], n_clusters=int(len(np.unique(g))),
               treated=cnt(t == 1), control=cnt(t == 0))
    if c.max() >= 0.1 * np.isfinite(col).sum():
        act = np.isfinite(col) & (col != vals[c.argmax()])
        out.update(mode=float(vals[c.argmax()]),
                   active_treated=cnt(act & (t == 1)), active_control=cnt(act & (t == 0)),
                   inactive_treated=cnt(~act & (t == 1)), inactive_control=cnt(~act & (t == 0)))
    return out


def cell_str(c):
    if "active_treated" in c:
        return f"active {c['active_treated']}/{c['active_control']} of {c['treated']}/{c['control']} {c['cluster']} (T/C)"
    return f"continuous; {c['treated']}/{c['control']} {c['cluster']} (T/C)"


def kept_dropped(sel, published):
    return dict(kept=[n for n in published if n in sel], dropped=[n for n in published if n not in sel],
                new=[n for n in sel if n not in published])


def worst_subset(fn, y, t, z, S, j, gate, full_max=10):
    """max over A ⊆ S\\{j} of p(j | A) and its argmax.

    Enumerated in full when |S| <= full_max.  Beyond that (only the unguarded
    sensitivity runs get there) the enumeration stops at the first A with
    p > gate, largest subsets first as in nexis(); the returned p is then a lower
    bound on the worst p and `exact` is False."""
    others = [s_ for s_ in S if s_ != j]
    exact = len(S) <= full_max
    worst, arg = -1.0, []
    for r in range(len(others), -1, -1):
        for A in combinations(others, r):
            p = float(fn(y, t, z, list(A), [j])[j])
            if p > worst:
                worst, arg = p, list(A)
            if not exact and p > gate:
                return worst, arg, False
    return worst, arg, True


def run(d, label, fn, cfg, pool=None, info=None, rule=None):
    """One NEXIS run on the columns `pool` of d["z"] with the test `fn`, and for every
    coordinate of S~ and of the final set its marginal p, its p given the rest of S~
    and of the final set, and its worst-subset p.  `info` (from build_rule) adds each
    coordinate's level and cluster; `rule` (the output of build_rule) adds its cluster
    counts per cell."""
    y, t = d["y"], d["t"]
    pool = list(range(d["z"].shape[1])) if pool is None else list(pool)
    z = d["z"][:, pool]
    if isinstance(fn, LevelAwareTest):
        fn = fn.subset(pool)
    nm = [d["names"][k] for k in pool]
    r = nexis(y, t, z, alpha=ALPHA, max_rounds=20, adjust="FWER", rho=RHO,
              pvalue_fn=fn, **cfg)
    m = z.shape[1]
    gate = ALPHA / m
    sel = list(r.selected)
    S_tilde = list(r.metadata["terminal_candidates"]) if cfg["terminal_filter"] else sel
    rows = []
    for j in sorted(set(S_tilde) | set(sel), key=lambda k: S_tilde.index(k) if k in S_tilde else 99):
        p_marg = float(fn(y, t, z, [], [j])[j])
        p_cond = float(fn(y, t, z, [k for k in S_tilde if k != j], [j])[j])
        p_final = float(fn(y, t, z, [k for k in sel if k != j], [j])[j]) if j in sel else None
        wp, wa, exact = worst_subset(fn, y, t, z, S_tilde, j, gate)
        row = dict(name=nm[j], selected=j in sel, p_marginal=p_marg,
                   p_cond_Stilde=p_cond, p_cond_final=p_final,
                   worst_subset_p=wp, worst_A=[nm[k] for k in wa], worst_exact=exact,
                   passes_terminal=bool(wp <= gate))
        if info is not None:
            ii = info[pool[j]]
            row.update(level=ii["level"], cluster=ii["cluster"], G=ii["G"])
        rows.append(row)
    out = dict(run=label, config="published (interleaved backward)" if cfg["backward"]
               else "new default (forward + terminal)", m=m, gate=gate,
               S_tilde=[nm[k] for k in S_tilde], selected=[nm[k] for k in sel], coords=rows)
    print(f"  [{label:3s}] m={m:3d} a/m={gate:.2e}  S~={out['S_tilde']}  ->  {out['selected']}")
    for rw in rows:
        lvl = f" {rw.get('cluster', ''):>16s} G={rw.get('G', '')}" if info is not None else ""
        pf = f"{rw['p_cond_final']:.1e}" if rw["p_cond_final"] is not None else "   -   "
        print(f"        {rw['name']:18s}{lvl}  marg {rw['p_marginal']:.1e}  cond|S~ {rw['p_cond_Stilde']:.1e}"
              f"  cond|final {pf}  worst {'' if rw['worst_exact'] else '>='}{rw['worst_subset_p']:.1e} at {rw['worst_A']}"
              f"  {'sel' if rw['selected'] else ''}")
    if rule is not None:
        idx = {n: j for j, n in enumerate(d["names"])}
        keys, parts, _ = rule
        for row in rows:
            row["cells"] = cells(d, keys, parts, idx[row["name"]])
        for row in rows:
            if row["selected"]:
                print(f"        cells {row['name']:14s} {cell_str(row['cells'])}")
    return out


# ── Post-hoc design-based checks ─────────────────────────────────────────────

def _cr_last(X, y, C, factor_G):
    """OLS coefficient on the last column of X and its CR1S (or HC1 if C is None) t.

    A singular design returns (None, 0.0).  In the permutation and bootstrap draws below
    this happens when a draw leaves no treated or no control cluster on the active side
    of a sparse Z_j, so T x Z_j is collinear; with t = 0 the draw does not count as at
    least as extreme as a nonzero observed statistic.  The callers count these draws
    (`degenerate_draws`)."""
    n, k = X.shape
    XtX = X.T @ X
    try:
        b = np.linalg.solve(XtX, X.T @ y)
        a = np.linalg.solve(XtX, np.eye(k)[:, -1])
    except np.linalg.LinAlgError:
        return None, 0.0
    e = y - X @ b
    u = X @ a
    if C is None:
        v = ((u * e) ** 2).sum() * n / (n - k)
    else:
        s = C.T @ (u * e)
        G = C.shape[1]
        v = (s ** 2).sum() * (G / (G - 1)) * ((n - 1) / (n - k))
    return b[-1], b[-1] / np.sqrt(v)


def _design(t, z, S, j, fe):
    n = len(t)
    cols = [np.ones(n), t] + ([] if fe is None else list(fe.T))
    cols += [z[:, k] for k in S] + [t * z[:, k] for k in S] + [z[:, j], t * z[:, j]]
    X = np.column_stack(cols)
    # drop collinear columns, always keeping the interaction (last)
    _, R, P = scipy.linalg.qr(X[:, :-1], mode="economic", pivoting=True)
    dd = np.abs(np.diag(R))
    keep = sorted(P[: int((dd > 1e-10 * dd[0]).sum())].tolist()) + [X.shape[1] - 1]
    return keep


def randomization_test(d, S, j, n_perm, seed=0):
    """Replay the lottery: permute the unit-level T (unit = d["assign"]) within blocks
    (d["block"]), holding the number of treated units per block fixed.  Null: the effect
    is linear in Z_S (Y(0) imputed from the restricted model without T*Z_j).  Statistic:
    interaction t with block FE, clustered by the assignment unit."""
    rng = np.random.default_rng(seed)
    y, t, z = d["y"], d["t"], d["z"]
    grp, dist = d["levels"][d["assign"]], d["levels"][d["block"]]
    fe = block_fe(dist)
    C = sp.csr_matrix((np.ones(len(grp)), (np.arange(len(grp)), grp)))
    keep = _design(t, z, S, j, fe)
    X = lambda tt: np.column_stack([np.ones(len(tt)), tt] + list(fe.T) + [z[:, k] for k in S]  # noqa: E731
                                   + [tt * z[:, k] for k in S] + [z[:, j], tt * z[:, j]])[:, keep]
    b_obs, t_obs = _cr_last(X(t), y, C, True)
    degenerate = int(b_obs is None)
    Xr = X(t)[:, :-1]
    br = np.linalg.lstsq(Xr, y, rcond=None)[0]
    names_r = keep[:-1]
    # effect under the null: coefficient on T plus the T*Z_S terms
    nfe, ns = fe.shape[1], len(S)
    tau = np.full(len(y), br[names_r.index(1)])
    for i, k in enumerate(S):
        col = 2 + nfe + ns + i
        if col in names_r:
            tau += br[names_r.index(col)] * z[:, k]
    y0 = y - t * tau
    ug = np.unique(grp)
    g_dist = np.array([dist[grp == g][0] for g in ug])
    g_t = np.array([t[grp == g][0] for g in ug])
    blocks = [np.where(g_dist == b)[0] for b in np.unique(g_dist)]
    n_t = [int(g_t[b].sum()) for b in blocks]
    count = 0
    for _ in range(n_perm):
        gt = np.zeros(len(ug))
        for b, k in zip(blocks, n_t):
            gt[rng.choice(b, size=k, replace=False)] = 1.0
        ts = gt[grp]
        b_star, t_star = _cr_last(X(ts), y0 + ts * tau, C, True)
        degenerate += b_star is None
        count += abs(t_star) >= abs(t_obs) - 1e-12
    return dict(t_obs=float(t_obs), p=(1 + count) / (n_perm + 1), floor=1 / (n_perm + 1),
                degenerate_draws=degenerate)


def wild_cluster_bootstrap(d, S, j, cluster, n_boot, seed=0):
    """Restricted (null-imposed) wild cluster bootstrap-t, Rademacher weights."""
    rng = np.random.default_rng(seed)
    y, t, z = d["y"], d["t"], d["z"]
    g = codes(cluster)
    C = sp.csr_matrix((np.ones(len(g)), (np.arange(len(g)), g)))
    keep = _design(t, z, S, j, None)
    cols = [np.ones(len(t)), t] + [z[:, k] for k in S] + [t * z[:, k] for k in S] + [z[:, j], t * z[:, j]]
    X = np.column_stack(cols)[:, keep]
    b_obs, t_obs = _cr_last(X, y, C, True)
    degenerate = int(b_obs is None)
    Xr = X[:, :-1]
    fit = Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]
    res = y - fit
    count = 0
    for _ in range(n_boot):
        w = rng.choice([-1.0, 1.0], size=C.shape[1])[g]
        b_star, tb = _cr_last(X, fit + w * res, C, True)
        degenerate += b_star is None
        count += abs(tb) >= abs(t_obs) - 1e-12
    return dict(t_obs=float(t_obs), p=(1 + count) / (n_boot + 1), floor=1 / (n_boot + 1),
                degenerate_draws=degenerate)
