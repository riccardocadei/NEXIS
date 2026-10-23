"""Post-hoc design-based checks of a selected set on multilevel data.

  worst_subset             max over A of p(j | A), A a subset of the rest of the set
  randomization_test       replays the lottery of the assignment units within blocks
  wild_cluster_bootstrap   restricted (null-imposed) wild cluster bootstrap-t

A pool `d` is the dict described in nexis.design.  These checks are not run by
nexis.discover; the application scripts run them on the published sets.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from .design import block_fe, codes


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


# ── Post-hoc design-based checks ─────────────────────────────────────────────

def _cr_last(X, y, C):
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


def _kept_columns(t, z, S, j, fe):
    """Columns of [1, T, FE, Z_S, T*Z_S, Z_j, T*Z_j] kept after dropping collinear ones."""
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
    keep = _kept_columns(t, z, S, j, fe)
    X = lambda tt: np.column_stack([np.ones(len(tt)), tt] + list(fe.T) + [z[:, k] for k in S]  # noqa: E731
                                   + [tt * z[:, k] for k in S] + [z[:, j], tt * z[:, j]])[:, keep]
    b_obs, t_obs = _cr_last(X(t), y, C)
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
        b_star, t_star = _cr_last(X(ts), y0 + ts * tau, C)
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
    keep = _kept_columns(t, z, S, j, None)
    cols = [np.ones(len(t)), t] + [z[:, k] for k in S] + [t * z[:, k] for k in S] + [z[:, j], t * z[:, j]]
    X = np.column_stack(cols)[:, keep]
    b_obs, t_obs = _cr_last(X, y, C)
    degenerate = int(b_obs is None)
    Xr = X[:, :-1]
    fit = Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]
    res = y - fit
    count = 0
    for _ in range(n_boot):
        w = rng.choice([-1.0, 1.0], size=C.shape[1])[g]
        b_star, tb = _cr_last(X, fit + w * res, C)
        degenerate += b_star is None
        count += abs(tb) >= abs(t_obs) - 1e-12
    return dict(t_obs=float(t_obs), p=(1 + count) / (n_boot + 1), floor=1 / (n_boot + 1),
                degenerate_draws=degenerate)
