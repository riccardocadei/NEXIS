"""Re-run the real-world NEXIS analyses with a CATE test corrected for multilevel data.

The CATE test inside NEXIS is the linear T x Z_j interaction t-test.  The published
runs treat every row as independent: Uganda YOP uses homoskedastic OLS SEs, and Ghana
LEAP 1000 uses CR1S SEs clustered by community for every candidate.  In both datasets,
though, treatment is assigned to a unit coarser than the row (Uganda) or the candidates
vary at coarser levels than the row (both).  This script replaces the test with a
level-aware clustered test and re-runs the search.

Correction rule (one rule for both applications)
------------------------------------------------
For candidate j, cluster the T x Z_j score at L_j = the finest partition that is
coarser than both

  (a) the unit at which treatment was assigned (Uganda: the YOP applicant group,
      randomised within district; Ghana: the household, eligible by its own PMT score),
  (b) the coarsest level at which Z_j varies (detected from the data, coarse to fine).

L_j is the join of the two partitions (connected components of the bipartite graph
between them), so it stays well defined when they are not nested (30 of 439 Uganda
groups span two sites).  The test is CR1S with a t(G_j - 1) reference, or HC1 with
t(n - p) when L_j is the row itself (Ghana household-level covariates).  Two further
parts of the rule:

  * Blocking.  When assignment was blocked with unequal treated shares, the block
    fixed effects enter the nuisance design (Uganda: 14 district FE; Blattman, Fiala &
    Martinez 2014 randomise within district and include district FE for this reason).
  * Support.  A clustered t-test on T x Z_j is identified by the clusters that hold
    each (side of Z_j) x (arm) cell.  A candidate with a mass point (binary or sparse)
    whose minority side has fewer than MIN_SUPPORT clusters in either arm is not
    testable at L_j (the few-treated-clusters failure, MacKinnon & Webb 2017) and is
    removed from the pool before the search.  The gate uses only Z, T and the
    clusters, never Y, so Bonferroni over the remaining pool stays valid.

The same test runs in the forward step, the interleaved backward step and the terminal
subset enumeration.  Design-based checks that are too expensive inside the search run
post hoc on the final sets: randomization inference (RI) replaying the Uganda
group lottery within district, and a restricted wild cluster bootstrap for Ghana.

Runs, for each outcome (Uganda skilled employment and log business assets; Ghana
consumption):
  1  published config (interleaved backward) with the published test; must reproduce
     the published set
  2  published config with the level-aware test
  3  new default (forward + terminal backward) with the level-aware test
  3b new default, level-aware clustering but no block FE (Uganda only; isolates FE)
  3c new default, level-aware test without the support gate
  4a new default, one clustering for every candidate: the assignment unit
  4b new default, one clustering for every candidate: the community / site
  3i new default, level-aware test, T = lottery assignment instead of the realised
     grant (Uganda only; intent-to-treat)
Post hoc, Uganda RI is reported for both T = realised grant (as NEXIS uses) and
T = assignment (p_itt; the contrast the lottery actually randomised).
rho = 0.5, alpha = 0.05, FWER forward gate, linear test throughout.

CPU only, a few minutes.  Reads data/ and results/ (local, not tracked); writes
results/realworld_clustered/report.json.

    python scripts/realworld_clustered_nexis.py [--n-perm 9999] [--n-boot 9999]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse as sp
from scipy import stats
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from src.method.nexis import nexis, conditional_interaction_pvalues  # noqa: E402
import verify_new_default_realworld as V  # noqa: E402

ALPHA, RHO, MIN_SUPPORT = 0.05, 0.5, 10
PUBLISHED_CFG = dict(backward=True, terminal_filter=False)
NEW_DEFAULT = dict(backward=False, terminal_filter=True)
OUT = ROOT / "results" / "realworld_clustered"


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
        # FE absorb Z_j (e.g. a language dummy is a sum of district dummies)
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
    conditional_interaction_pvalues(cluster=..., hc1=...) (asserted in analyse) and is
    much faster than its per-cluster Python loop, which matters for the terminal
    subset enumeration.  Homoskedastic OLS calls the core test directly."""
    if cluster is not None or hc1:
        return LevelAwareTest(["c"] * m, {"c": None if cluster is None else cluster})

    def fn(y, t, z, S, candidates, return_tstats=False):
        return conditional_interaction_pvalues(y=y, t=t, z=z, S=S, candidates=candidates,
                                               return_tstats=return_tstats)
    return fn


# ── Data ──────────────────────────────────────────────────────────────────────

def uganda(outcome):
    """Published Uganda pool (via verify_new_default_realworld) plus its hierarchy."""
    from apps.uganda.data import resolve_outcome
    d = V.uganda_data(outcome)
    df = pd.read_csv(ROOT / "data" / "uganda" / "UgandaDataProcessed.csv", low_memory=False)
    Zall = np.load(ROOT / "results" / "uganda" / "prithvi_l5_1024" / "individual_features.npz")["features"]
    mask = df[resolve_outcome(outcome)].notna() & np.isfinite(Zall[:, 0])
    df = df[mask].reset_index(drop=True)
    assert len(df) == len(d["y"]) and np.array_equal(df["Wobs"].values, d["t"])
    levels = {"region": codes(df["lang_group"]), "district": codes(df["district"]),
              "community": codes(df["geo_long_lat_key"]), "group": codes(df["groupid"]),
              "individual": np.arange(len(df))}
    return dict(app="uganda", outcome=outcome, y=d["y"], t=d["t"], z=d["z"], names=d["names"],
                assigned=df["assigned"].values.astype(float),
                published=d["published"], levels=levels,
                order=["region", "district", "community", "group", "individual"],
                assign="group", block="district", community="community",
                published_test=plain_test(), published_label="homoskedastic OLS")


def ghana():
    """Ghana joint pool of 155 = 131 SAE neurons + 24 survey covariates (as the paper states)."""
    from src.apps.ghana.data import load_data, W_ALL
    df = load_data(ROOT / "data" / "ghana")
    both = df.groupby("hhid")["wave"].nunique()
    df = df[df["hhid"].isin(both[both == 2].index)]
    df0, df1 = df[df["wave"] == 0], df[df["wave"] == 1]
    merged = (df0.set_index("hhid")[["T", "comm", "district", "region", "Y"] + W_ALL]
              .join(df1.set_index("hhid")[["Y"]].rename(columns={"Y": "Y1"}))
              .dropna(subset=["Y1"]))
    d = V.ghana_data()
    y = (merged["Y1"] - merged["Y"]).values.astype(float)
    assert np.allclose(y, d["y"]) and np.array_equal(merged["comm"].values, d["run_kw"]["cluster"])
    z = np.hstack([d["z"], d["run_kw"]["w"]])
    names = d["names"] + [f"W_{c}" for c in W_ALL]
    comm = merged["comm"].values
    levels = {"region": codes(merged["region"]), "district": codes(merged["district"]),
              "community": codes(comm), "household": np.arange(len(merged))}
    return dict(app="ghana", outcome="consumption", y=y, t=d["t"], z=z, names=names,
                published=["Z_3821", "Z_2095"], levels=levels,
                order=["region", "district", "community", "household"],
                assign="household", block=None, community="community",
                published_test=plain_test(cluster=comm, m=z.shape[1]), published_label="CR1S by community")


# ── The correction rule ───────────────────────────────────────────────────────

def block_fe(block):
    c = codes(block)
    return np.column_stack([(c == b).astype(float) for b in range(1, c.max() + 1)])


def build_rule(d):
    """Per-candidate level, cluster partition L_j, G_j and support at L_j."""
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
            cells = [side & (t == 1), side & (t == 0), other & (t == 1), other & (t == 0)]
            support = int(min(len(np.unique(gg[c])) for c in cells))
        testable = bool(fin.all()) and (support is None or support >= MIN_SUPPORT)
        keys.append(key)
        info.append(dict(name=name, level=lvl, cluster=key, G=G, support=support,
                         testable=testable))
    return keys, parts, info


# ── Reporting helpers ─────────────────────────────────────────────────────────

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


def run(d, label, fn, cfg, pool=None, info=None):
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
    return out


# ── Post-hoc design-based checks ─────────────────────────────────────────────

def _cr_last(X, y, C, factor_G):
    """OLS coefficient on the last column of X and its CR1S (or HC1 if C is None) t."""
    n, k = X.shape
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ b
    a = np.linalg.solve(XtX, np.eye(k)[:, -1])
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
    """Replay the Uganda lottery: permute group-level T within district, holding the
    number of treated groups per district fixed.  Null: the effect is linear in Z_S
    (Y(0) imputed from the restricted model without T*Z_j).  Statistic: interaction t
    with district FE, clustered by group (the assignment unit)."""
    rng = np.random.default_rng(seed)
    y, t, z = d["y"], d["t"], d["z"]
    grp, dist = d["levels"]["group"], d["levels"]["district"]
    fe = block_fe(dist)
    C = sp.csr_matrix((np.ones(len(grp)), (np.arange(len(grp)), grp)))
    keep = _design(t, z, S, j, fe)
    X = lambda tt: np.column_stack([np.ones(len(tt)), tt] + list(fe.T) + [z[:, k] for k in S]  # noqa: E731
                                   + [tt * z[:, k] for k in S] + [z[:, j], tt * z[:, j]])[:, keep]
    _, t_obs = _cr_last(X(t), y, C, True)
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
        _, t_star = _cr_last(X(ts), y0 + ts * tau, C, True)
        count += abs(t_star) >= abs(t_obs) - 1e-12
    return dict(t_obs=float(t_obs), p=(1 + count) / (n_perm + 1), floor=1 / (n_perm + 1))


def wild_cluster_bootstrap(d, S, j, cluster, n_boot, seed=0):
    """Restricted (null-imposed) wild cluster bootstrap-t, Rademacher weights."""
    rng = np.random.default_rng(seed)
    y, t, z = d["y"], d["t"], d["z"]
    g = codes(cluster)
    C = sp.csr_matrix((np.ones(len(g)), (np.arange(len(g)), g)))
    keep = _design(t, z, S, j, None)
    cols = [np.ones(len(t)), t] + [z[:, k] for k in S] + [t * z[:, k] for k in S] + [z[:, j], t * z[:, j]]
    X = np.column_stack(cols)[:, keep]
    _, t_obs = _cr_last(X, y, C, True)
    Xr = X[:, :-1]
    fit = Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]
    res = y - fit
    count = 0
    for _ in range(n_boot):
        w = rng.choice([-1.0, 1.0], size=C.shape[1])[g]
        _, tb = _cr_last(X, fit + w * res, C, True)
        count += abs(tb) >= abs(t_obs) - 1e-12
    return dict(t_obs=float(t_obs), p=(1 + count) / (n_boot + 1), floor=1 / (n_boot + 1))


# ── Main ──────────────────────────────────────────────────────────────────────

def analyse(d, args):
    print(f"\n{'=' * 100}\n{d['app']} / {d['outcome']}   n={len(d['y'])}   pool={d['z'].shape[1]}")
    keys, parts, info = build_rule(d)
    print("  levels:", pd.Series([i["level"] for i in info]).value_counts().to_dict())
    print("  cluster L_j:", {k: (int(len(np.unique(v))) if v is not None else "HC1")
                            for k, v in parts.items()},
          pd.Series(keys).value_counts().to_dict())
    untestable = [i["name"] for i in info if not i["testable"]]
    print(f"  support gate (>= {MIN_SUPPORT} clusters per cell): {len(untestable)} removed, "
          f"published ones removed: {[n for n in untestable if n in d['published']]}")
    pool = [j for j, i in enumerate(info) if i["testable"]]
    fe = block_fe(d["levels"][d["block"]]) if d["block"] else None
    rule = LevelAwareTest(keys, parts, fe=fe)
    y, t, z = d["y"], d["t"], d["z"]

    # sanity: LevelAwareTest reproduces the core test under one clustering, no FE
    m_all = z.shape[1]
    for kw in [dict(cluster=d["levels"][d["community"]]), dict(hc1=True)]:
        p1 = plain_test(m=m_all, **kw)(y, t, z, [0, 1], range(m_all))
        p2 = conditional_interaction_pvalues(y, t, z, S=[0, 1], **kw)
        assert np.allclose(np.log(p1), np.log(p2), atol=1e-6), "level-aware test != core test"

    runs = {}
    runs["1"] = run(d, "1", d["published_test"], PUBLISHED_CFG)
    if sorted(runs["1"]["selected"]) != sorted(d["published"]):
        raise SystemExit(f"published set NOT reproduced: {runs['1']['selected']} vs {d['published']}")
    runs["2"] = run(d, "2", rule, PUBLISHED_CFG, pool, info)
    runs["3"] = run(d, "3", rule, NEW_DEFAULT, pool, info)
    if fe is not None:
        runs["3b"] = run(d, "3b", LevelAwareTest(keys, parts, fe=None), NEW_DEFAULT, pool, info)
    runs["3c"] = run(d, "3c", rule, NEW_DEFAULT, None, info)
    a = d["levels"][d["assign"]]
    runs["4a"] = run(d, "4a", plain_test(hc1=True, m=m_all) if d["assign"] == d["order"][-1]
                     else plain_test(cluster=a, m=m_all), NEW_DEFAULT)
    runs["4b"] = run(d, "4b", plain_test(cluster=d["levels"][d["community"]], m=m_all), NEW_DEFAULT)

    # Uganda: T is the realised grant (Wobs); the lottery assigned `assigned`, and 11% of
    # assigned groups were never funded.  Re-run the rule on the intent-to-treat contrast.
    d_itt = None
    if "assigned" in d:
        d_itt = dict(d, t=d["assigned"])
        k_i, p_i, info_i = build_rule(d_itt)
        pool_i = [j for j, i in enumerate(info_i) if i["testable"]]
        runs["3i"] = run(d_itt, "3i", LevelAwareTest(k_i, p_i, fe=fe), NEW_DEFAULT, pool_i, info_i)

    # the published discoveries under the rule, conditional on the rest of the published set
    idx = {n: j for j, n in enumerate(d["names"])}
    pub = [idx[n] for n in d["published"]]
    m_rule = len(pool)
    pubrows = []
    for j in pub:
        S = [k for k in pub if k != j]
        pubrows.append(dict(name=d["names"][j], level=info[j]["level"], cluster=info[j]["cluster"],
                            G=info[j]["G"], support=info[j]["support"], testable=info[j]["testable"],
                            p_rule_cond_published=float(rule(y, t, z, S, [j])[j]),
                            p_rule_marginal=float(rule(y, t, z, [], [j])[j]),
                            gate_rule=ALPHA / m_rule))

    # post-hoc design-based checks on every final set
    posthoc = []
    todo = {}
    for rk in [k for k in ["1", "2", "3", "3i"] if k in runs]:
        sel = [idx[n] for n in runs[rk]["selected"]]
        for j in sel:
            todo[(j, tuple(sorted(k for k in sel if k != j)))] = rk
    t0 = time.time()
    for (j, S), rk in todo.items():
        S = list(S)
        if d["app"] == "uganda":
            r = randomization_test(d, S, j, args.n_perm)
            ri = randomization_test(d_itt, S, j, args.n_perm)
            r.update(p_itt=ri["p"], t_obs_itt=ri["t_obs"])
            kind = "RI: group lottery within district, district FE, t clustered by group"
        else:
            r = wild_cluster_bootstrap(d, S, j, d["levels"]["community"], args.n_boot)
            kind = "restricted wild cluster bootstrap-t, clustered by community"
        posthoc.append(dict(name=d["names"][j], given=[d["names"][k] for k in S], from_run=rk,
                            test=kind, **r))
        itt = f"  ITT p={r['p_itt']:.1e}" if "p_itt" in r else ""
        print(f"  post-hoc {d['names'][j]:14s} | {[d['names'][k] for k in S]}  p={r['p']:.1e}{itt} "
              f"(floor {r['floor']:.0e})  [{kind.split(':')[0]}]")
    print(f"  post-hoc checks: {time.time() - t0:.0f}s")
    return dict(n=len(y), pool=int(z.shape[1]), m_rule=m_rule,
                partitions={k: (int(len(np.unique(v))) if v is not None else "HC1")
                            for k, v in parts.items()},
                candidates=info, untestable=untestable, published=d["published"],
                published_under_rule=pubrows, runs=runs, posthoc=posthoc,
                published_test=d["published_label"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=9999)
    ap.add_argument("--n-boot", type=int, default=9999)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"alpha": ALPHA, "rho": RHO, "min_support": MIN_SUPPORT}
    for o in ["skilled_employed", "log_biz_assets"]:
        report[f"uganda/{o}"] = analyse(uganda(o), args)
    report["ghana/consumption"] = analyse(ghana(), args)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nSaved → {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
