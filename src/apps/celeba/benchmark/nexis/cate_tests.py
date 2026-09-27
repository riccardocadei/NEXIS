"""CATE-equivalence tests p_j(S) for H0(j | S): E[tau | Z^{S u {j}}] = E[tau | Z^S].

Every test has the same interface:

    pvalues(y, t, z, S, candidates, return_tstats=False) -> p  or  (p, T)

where ``p`` and ``T`` are length-m vectors (p = 1 and T = 0 off ``candidates``).
NEXIS only consumes these p-values and statistics, so any valid test plugs in.

Implemented (appendix "CATE-equivalence test" of the paper):

* ``linear_pvalues``  linear treatment-interaction t-test (default);
* ``gcm_pvalues``     doubly-robust Generalised Covariance Measure on the R-learner
                      pseudo-outcome, with quadratic (Ridge on degree-2 polynomial
                      features) or LightGBM nuisances;
* ``pcm_pvalues``     Projected Covariance Measure (Lundborg et al., 2024) on the same
                      pseudo-outcome, with a quadratic or LightGBM projection; the two
                      sample-split directions are combined by min{1, 2 min(p1, p2)}.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

# LightGBM settings of the "lgbm" nuisance and projection models.
LGBM_PARAMS = dict(n_estimators=50, max_depth=4, num_leaves=2 ** 4 - 1, verbose=-1, n_jobs=1)


# ── shared helpers ────────────────────────────────────────────────────────────

def _candidates(m: int, S: Sequence[int], candidates: Optional[Sequence[int]]) -> np.ndarray:
    if candidates is None:
        return np.array([j for j in range(m) if j not in S], dtype=int)
    return np.array([int(j) for j in candidates if int(j) not in S], dtype=int)


def _residualize_against(D: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Residuals of the columns of V, (n, k) or (n,), on the column space of D, (n, q)."""
    Vm = np.asarray(V, dtype=float)
    vec = Vm.ndim == 1
    if vec:
        Vm = Vm[:, None]
    if D.size == 0 or D.shape[1] == 0:
        return V if not vec else Vm[:, 0]
    Q, _ = np.linalg.qr(D, mode="reduced")
    R = Vm - Q @ (Q.T @ Vm)
    return R[:, 0] if vec else R


def _nuisance_model(kind: str, random_state: int = 0):
    """Factory for the cross-fitted regressions: "poly2" (quadratic) or "lgbm"."""
    if kind == "poly2":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures
        return lambda: Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=1.0)),
        ])
    if kind == "lgbm":
        import lightgbm as lgb
        return lambda: lgb.LGBMRegressor(**LGBM_PARAMS, random_state=random_state)
    raise ValueError(f"nuisance must be 'poly2' or 'lgbm'; got {kind!r}")


def _make_splits(X: np.ndarray, n_splits: int, random_state: int = 0):
    from sklearn.model_selection import KFold
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X))


def _crossfit(X: np.ndarray, y: np.ndarray, model_factory, splits) -> np.ndarray:
    """K-fold cross-fitted predictions of y from X."""
    pred = np.zeros_like(y, dtype=float)
    for tr, te in splits:
        mdl = model_factory()
        mdl.fit(X[tr], y[tr])
        pred[te] = mdl.predict(X[te])
    return pred


def _studentise(R: np.ndarray) -> np.ndarray:
    """sqrt(n) mean(R) / sd(R) per column; 0 where the column is degenerate."""
    n = R.shape[0]
    mu = R.mean(axis=0)
    sd = R.std(axis=0, ddof=1)
    good = sd > 1e-12
    T = np.zeros(R.shape[1])
    T[good] = np.sqrt(n) * mu[good] / sd[good]
    return T


# ── (i) linear treatment-interaction test ─────────────────────────────────────

def linear_pvalues(y, t, z, S: Optional[Sequence[int]] = None,
                   candidates: Optional[Sequence[int]] = None, return_tstats: bool = False):
    """t-test of delta_j = 0 in the working model

        Y = a + T b_T + Z^S g_S + T Z^S d_S + Z^j b_j + T Z^j delta_j + e,

    vectorised over candidates by Frisch-Waugh-Lovell: every column is residualised on
    D = [1, T, Z^S, T Z^S], then a two-regressor OLS on [Z^j, T Z^j] is solved in
    closed form per candidate (homoskedastic standard errors, n - |D| - 2 dof).
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    Z = np.asarray(z, dtype=float)
    n, m = Z.shape
    S = [] if S is None else sorted(set(int(k) for k in S))
    cand = _candidates(m, S, candidates)

    pvals = np.ones(m, dtype=float)
    all_t = np.zeros(m, dtype=float)
    if cand.size == 0:
        return (pvals, all_t) if return_tstats else pvals

    D = np.column_stack([np.ones(n), t] + [Z[:, k] for k in S] + [t * Z[:, k] for k in S])
    y_tilde = _residualize_against(D, y)
    yy = np.sum(y_tilde ** 2)

    Z_c = Z[:, cand]
    Z_tilde = _residualize_against(D, Z_c)
    X_tilde = _residualize_against(D, t[:, None] * Z_c)

    zz = np.sum(Z_tilde * Z_tilde, axis=0)
    xx = np.sum(X_tilde * X_tilde, axis=0)
    zx = np.sum(Z_tilde * X_tilde, axis=0)
    zy = np.sum(Z_tilde * y_tilde[:, None], axis=0)
    xy = np.sum(X_tilde * y_tilde[:, None], axis=0)

    det = zz * xx - zx * zx
    valid = det > 1e-12
    dof = n - (D.shape[1] + 2)
    if dof > 0:
        beta_x = np.zeros_like(det)
        beta_z = np.zeros_like(det)
        beta_x[valid] = (zz[valid] * xy[valid] - zx[valid] * zy[valid]) / det[valid]
        beta_z[valid] = (xx[valid] * zy[valid] - zx[valid] * xy[valid]) / det[valid]

        rss = np.full_like(det, np.nan, dtype=float)
        rss[valid] = yy - beta_z[valid] * zy[valid] - beta_x[valid] * xy[valid]
        rss = np.maximum(rss, 0.0)
        sigma2 = np.full_like(det, np.nan, dtype=float)
        sigma2[valid] = rss[valid] / dof
        var_bx = np.full_like(det, np.nan, dtype=float)
        var_bx[valid] = sigma2[valid] * (zz[valid] / det[valid])

        ok = valid & np.isfinite(var_bx) & (var_bx > 0)
        tstat = np.zeros_like(det, dtype=float)
        tstat[ok] = beta_x[ok] / np.sqrt(var_bx[ok])
        p = np.ones_like(det, dtype=float)
        p[ok] = 2.0 * stats.t.sf(np.abs(tstat[ok]), df=dof)
        p = np.clip(np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)
        pvals[cand] = p
        all_t[cand] = tstat

    return (pvals, all_t) if return_tstats else pvals


# ── (ii) doubly-robust GCM ────────────────────────────────────────────────────

def gcm_pvalues(y, t, z, S: Optional[Sequence[int]] = None,
                candidates: Optional[Sequence[int]] = None, nuisance: str = "poly2",
                n_splits: int = 3, return_tstats: bool = False):
    """GCM test on the R-learner pseudo-outcome phi = (Y - m(Z^S))(T - e) / (e(1-e)).

    m = E[Y | Z^S] and E[phi | Z^S] are cross-fitted (K = n_splits folds) with the
    chosen nuisance model; Z^j is residualised on [1, Z^S] linearly for all candidates
    at once, and T_n = sqrt(n) mean(R) / sd(R) with R = phi_resid * Z^j_resid (two-sided).
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    Z = np.asarray(z, dtype=float)
    n, m = Z.shape
    S_list = [] if S is None else sorted(set(int(k) for k in S))
    cand = _candidates(m, S_list, candidates)

    pvals = np.ones(m, dtype=float)
    if cand.size == 0:
        return (pvals, np.zeros(m)) if return_tstats else pvals
    e = float(t.mean())
    if abs(e * (1 - e)) < 1e-12:
        return (pvals, np.zeros(m)) if return_tstats else pvals

    model_fn = _nuisance_model(nuisance, random_state=0)
    if S_list:
        Z_S = Z[:, S_list]
        splits = _make_splits(Z_S, n_splits=n_splits)
        m_hat = _crossfit(Z_S, y, model_fn, splits)
        phi = (y - m_hat) * (t - e) / (e * (1 - e))
        phi_resid = phi - _crossfit(Z_S, phi, model_fn, splits)
    else:
        m_hat = np.full(n, y.mean())
        phi = (y - m_hat) * (t - e) / (e * (1 - e))
        phi_resid = phi - phi.mean()

    D_lin = np.column_stack([np.ones(n)] + ([Z[:, S_list]] if S_list else []))
    Z_cand_resid = _residualize_against(D_lin, Z[:, cand])

    R = phi_resid[:, None] * Z_cand_resid
    R_mean = R.mean(axis=0)
    R_std = R.std(axis=0, ddof=1)
    valid = R_std > 1e-12
    Tn = np.zeros(len(cand))
    Tn[valid] = np.sqrt(n) * R_mean[valid] / R_std[valid]

    p = 2.0 * stats.norm.sf(np.abs(Tn))
    p = np.clip(np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)
    pvals[cand] = p
    if return_tstats:
        all_t = np.zeros(m, dtype=float)
        all_t[cand] = Tn
        return pvals, all_t
    return pvals


# ── (iii) projected covariance measure ────────────────────────────────────────

def _pcm_basis(x: np.ndarray, order: int) -> List[np.ndarray]:
    return [x ** k for k in range(1, order + 1)]


def _pcm_fit_projection(Zc_tr, phi_tr_resid, Q_tr, order: int):
    """Per-candidate polynomial projection, vectorised over candidates.

    Regresses the training-half pseudo-outcome (residualised on [1, Z^S]) on the
    standardised basis (Z^j, (Z^j)^2) partialled out of [1, Z^S], then rescales the
    fitted function to unit variance.  Returns the coefficients, the standardisation
    constants, the explained sum of squares (used to screen candidates for the
    LightGBM projection) and a validity mask.
    """
    n_tr, c = Zc_tr.shape
    B = _pcm_basis(Zc_tr, order)
    mu = np.empty((order, c))
    sd = np.empty((order, c))
    Bt: List[np.ndarray] = []
    for k, b in enumerate(B):
        mu[k] = b.mean(axis=0)
        s = b.std(axis=0)
        s[s < 1e-12] = 1.0
        sd[k] = s
        bs = (b - mu[k]) / s
        Bt.append(bs - Q_tr @ (Q_tr.T @ bs))

    G = np.empty((order, order, c))
    for a in range(order):
        for b_ in range(a, order):
            G[a, b_] = G[b_, a] = (Bt[a] * Bt[b_]).sum(axis=0)
    r = np.stack([(Bt[a] * phi_tr_resid[:, None]).sum(axis=0) for a in range(order)])

    lam = 1e-6 * np.einsum("aac->c", G) / order + 1e-12
    Gm = np.moveaxis(G, 2, 0).copy()
    idx = np.arange(order)
    Gm[:, idx, idx] += lam[:, None]

    beta = np.zeros((order, c))
    try:
        sol = np.linalg.solve(Gm, np.moveaxis(r, 0, 1)[..., None])[:, :, 0]
        beta = np.moveaxis(sol, 0, 1)
    except np.linalg.LinAlgError:
        for jj in range(c):
            try:
                beta[:, jj] = np.linalg.solve(Gm[jj], r[:, jj])
            except np.linalg.LinAlgError:
                beta[:, jj] = 0.0
    beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)

    fit_tr = sum(beta[a] * Bt[a] for a in range(order))
    scale = np.sqrt((fit_tr ** 2).mean(axis=0))
    ess = (beta * r).sum(axis=0)
    ok = scale > 1e-10
    beta = np.where(ok, beta / np.where(ok, scale, 1.0), 0.0)
    return beta, mu, sd, ess, ok


def _pcm_eval_projection(Zc_te, beta, mu, sd, order: int) -> np.ndarray:
    out = np.zeros_like(Zc_te, dtype=float)
    for k, b in enumerate(_pcm_basis(Zc_te, order)):
        out += beta[k] * ((b - mu[k]) / sd[k])
    return out


def pcm_pvalues(y, t, z, S: Optional[Sequence[int]] = None,
                candidates: Optional[Sequence[int]] = None, projection: str = "poly",
                n_splits: int = 3, order: int = 2, screen_top: int = 32,
                random_state: int = 0, chunk: int = 1024, return_tstats: bool = False):
    """One-sided PCM test of H0(j | S) with the pseudo-outcome phi as response.

    1. The sample is split into halves A and B; within each half, m = E[Y | Z^S] is
       cross-fitted (quadratic nuisance) and phi = (Y - m)(T - e) / (e(1-e)).
    2. For each direction (train, test) in {(A, B), (B, A)}: on ``train`` a projection
       f_j of the conditional-mean contrast is fitted for every candidate (polynomial
       of degree ``order`` in Z^j; with projection="lgbm", the ``screen_top`` candidates
       with the largest training-half explained sum of squares are refitted by LightGBM
       on (Z^j, Z^S)); on ``test`` the residual product
       R = (phi - E[phi | Z^S]) (f_j - E[f_j | Z^S]) is studentised.
    3. The two one-sided p-values are combined as min{1, 2 min(p1, p2)}.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    Z = np.asarray(z, dtype=float)
    n, m = Z.shape
    S_list = [] if S is None else sorted(set(int(k) for k in S))
    cand = _candidates(m, S_list, candidates)

    pvals = np.ones(m, dtype=float)
    all_t = np.zeros(m, dtype=float)
    if cand.size == 0 or n < 8:
        return (pvals, all_t) if return_tstats else pvals
    e = float(t.mean())
    if abs(e * (1 - e)) < 1e-12:
        return (pvals, all_t) if return_tstats else pvals

    model_fn = _nuisance_model("poly2", random_state=0)
    Z_S = Z[:, S_list] if S_list else np.zeros((n, 0))

    # half-split
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    A, B = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    if len(A) < 8 or len(B) < 8:
        return (pvals, all_t) if return_tstats else pvals

    # pseudo-outcome, with m cross-fitted within each half so the halves stay independent
    phi = np.empty(n)
    for half in (A, B):
        if S_list:
            k_h = max(2, min(n_splits, len(half) // 4))
            m_h = _crossfit(Z_S[half], y[half], model_fn,
                            _make_splits(Z_S[half], n_splits=k_h))
        else:
            m_h = np.full(len(half), y[half].mean())
        phi[half] = (y[half] - m_h) * (t[half] - e) / (e * (1 - e))

    def _prep(tr, te):
        D_tr = np.column_stack([np.ones(len(tr))] + ([Z_S[tr]] if S_list else []))
        Q_tr, _ = np.linalg.qr(D_tr, mode="reduced")
        phi_tr = phi[tr] - Q_tr @ (Q_tr.T @ phi[tr])
        D_te = np.column_stack([np.ones(len(te))] + ([Z_S[te]] if S_list else []))
        Q_te, _ = np.linalg.qr(D_te, mode="reduced")
        if S_list:
            k_te = max(2, min(n_splits, len(te) // 4))
            phi_te = phi[te] - _crossfit(Z_S[te], phi[te], model_fn,
                                         _make_splits(Z_S[te], n_splits=k_te))
        else:
            phi_te = phi[te] - phi[te].mean()
        return Q_tr, phi_tr, Q_te, phi_te

    directions = [(A, B), (B, A)]
    prepped = [_prep(tr, te) for tr, te in directions]
    T_dir = np.full((len(directions), cand.size), np.nan)

    for d, ((tr, te), (Q_tr, phi_tr, Q_te, phi_te)) in enumerate(zip(directions, prepped)):
        beta_all = np.zeros((order, cand.size))
        mu_all = np.zeros((order, cand.size))
        sd_all = np.ones((order, cand.size))
        ess_all = np.zeros(cand.size)
        ok_all = np.zeros(cand.size, dtype=bool)
        for a in range(0, cand.size, chunk):
            b_ = min(a + chunk, cand.size)
            beta, mu, sd, ess, ok = _pcm_fit_projection(
                Z[np.ix_(tr, cand[a:b_])], phi_tr, Q_tr, order)
            beta_all[:, a:b_] = beta
            mu_all[:, a:b_] = mu
            sd_all[:, a:b_] = sd
            ess_all[a:b_] = ess
            ok_all[a:b_] = ok

        # LightGBM projection for the training-half top candidates (the screening is
        # measurable with respect to the training half, so the held-out test stays valid)
        ml_cols: Dict[int, np.ndarray] = {}
        if projection == "lgbm" and screen_top > 0:
            k_top = int(min(screen_top, cand.size))
            top = [int(i) for i in np.argsort(-ess_all)[:k_top] if ok_all[int(i)]]
            if top:
                ml_fn = _nuisance_model("lgbm", random_state=random_state)
                for i in top:
                    j = int(cand[i])
                    Xtr = np.column_stack([Z[tr, j]] + ([Z_S[tr]] if S_list else []))
                    Xte = np.column_stack([Z[te, j]] + ([Z_S[te]] if S_list else []))
                    mdl = ml_fn()
                    mdl.fit(Xtr, phi[tr])
                    g = np.asarray(mdl.predict(Xte), dtype=float)
                    if g.std() < 1e-12:
                        continue
                    g = g / g.std()
                    if S_list:
                        k_te = max(2, min(n_splits, len(te) // 4))
                        g = g - _crossfit(Z_S[te], g, ml_fn,
                                          _make_splits(Z_S[te], n_splits=k_te))
                    else:
                        g = g - g.mean()
                    ml_cols[i] = g

        for a in range(0, cand.size, chunk):
            b_ = min(a + chunk, cand.size)
            g = _pcm_eval_projection(Z[np.ix_(te, cand[a:b_])], beta_all[:, a:b_],
                                     mu_all[:, a:b_], sd_all[:, a:b_], order)
            g = g - Q_te @ (Q_te.T @ g)
            for i, col in ml_cols.items():
                if a <= i < b_:
                    g[:, i - a] = col
            R = phi_te[:, None] * g
            R[:, ~ok_all[a:b_]] = 0.0
            T_dir[d, a:b_] = _studentise(R)

    p_dir = stats.norm.sf(np.nan_to_num(T_dir, nan=-np.inf))
    p = np.clip(len(directions) * p_dir.min(axis=0), 0.0, 1.0)
    Tn = np.nanmax(T_dir, axis=0)
    p = np.clip(np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)
    pvals[cand] = p
    if return_tstats:
        all_t[cand] = Tn
        return pvals, all_t
    return pvals


# ── registry ──────────────────────────────────────────────────────────────────

TESTS = ("linear", "GCM: quadratic", "GCM: lgbm", "PCM: quadratic", "PCM: lgbm")


def make_test(name: str, n_splits: int = 3):
    """Return pvalues(y, t, z, S, candidates, return_tstats) for a test name in TESTS."""
    if name == "linear":
        return linear_pvalues
    if name in ("GCM: quadratic", "GCM: lgbm"):
        nuisance = "poly2" if name.endswith("quadratic") else "lgbm"

        def gcm(y, t, z, S, candidates, return_tstats=False):
            return gcm_pvalues(y, t, z, S, candidates, nuisance=nuisance,
                               n_splits=n_splits, return_tstats=return_tstats)
        return gcm
    if name in ("PCM: quadratic", "PCM: lgbm"):
        projection = "poly" if name.endswith("quadratic") else "lgbm"

        def pcm(y, t, z, S, candidates, return_tstats=False):
            return pcm_pvalues(y, t, z, S, candidates, projection=projection,
                               n_splits=n_splits, return_tstats=return_tstats)
        return pcm
    raise ValueError(f"test must be one of {TESTS}; got {name!r}")
