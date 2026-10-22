"""Conditional interaction tests: p-values for H0(j | S) over candidates j.

The CATE test inside NEXIS asks whether candidate Z_j modifies the treatment effect
given the already selected set Z_S.  Three families:

  conditional_interaction_pvalues        linear T x Z_j t-test (homoskedastic OLS,
                                         HC1 or CR1S cluster-robust SEs)
  conditional_interaction_pvalues_gcm    GCM-hybrid test (cross-fitted nuisance)
  conditional_interaction_pvalues_pcm    modified Projected Covariance Measure

All return a length-m vector of p-values (ones off `candidates`), or (pvalues, tstats)
with return_tstats=True; any function with that signature can be passed to nexis() as
`pvalue_fn`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import numpy as np
from scipy import stats


# ── Linear interaction test ───────────────────────────────────────────────────

def _residualize_against(D: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Residualize columns of V against the column space of D using QR.
    D: (n, q), V: (n, k) or (n,)
    """
    V2 = np.asarray(V, dtype=float)
    vec = (V2.ndim == 1)
    if vec:
        V2 = V2[:, None]

    # If D is empty, return V
    if D.size == 0 or D.shape[1] == 0:
        return V if not vec else V2[:, 0]

    # Reduced QR; works well when q is small
    Q, _ = np.linalg.qr(D, mode="reduced")
    R = V2 - Q @ (Q.T @ V2)
    return R[:, 0] if vec else R


def conditional_interaction_pvalues(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    S: Optional[Sequence[int]] = None,
    candidates: Optional[Sequence[int]] = None,
    return_tstats: bool = False,
    cluster: Optional[np.ndarray] = None,
    hc1: bool = False,
):
    """
    Vectorized p-values for H0(j|S) over j in candidates.
    Working model:
      Y = beta0 + betaT T + beta_S' Z_S + beta_j Z_j + gamma_S'(T*Z_S) + gamma_j (T*Z_j) + e
    Tests gamma_j = 0 for each j via FWL residualization against D=[1, T, Z_S, T*Z_S],
    then 2-regressor OLS per candidate on [Z_j, T*Z_j].

    cluster: array of group labels (length n) for CR1S cluster-robust SEs.
             Takes priority over hc1.  Even when Z_j is constant within clusters,
             after FWL residualization both Z_tilde_j and (T*Z_j)_tilde have
             within-cluster variation (T varies within communities), so CRVE is
             valid and well-defined.  Implementation: loop over G clusters —
             O(n×K) total work, same cost as HC1.  df = G-1 for the t-test.

    hc1:    if True and cluster is None, use HC1 (White sandwich, n/(n-k)
            correction) instead of homoskedastic OLS variance.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    Z = np.asarray(z, dtype=float)
    n, m = Z.shape
    if y.shape[0] != n or t.shape[0] != n:
        raise ValueError("Shape mismatch among y, t, z")

    S = [] if S is None else sorted(set(int(k) for k in S))
    if candidates is None:
        cand = np.array([j for j in range(m) if j not in S], dtype=int)
    else:
        cand = np.array([int(j) for j in candidates if int(j) not in S], dtype=int)

    pvals = np.ones(m, dtype=float)
    all_tstats = np.zeros(m, dtype=float)
    if cand.size == 0:
        return (pvals, all_tstats) if return_tstats else pvals

    # Nuisance design D = [1, T, Z_S, T*Z_S]
    D_cols = [np.ones(n), t]
    for k in S:
        D_cols.append(Z[:, k])
    for k in S:
        D_cols.append(t * Z[:, k])
    D = np.column_stack(D_cols) if len(D_cols) > 0 else np.empty((n, 0), dtype=float)

    y_tilde = _residualize_against(D, y)  # (n,)
    yy = np.sum(y_tilde ** 2)

    Z_c = Z[:, cand]
    TZ_c = t[:, None] * Z_c
    Z_tilde = _residualize_against(D, Z_c)
    X_tilde = _residualize_against(D, TZ_c)

    zz = np.sum(Z_tilde * Z_tilde, axis=0)
    xx = np.sum(X_tilde * X_tilde, axis=0)
    zx = np.sum(Z_tilde * X_tilde, axis=0)
    zy = np.sum(Z_tilde * y_tilde[:, None], axis=0)
    xy = np.sum(X_tilde * y_tilde[:, None], axis=0)

    det = zz * xx - zx * zx
    valid = det > 1e-12

    p_full = D.shape[1] + 2
    dof = n - p_full

    if dof > 0:
        beta_x = np.zeros_like(det)
        beta_z = np.zeros_like(det)
        beta_x[valid] = (zz[valid] * xy[valid] - zx[valid] * zy[valid]) / det[valid]
        beta_z[valid] = (xx[valid] * zy[valid] - zx[valid] * xy[valid]) / det[valid]

        e_mat = y_tilde[:, None] - Z_tilde * beta_z[None, :] - X_tilde * beta_x[None, :]

        if cluster is not None:
            # CR1S cluster-robust sandwich variance.
            # For each cluster g: Ze_g = sum_{i in g} Z_tilde[i,j]*e[i,j]  (K,)
            #                     Xe_g = sum_{i in g} X_tilde[i,j]*e[i,j]  (K,)
            # meat diagonal blocks: zzee = sum_g Ze_g², xxee = sum_g Xe_g², zxee = sum_g Ze_g*Xe_g
            groups   = np.asarray(cluster)
            unique_g = np.unique(groups)
            G        = len(unique_g)
            zzee = np.zeros(len(cand), dtype=float)
            xxee = np.zeros(len(cand), dtype=float)
            zxee = np.zeros(len(cand), dtype=float)
            for g in unique_g:
                mask = groups == g
                Ze_g  = (Z_tilde[mask] * e_mat[mask]).sum(axis=0)
                Xe_g  = (X_tilde[mask] * e_mat[mask]).sum(axis=0)
                zzee += Ze_g ** 2
                xxee += Xe_g ** 2
                zxee += Ze_g * Xe_g
            # CR1S small-sample correction
            cr1s  = (G / (G - 1)) * ((n - 1) / (n - p_full))
            zzee *= cr1s;  xxee *= cr1s;  zxee *= cr1s
            var_bx = np.full_like(det, np.nan, dtype=float)
            var_bx[valid] = (
                (zz[valid] ** 2 * xxee[valid]
                 - 2 * zz[valid] * zx[valid] * zxee[valid]
                 + zx[valid] ** 2 * zzee[valid])
                / det[valid] ** 2
            )
            t_df = G - 1
        elif hc1:
            # HC1 sandwich variance (fully vectorized).
            e2   = e_mat ** 2
            zzee = np.sum(Z_tilde ** 2 * e2, axis=0)
            xxee = np.sum(X_tilde ** 2 * e2, axis=0)
            zxee = np.sum(Z_tilde * X_tilde * e2, axis=0)
            # var(beta_x) = (n/(n-p_full)) * (zz²·xxee − 2·zz·zx·zxee + zx²·zzee) / det²
            scale = n / dof
            var_bx = np.full_like(det, np.nan, dtype=float)
            var_bx[valid] = (
                scale * (zz[valid] ** 2 * xxee[valid]
                         - 2 * zz[valid] * zx[valid] * zxee[valid]
                         + zx[valid] ** 2 * zzee[valid])
                / det[valid] ** 2
            )
            t_df = dof
        else:
            rss = np.full_like(det, np.nan, dtype=float)
            rss[valid] = yy - beta_z[valid] * zy[valid] - beta_x[valid] * xy[valid]
            rss = np.maximum(rss, 0.0)
            sigma2 = np.full_like(det, np.nan, dtype=float)
            sigma2[valid] = rss[valid] / dof
            var_bx = np.full_like(det, np.nan, dtype=float)
            var_bx[valid] = sigma2[valid] * (zz[valid] / det[valid])
            t_df = dof

        ok = valid & np.isfinite(var_bx) & (var_bx > 0)
        tstat = np.zeros_like(det, dtype=float)
        tstat[ok] = beta_x[ok] / np.sqrt(var_bx[ok])

        p = np.ones_like(det, dtype=float)
        p[ok] = 2.0 * stats.t.sf(np.abs(tstat[ok]), df=t_df)
        p = np.clip(np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)
        pvals[cand] = p
        all_tstats[cand] = tstat

    return (pvals, all_tstats) if return_tstats else pvals


# ── GCM helpers ───────────────────────────────────────────────────────────────

def _make_nuisance_model(nuisance: str, n_estimators: int, max_depth: Optional[int],
                         random_state: int):
    """Return a fitted-model factory for the chosen nuisance estimator.

    nuisance options:
      "poly2"  — Ridge on degree-2 polynomial features of Z^S.  ~5ms per fit,
                 handles quadratic main effects.  ~1.2× overhead vs linear.
      "lgbm"   — LightGBM shallow trees.  ~35ms per fit, fully nonparametric.
                 ~3× overhead vs linear.  Requires lightgbm package.
      "rf"     — Random Forest (sklearn).  ~1s per fit, fully nonparametric.
                 ~27× overhead vs linear.  Most robust, use for final results.
    """
    if nuisance == "poly2":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        return lambda: Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=1.0)),
        ])
    elif nuisance == "lgbm":
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is required for nuisance='lgbm'. "
                              "Install with: pip install lightgbm")
        _n = n_estimators if n_estimators != 100 else 50
        _d = max_depth if max_depth is not None else 4
        return lambda: lgb.LGBMRegressor(
            n_estimators=_n, max_depth=_d, num_leaves=2**_d - 1,
            verbose=-1, n_jobs=1, random_state=random_state,
        )
    elif nuisance == "rf":
        from sklearn.ensemble import RandomForestRegressor
        _d = max_depth  # None = unlimited
        return lambda: RandomForestRegressor(
            n_estimators=n_estimators, max_depth=_d,
            n_jobs=1, random_state=random_state,
        )
    else:
        raise ValueError(f"nuisance must be 'poly2', 'lgbm', or 'rf'; got '{nuisance}'")


def _crossfit(
    X: np.ndarray,
    y: np.ndarray,
    model_factory,
    n_splits: int = 5,
    random_state: int = 0,
    splits=None,
) -> np.ndarray:
    """K-fold cross-fitted predictions from any sklearn-compatible model.

    Pass pre-computed ``splits`` (from ``_make_splits``) to reuse the same fold
    assignment across multiple calls on the same X, ensuring consistency and
    avoiding redundant KFold construction.
    """
    if splits is None:
        splits = _make_splits(X, n_splits=n_splits, random_state=random_state)
    pred = np.zeros_like(y, dtype=float)
    for tr, te in splits:
        m = model_factory()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def _make_splits(X: np.ndarray, n_splits: int, random_state: int = 0):
    """Pre-compute KFold split indices to reuse across multiple cross-fit calls."""
    from sklearn.model_selection import KFold
    return list(KFold(n_splits=n_splits, shuffle=True,
                      random_state=random_state).split(X))


def conditional_interaction_pvalues_gcm(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    S: Optional[Sequence[int]] = None,
    candidates: Optional[Sequence[int]] = None,
    nuisance: str = "poly2",
    n_splits: int = 5,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    return_tstats: bool = False,
) -> np.ndarray:
    """GCM-hybrid p-values for H0(j|S) over j in candidates.

    R-learner pseudo-outcome φ̂ = (Y − m̂(Z^S))(T−e)/(e(1−e)) is computed via
    cross-fitted nuisance regression (choice controlled by `nuisance`), then a
    GCM z-statistic is formed using vectorized linear residualization of all Z^j
    candidates (O(n×m), fast regardless of nuisance choice).

    nuisance:
      "poly2"  ~1.2× slower than linear — Ridge on poly(2) features of Z^S.
               Handles quadratic main-effect nonlinearity; best default.
      "lgbm"   ~3× slower — LightGBM shallow trees; fully nonparametric.
      "rf"     ~27× slower — Random Forest; most robust for final results.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    Z = np.asarray(z, dtype=float)
    n, m = Z.shape
    S_list = [] if S is None else sorted(set(int(k) for k in S))
    if candidates is None:
        cand = np.array([j for j in range(m) if j not in S_list], dtype=int)
    else:
        cand = np.array([int(j) for j in candidates if int(j) not in S_list], dtype=int)

    pvals = np.ones(m, dtype=float)
    if cand.size == 0:
        return pvals

    e = float(t.mean())
    if abs(e * (1 - e)) < 1e-12:
        return pvals

    model_fn = _make_nuisance_model(nuisance, n_estimators, max_depth, random_state=0)

    if S_list:
        # Conditioning input for nuisance model: Z^S columns
        Z_S_fit = Z[:, S_list]
        # Pre-compute splits once; reuse for both cross-fit passes to ensure identical folds
        splits = _make_splits(Z_S_fit, n_splits=n_splits)
        m_hat = _crossfit(Z_S_fit, y, model_fn, splits=splits)
        phi = (y - m_hat) * (t - e) / (e * (1 - e))
        phi_resid = phi - _crossfit(Z_S_fit, phi, model_fn, splits=splits)
    else:
        # Fast path: no conditioning — m_hat = mean(Y), phi_resid = phi - mean(phi)
        m_hat = np.full(n, y.mean())
        phi = (y - m_hat) * (t - e) / (e * (1 - e))
        phi_resid = phi - phi.mean()

    # Residualize all Z^j candidates on [1, Z^S] linearly (vectorized, O(n×m))
    D_lin = np.column_stack([np.ones(n)] + ([Z[:, S_list]] if S_list else []))
    Z_cand_resid = _residualize_against(D_lin, Z[:, cand])  # (n, K)

    # GCM z-statistic: sqrt(n) * mean(R) / std(R),  R_i = phi_resid_i * Z^j_resid_i
    R = phi_resid[:, None] * Z_cand_resid
    R_mean = R.mean(axis=0)
    R_std  = R.std(axis=0, ddof=1)

    valid = R_std > 1e-12
    Tn = np.zeros(len(cand))
    Tn[valid] = np.sqrt(n) * R_mean[valid] / R_std[valid]

    p = 2.0 * stats.norm.sf(np.abs(Tn))
    p = np.clip(np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)
    pvals[cand] = p
    if return_tstats:
        all_tstats = np.zeros(m, dtype=float)
        all_tstats[cand] = Tn
        return pvals, all_tstats
    return pvals


# ── PCM helpers ───────────────────────────────────────────────────────────────
#
# Projected Covariance Measure (Lundborg, Kim, Shah & Samworth, Ann. Statist. 2024),
# modified for the CATE-equivalence null of Equation (8).
#
# PCM tests conditional MEAN independence, H0: E[V | X, Z] = E[V | Z], by
#   (i)  learning, on one half of the sample, a scalar projection f̂(X, Z) that
#        approximates the conditional-mean contrast E[V|X,Z] − E[V|Z], and
#   (ii) forming a GCM-style residual-product statistic on the other half using
#        f̂ in place of the raw X.
# Because f̂ is estimated on data independent of the half it is evaluated on, the
# statistic is asymptotically N(0,1) under H0 no matter how bad f̂ is, while under
# the alternative the covariance is positive by construction — hence a ONE-SIDED
# test that is consistent against essentially any conditional-mean alternative,
# including those with vanishing conditional covariance that the GCM cannot see.
#
# The modification needed here is that the response of interest, τ = Y(1) − Y(0),
# is never observed.  Under randomisation with known e = P(T=1), the R-learner
# pseudo-outcome  φ = (Y − m(Z^S))(T − e)/(e(1−e))  satisfies E[φ | Z] = E[τ | Z]
# for ANY function m of Z (the choice of m affects variance only), so
#
#     H0(j | S):  E[τ | Z^{S∪{j}}] = E[τ | Z^S]   ⟺   E[φ | Z^{S∪{j}}] = E[φ | Z^S],
#
# which is exactly PCM's null with V = φ, X = Z^j and Z = Z^S.

def _pcm_basis(x: np.ndarray, order: int) -> List[np.ndarray]:
    """Polynomial basis {x, x², …} applied column-wise to an (n, c) block."""
    return [x ** k for k in range(1, order + 1)]


def _pcm_fit_projection(
    Zc_tr: np.ndarray,        # (n_tr, c) candidate columns on the training half
    phi_tr_resid: np.ndarray, # (n_tr,) φ residualised linearly against [1, Z^S]
    Q_tr: np.ndarray,         # (n_tr, d) orthonormal basis of [1, Z^S] on the training half
    order: int,
):
    """Vectorised per-candidate projection fit.

    For every candidate column of ``Zc_tr`` independently, regress the training-half
    pseudo-outcome on the polynomial basis of that column after partialling out
    [1, Z^S], and return the coefficients (normalised so the fitted function has
    unit variance), the standardisation constants needed to re-evaluate the basis
    on the held-out half, and the explained sum of squares (used for screening).

    Unit-norming is what keeps the projection non-degenerate: under H0 the fitted
    coefficients are pure noise and shrink to zero, which would make both the mean
    and the variance of the residual product vanish and the studentised statistic
    ill-defined.  Rescaling to unit variance fixes the *direction* without touching
    the (train-half-measurable, hence null-irrelevant) magnitude.
    """
    n_tr, c = Zc_tr.shape
    B = _pcm_basis(Zc_tr, order)                       # order × (n_tr, c)

    mu = np.empty((order, c)); sd = np.empty((order, c))
    Bt: List[np.ndarray] = []
    for k, b in enumerate(B):
        mu[k] = b.mean(axis=0)
        s = b.std(axis=0)
        s[s < 1e-12] = 1.0
        sd[k] = s
        bs = (b - mu[k]) / s
        Bt.append(bs - Q_tr @ (Q_tr.T @ bs))           # partial out [1, Z^S]

    # Per-candidate normal equations: (order × order) Gram + rhs, all vectorised.
    G = np.empty((order, order, c))
    for a in range(order):
        for b_ in range(a, order):
            G[a, b_] = G[b_, a] = (Bt[a] * Bt[b_]).sum(axis=0)
    r = np.stack([(Bt[a] * phi_tr_resid[:, None]).sum(axis=0) for a in range(order)])

    lam = 1e-6 * np.einsum("aac->c", G) / order + 1e-12
    Gm = np.moveaxis(G, 2, 0).copy()                   # (c, order, order)
    idx = np.arange(order)
    Gm[:, idx, idx] += lam[:, None]

    beta = np.zeros((order, c))
    try:
        sol = np.linalg.solve(Gm, np.moveaxis(r, 0, 1)[..., None])[:, :, 0]  # (c, order)
        beta = np.moveaxis(sol, 0, 1)
    except np.linalg.LinAlgError:                      # fall back candidate-by-candidate
        for jj in range(c):
            try:
                beta[:, jj] = np.linalg.solve(Gm[jj], r[:, jj])
            except np.linalg.LinAlgError:
                beta[:, jj] = 0.0
    beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)

    fit_tr = sum(beta[a] * Bt[a] for a in range(order))            # (n_tr, c)
    scale = np.sqrt((fit_tr ** 2).mean(axis=0))
    ess = (beta * r).sum(axis=0)                                   # explained SS (screening)
    ok = scale > 1e-10
    beta = np.where(ok, beta / np.where(ok, scale, 1.0), 0.0)
    return beta, mu, sd, ess, ok


def _pcm_eval_projection(Zc_te, beta, mu, sd, order: int) -> np.ndarray:
    """Evaluate the fitted projection on the held-out half → (n_te, c).

    Uses the training-half standardisation, so the evaluated function is exactly
    the one that was fitted.
    """
    out = np.zeros_like(Zc_te, dtype=float)
    for k, b in enumerate(_pcm_basis(Zc_te, order)):
        out += beta[k] * ((b - mu[k]) / sd[k])
    return out


def _studentise(R: np.ndarray) -> np.ndarray:
    """√n · mean(R)/sd(R) per column; 0 where the column is degenerate.

    Deliberately plain.  The PCM is one-sided and NEXIS reads its p-values at the
    Bonferroni gate α/m ≈ 5×10⁻⁶, so far-tail accuracy is what matters, and a
    one-term Cornish–Fisher correction T + γ̂(2T²+1)/(6√n) was tried and *worsened*
    it (rejections at 10⁻³ over 8×9216 null tests at n=5000: 100 uncorrected vs 141
    corrected, against 74 expected).  The excess is driven by a handful of
    high-leverage observations rather than by third-order skewness, so the Edgeworth
    expansion does not apply and its estimated γ̂ only adds noise.  The residual
    finite-sample tail excess is reported as a documented limitation instead.
    """
    n = R.shape[0]
    mu = R.mean(axis=0)
    sd = R.std(axis=0, ddof=1)
    good = sd > 1e-12
    T = np.zeros(R.shape[1])
    T[good] = np.sqrt(n) * mu[good] / sd[good]
    return T


def conditional_interaction_pvalues_pcm(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    S: Optional[Sequence[int]] = None,
    candidates: Optional[Sequence[int]] = None,
    nuisance: str = "poly2",
    n_splits: int = 3,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    order: int = 2,
    projection: str = "poly",       # "poly" | "lgbm"
    screen_top: int = 32,           # lgbm only: candidates given an ML projection
    combine: str = "bonferroni",    # "bonferroni" (2·min) | "single" | "crossfit"
    random_state: int = 0,
    chunk: int = 1024,
    return_tstats: bool = False,
) -> np.ndarray:
    """Modified-PCM p-values for H0(j|S) over j in candidates.

    Sample-splitting scheme (both halves used, hence no data is wasted):

      1. φ = (Y − m̂(Z^S))(T − e)/(e(1−e)) with m̂ cross-fitted on the full sample.
      2. Split [n] into halves A, B.  For each direction (train, test) ∈ {(A,B),(B,A)}:
         a. on `train`, fit a projection f̂_j for every candidate j — a polynomial in
            Z^j (vectorised over all candidates at once) and, for the `screen_top`
            candidates with the largest training-half explained sum of squares, a
            LightGBM fit of E[φ | Z^j, Z^S] when projection="lgbm";
         b. on `test`, form R_i = (φ_i − Ê[φ|Z^S_i])·(f̂_j(Z_i) − Ê[f̂_j|Z^S_i]) with
            Ê[φ|Z^S] cross-fitted within the test half.
      3. combine="bonferroni" (default) studentises each direction separately and
         reports 2·min(p₁, p₂).  combine="single" keeps only the (A→B) direction.
         combine="crossfit" pools the residual products from both directions and
         studentises over all n — DO NOT USE: unlike in DML, the projection does not
         converge under H0 (it is normalised noise), so the two half-blocks stay
         strongly dependent and the pooled variance is understated.  Measured size at
         α=0.05 on a null design (a calibration check since removed from the tree, at
         git tag pre-cleanup-2026-09): 0.090 (crossfit) vs 0.048 (bonferroni) and 0.045
         (single).  Bonferroni also dominates single on power, so it is the default.

    The test is ONE-SIDED (large positive statistic ⇒ evidence against H0): under the
    alternative the projection is aligned with the conditional-mean contrast by
    construction, so the population covariance is positive.

    projection:
      "poly"  — vectorised polynomial projection of degree `order` for every
                candidate.  Cost is O(n·m·order), comparable to the GCM.
      "lgbm"  — additionally refits the `screen_top` most promising candidates with
                gradient-boosted trees on (Z^j, Z^S) and residualises them against
                Z^S nonparametrically.  Which candidates get the ML projection is
                decided on the training half only, so held-out validity is retained.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    Z = np.asarray(z, dtype=float)
    n, m = Z.shape
    S_list = [] if S is None else sorted(set(int(k) for k in S))
    if candidates is None:
        cand = np.array([j for j in range(m) if j not in S_list], dtype=int)
    else:
        cand = np.array([int(j) for j in candidates if int(j) not in S_list], dtype=int)

    pvals = np.ones(m, dtype=float)
    all_t = np.zeros(m, dtype=float)
    if cand.size == 0 or n < 8:
        return (pvals, all_t) if return_tstats else pvals

    e = float(t.mean())
    if abs(e * (1 - e)) < 1e-12:
        return (pvals, all_t) if return_tstats else pvals

    model_fn = _make_nuisance_model(nuisance, n_estimators, max_depth, random_state=0)
    Z_S = Z[:, S_list] if S_list else np.zeros((n, 0))

    # ── Step 1: half-split ──────────────────────────────────────────────────────
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    A, B = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    if len(A) < 8 or len(B) < 8:
        return (pvals, all_t) if return_tstats else pvals

    # ── Step 2: doubly-robust pseudo-outcome, cross-fitted WITHIN each half ─────
    # m̂ only affects variance, never bias, but it must not couple the two halves:
    # a projection fitted on φ_train would otherwise see held-out outcomes through
    # m̂ and break the independence the held-out statistic relies on.  Leaving it
    # full-sample cross-fitted measures 0.095 size at α=0.05 for the lgbm
    # projection; splitting it restores nominal size.
    phi = np.empty(n)
    for half in (A, B):
        if S_list:
            k_h = max(2, min(n_splits, len(half) // 4))
            m_h = _crossfit(Z_S[half], y[half], model_fn,
                            splits=_make_splits(Z_S[half], n_splits=k_h))
        else:
            m_h = np.full(len(half), y[half].mean())
        phi[half] = (y[half] - m_h) * (t[half] - e) / (e * (1 - e))

    def _prep(tr, te):
        """Direction-level quantities that do not depend on the candidate."""
        D_tr = np.column_stack([np.ones(len(tr))] + ([Z_S[tr]] if S_list else []))
        Q_tr, _ = np.linalg.qr(D_tr, mode="reduced")
        phi_tr = phi[tr] - Q_tr @ (Q_tr.T @ phi[tr])
        D_te = np.column_stack([np.ones(len(te))] + ([Z_S[te]] if S_list else []))
        Q_te, _ = np.linalg.qr(D_te, mode="reduced")
        if S_list:
            k_te = max(2, min(n_splits, len(te) // 4))
            phi_te = phi[te] - _crossfit(Z_S[te], phi[te], model_fn,
                                         splits=_make_splits(Z_S[te], n_splits=k_te))
        else:
            phi_te = phi[te] - phi[te].mean()
        return Q_tr, phi_tr, Q_te, phi_te

    directions = [(A, B)] if combine == "single" else [(A, B), (B, A)]
    prepped = [_prep(tr, te) for tr, te in directions]

    # ── Step 3: projections, residual products, statistic ───────────────────────
    R_pool = np.zeros((n, cand.size)) if combine == "crossfit" else None
    T_dir = np.full((len(directions), cand.size), np.nan)

    for d, ((tr, te), (Q_tr, phi_tr, Q_te, phi_te)) in enumerate(zip(directions, prepped)):
        beta_all = np.zeros((order, cand.size))
        mu_all   = np.zeros((order, cand.size))
        sd_all   = np.ones((order, cand.size))
        ess_all  = np.zeros(cand.size)
        ok_all   = np.zeros(cand.size, dtype=bool)

        for a in range(0, cand.size, chunk):
            b_ = min(a + chunk, cand.size)
            beta, mu, sd, ess, ok = _pcm_fit_projection(
                Z[np.ix_(tr, cand[a:b_])], phi_tr, Q_tr, order)
            beta_all[:, a:b_] = beta
            mu_all[:, a:b_]   = mu
            sd_all[:, a:b_]   = sd
            ess_all[a:b_]     = ess
            ok_all[a:b_]      = ok

        # ML projection for the training-half top candidates (selection is
        # train-measurable, so the held-out statistic stays valid).
        ml_cols: Dict[int, np.ndarray] = {}
        if projection == "lgbm" and screen_top > 0:
            k_top = int(min(screen_top, cand.size))
            top = np.argsort(-ess_all)[:k_top]
            top = [int(i) for i in top if ok_all[i]]
            if top:
                ml_fn = _make_nuisance_model("lgbm", n_estimators, max_depth,
                                             random_state=random_state)
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
                    # v̂ = Ê[f̂ | Z^S] estimated nonparametrically within the test half
                    if S_list:
                        k_te = max(2, min(n_splits, len(te) // 4))
                        g = g - _crossfit(Z_S[te], g, ml_fn,
                                          splits=_make_splits(Z_S[te], n_splits=k_te))
                    else:
                        g = g - g.mean()
                    ml_cols[i] = g

        for a in range(0, cand.size, chunk):
            b_ = min(a + chunk, cand.size)
            g = _pcm_eval_projection(Z[np.ix_(te, cand[a:b_])],
                                     beta_all[:, a:b_], mu_all[:, a:b_],
                                     sd_all[:, a:b_], order)
            g = g - Q_te @ (Q_te.T @ g)                # linear v̂ for the poly projection
            for i, col in ml_cols.items():
                if a <= i < b_:
                    g[:, i - a] = col
            R = phi_te[:, None] * g
            R[:, ~ok_all[a:b_]] = 0.0
            if combine == "crossfit":
                R_pool[np.ix_(te, np.arange(a, b_))] = R
            else:
                T_dir[d, a:b_] = _studentise(R)

    if combine == "crossfit":
        Tn = _studentise(R_pool)
        p = stats.norm.sf(Tn)                          # one-sided
    else:
        p_dir = stats.norm.sf(np.nan_to_num(T_dir, nan=-np.inf))
        p = np.clip(len(directions) * p_dir.min(axis=0), 0.0, 1.0)
        Tn = np.nanmax(T_dir, axis=0)

    p = np.clip(np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)
    pvals[cand] = p
    if return_tstats:
        all_t[cand] = Tn
        return pvals, all_t
    return pvals
