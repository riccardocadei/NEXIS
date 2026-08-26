
from __future__ import annotations

from dataclasses import dataclass, field
from math import lgamma, log10
from typing import Callable, List, Optional, Sequence, Dict, Tuple, Union
import numpy as np
from scipy import stats


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
         α=0.05 on the null design of `check_pcm_calibration.py`: 0.090 (crossfit) vs
         0.048 (bonferroni) and 0.045 (single).  Bonferroni also dominates single on
         power, so it is the default.

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


@dataclass
class SelectionResult:
    selected: List[int]            # indices into the feature space nexis ran on
    pvalues: np.ndarray            # one entry per feature in that space
    method: str
    alpha: float
    metadata: Dict[str, float]
    feature_names: List[str] = field(default_factory=list)  # w_{name} / z_{j} labels


# ── Pathwise Bonferroni backward gate ─────────────────────────────────────────
#
# The backward sweep tests H0(j | S\{j}) for j in S.  Correcting only over the
# |S| coordinates on the realised path ignores that S itself was chosen by the
# data.  The family of backward hypotheses reachable on ANY data-dependent path
# partitions by selected-set size s = |S|: at depth s there are
#
#     N_s = m * C(m-1, s-1) = s * C(m, s)
#
# distinct (coordinate, conditioning-set) pairs.  Spending alpha across depths
# with deterministic weights w_s >= 0, sum_s w_s <= 1, and Bonferroni within each
# depth gives the gate
#
#     g_s = alpha * w_s / N_s
#
# so the total Type-I budget over every reachable backward hypothesis is <= alpha.
# Default w_s = 1/(s(s+1)) sums to 1 over all positive integers, so no maximum
# search depth has to be pre-specified.

def default_alpha_spending(s: int) -> float:
    """w_s = 1 / (s (s+1));  sum_{s>=1} w_s = 1."""
    return 1.0 / (s * (s + 1))


AlphaSpending = Union[Callable[[int], float], Sequence[float], None]


def _log_comb(n: int, k: int) -> float:
    """Natural log of C(n, k)."""
    if k < 0 or k > n:
        return -np.inf
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def log10_n_backward_hypotheses(m: int, s: int) -> float:
    """log10 N_s = log10( m * C(m-1, s-1) )."""
    if s < 1 or m < 1:
        return -np.inf
    return (np.log(m) + _log_comb(m - 1, s - 1)) / np.log(10)


def resolve_alpha_spending(spending: AlphaSpending,
                           max_depth: Optional[int] = None) -> Callable[[int], float]:
    """Validate and normalise an alpha-spending specification.

    spending may be None (use the default 1/(s(s+1))), a callable s -> w_s, or a
    sequence [w_1, w_2, ...].  Weights must be non-negative and their total mass
    over the supported depths must not exceed 1 (checked exactly for sequences,
    and over 1..max_depth for callables when max_depth is given).
    """
    if spending is None:
        return default_alpha_spending

    if callable(spending):
        fn = spending
        if max_depth is not None:
            ws = [float(fn(s)) for s in range(1, int(max_depth) + 1)]
            if any(w < 0 for w in ws):
                raise ValueError("alpha_spending weights must be non-negative")
            if sum(ws) > 1.0 + 1e-9:
                raise ValueError(
                    f"alpha_spending mass over depths 1..{max_depth} is "
                    f"{sum(ws):.6f} > 1")

        def _checked(s: int) -> float:
            w = float(fn(s))
            if w < 0:
                raise ValueError(f"alpha_spending weight w_{s}={w} is negative")
            return w

        return _checked

    ws = [float(w) for w in spending]
    if any(w < 0 for w in ws):
        raise ValueError("alpha_spending weights must be non-negative")
    total = sum(ws)
    if total > 1.0 + 1e-9:
        raise ValueError(f"alpha_spending weights sum to {total:.6f} > 1")

    def _from_seq(s: int) -> float:
        return ws[s - 1] if 1 <= s <= len(ws) else 0.0

    return _from_seq


def pathwise_backward_gate(alpha: float, m: int, s: int,
                           weight_fn: Callable[[int], float]) -> Tuple[float, float, float]:
    """Return (g_s, log10 g_s, log10 N_s) for the pathwise backward gate at depth s.

    g_s underflows to 0.0 in float64 for very large s; log10 g_s is exact, so
    comparisons are made in log space (see _passes_log10).
    """
    w_s = weight_fn(s)
    log10_ns = log10_n_backward_hypotheses(m, s)
    if w_s <= 0:
        return 0.0, -np.inf, log10_ns
    log10_g = log10(alpha) + log10(w_s) - log10_ns
    g = 10.0 ** log10_g if log10_g > -300 else 0.0
    return g, log10_g, log10_ns


def _passes_log10(p: float, log10_thr: float) -> bool:
    """p <= 10**log10_thr, evaluated in log space so tiny gates don't underflow."""
    if p <= 0.0:
        return True
    if not np.isfinite(log10_thr):
        return False
    return log10(p) <= log10_thr


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


def interaction_test_pvalue(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    j: int,
    S: Optional[Sequence[int]] = None,
) -> float:
    pvals = conditional_interaction_pvalues(y=y, t=t, z=z, S=S, candidates=[j])
    return float(pvals[int(j)])


def marginal_interaction_pvalues(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    cluster: Optional[np.ndarray] = None,
) -> np.ndarray:
    return conditional_interaction_pvalues(y=y, t=t, z=z, S=[], cluster=cluster)


def marginal_select(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    alpha: float = 0.05,
    adjust: Optional[str] = None,  # None | "FWER" | "FDR"
    groups: Optional[Dict[str, List[int]]] = None,
    cluster: Optional[np.ndarray] = None,
) -> SelectionResult:
    """Marginal interaction test with optional multiple-testing adjustment.

    adjust=None  : raw threshold at level alpha (no correction).
    adjust="FWER": Bonferroni correction (α/M).  groups splits the budget
                   per group instead of globally.
    adjust="FDR" : Benjamini-Hochberg step-up procedure at level alpha.
    """
    pvals = conditional_interaction_pvalues(y=y, t=t, z=z, S=[], cluster=cluster)
    m = len(pvals)
    _adj = adjust.upper() if adjust is not None else None

    if _adj is None:
        selected = np.where(pvals <= alpha)[0].tolist()
        method = "marginal_raw"
        metadata: Dict[str, float] = {"threshold": float(alpha), "m": float(m)}
    elif _adj == "FWER":
        if groups is not None:
            # Per-group Bonferroni: each group corrects for its own size only.
            mask = np.zeros(m, dtype=bool)
            for gname, gidxs in groups.items():
                thr_g = alpha / max(len(gidxs), 1)
                for j in gidxs:
                    if pvals[j] <= thr_g:
                        mask[j] = True
            selected = np.where(mask)[0].tolist()
            method = "marginal_fwer_grouped"
            metadata = {"m": float(m), **{
                f"thr_{g}": alpha / max(len(idxs), 1)
                for g, idxs in groups.items()
            }}
        else:
            thr = alpha / max(m, 1)
            selected = np.where(pvals <= thr)[0].tolist()
            method = "marginal_fwer"
            metadata = {"threshold": float(thr), "m": float(m)}
    elif _adj == "FDR":
        order = np.argsort(pvals)
        thresholds = (np.arange(1, m + 1) / m) * alpha
        below = pvals[order] <= thresholds
        if below.any():
            kstar = int(np.where(below)[0].max())
            selected = order[:kstar + 1].tolist()
        else:
            selected = []
        method = "marginal_fdr"
        metadata = {"alpha": float(alpha), "m": float(m)}
    else:
        raise ValueError("adjust must be None, 'FWER', or 'FDR'")

    return SelectionResult(
        selected=selected,
        pvalues=pvals,
        method=method,
        alpha=alpha,
        metadata=metadata,
    )


# ── NEXIS ──────────────────────────────────────────────────────────────────────

def nexis(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    w: Optional[np.ndarray] = None,
    w_names: Optional[List[str]] = None,
    z_names: Optional[List[str]] = None,
    alpha: float = 0.05,
    max_rounds: Optional[int] = 20,
    rho: Optional[float] = 0.5,
    backward: bool = True,
    adjust: Optional[str] = "FWER",  # None | "FWER" (Bonferroni) | "FDR" (BH)
    test: str = "linear",            # "linear" | "quadratic" | "GCM" | "PCM" variants
    nuisance: str = "poly2",         # gcm/pcm only: "poly2" | "lgbm" | "rf"
    n_splits: int = 5,             # gcm/pcm only
    n_estimators: int = 100,       # gcm/pcm only
    max_depth: Optional[int] = None,  # gcm/pcm only
    pcm_projection: str = "poly",    # pcm only: "poly" | "lgbm"
    pcm_order: int = 2,              # pcm only: polynomial-projection degree
    pcm_screen_top: int = 32,        # pcm only: candidates refitted with the ML projection
    pcm_combine: str = "crossfit",   # pcm only: "crossfit" | "bonferroni"
    cluster: Optional[np.ndarray] = None,  # CR1S cluster-robust SEs for Z-phase (linear test)
    hc1: bool = False,                     # HC1 robust SEs for W-phase and Z-phase fallback
    backward_gate: str = "standard",       # "standard" (alpha/s) | "pathwise" (g_s)
    alpha_spending: AlphaSpending = None,  # pathwise only; default w_s = 1/(s(s+1))
    pvalue_fn=None,                        # custom conditional test (see below)
    verbose: bool = False,
) -> SelectionResult:
    """Forward(-backward) selection (NEXIS — Neural Exposure Interaction Search).

    Each round:
      1. Forward: among candidates passing Gate 1, pick the best.
         Gate 1 depends on adjust:
           None    : p ≤ α
           "FWER"  : p ≤ α/|remaining|  (Bonferroni, default)
           "FDR"   : BH applied to all remaining p-values at level α
         Spectral gap (ρ): stop if the chosen candidate's
         |t| < rho * min(|t| of already-selected features).
      2. Backward (skipped when backward=False): remove j ∈ S if it no longer
         passes the gate given S\\{j}.
         None/"FWER" remove sequentially; "FDR" batches all backward p-values
         with the current S and applies BH before removing.
    Repeats until S is unchanged (fixed point).

    backward=False runs a pure greedy forward pass — useful for ablation.

    w: optional (n, q) matrix of interpretable covariates.  When provided, a
      preliminary phase runs NEXIS on W first; the features selected there seed
      the initial S for the main phase on Z.  Both W and Z features compete
      symmetrically in that phase: forward can re-add expelled W features,
      backward can expel W features.  SelectionResult.feature_names labels
      each column as w_{name} (using w_names if given, else column index) or
      z_{j}.

    test:
      "linear"  — parametric interaction t-test; fast, assumes linear effects.
                  Supports controls / main_controls / interaction_only.
      "gcm"     — GCM-hybrid test (nonparametric φ̂, linear Z^j residualisation).
                  Speed: ~1.2× (poly2), ~3× (lgbm), ~27× (rf) vs linear.
                  controls / main_controls / interaction_only are ignored.
      "PCM: quadratic" / "PCM: lgbm"
                — modified Projected Covariance Measure (Lundborg, Kim, Shah &
                  Samworth, 2024): a one-sided conditional-MEAN-independence test
                  built on a half-sample-learned projection of the conditional-mean
                  contrast.  Unlike the GCM it stays consistent against alternatives
                  whose conditional covariance with Z^j vanishes (U-shapes, symmetric
                  thresholds), at the cost of splitting the sample.  Tuned via
                  pcm_projection / pcm_order / pcm_screen_top / pcm_combine.

    backward_gate:
      "standard" — the published rule: remove j if p(j|S\{j}) > α/|S| (or > α when
                   adjust=None).  Corrects only over coordinates on the realised path.
      "pathwise" — Pathwise Bonferroni: remove j if p(j|S\{j}) > g_s, where
                   g_s = α·w_s / (m·C(m-1, s-1)), s = |S| and m is the number of
                   candidate coordinates entering NEXIS.  Corrects simultaneously over
                   every backward hypothesis reachable on ANY data-dependent path, so
                   the total Type-I budget over that family is ≤ α.  Only the backward
                   threshold changes; the forward gate is untouched and no separate
                   certification stage is added.

    alpha_spending (pathwise only):
      Deterministic depth weights w_s ≥ 0 with Σ_s w_s ≤ 1, fixed before running.
      None (default) uses w_s = 1/(s(s+1)), which sums to 1 over all positive integers
      and so needs no pre-specified maximum depth.  May also be a callable s → w_s or
      a sequence [w_1, w_2, …].  Validated for non-negativity and total mass ≤ 1.

    pvalue_fn:
      Optional drop-in replacement for the conditional interaction test.  Called as
      pvalue_fn(y=, t=, z=, S=, candidates=, return_tstats=) and must return the same
      shapes as conditional_interaction_pvalues: a length-m vector of p-values (ones
      off `candidates`), or (pvalues, tstats) when return_tstats=True.  Overrides
      `test`, `cluster` and `hc1`.  NEXIS only ever consumes p-values and t-statistics
      from the test, so any valid test plugs in here — e.g. a level-aware clustered
      test on a candidate pool that spans several levels of nesting
      (src/apps/uganda/multilevel_inference.py).

    rho (ρ):
      Relative stopping threshold in (0, 1].  At each forward step the new
      candidate is admitted only if |t_new| ≥ ρ × min_{j∈S}|t_j|.
      Equivalently, ρ = 1/K where K is the maximum plausible ratio between
      the strongest and weakest true direct-modifier CATE contrasts.
      Recommended range: 0.2 (effects may vary 5×) to 0.5 (effects within 2×).
    """
    # Normalise test aliases and set nuisance accordingly.
    # "quadratic" → gcm + poly2 nuisance
    # "GCM"       → gcm + lgbm nuisance
    # "linear"    → linear (nuisance unused)
    # The canonical lowercase keys "gcm" and "pcm" are pass-through: they keep the
    # caller's `nuisance` / `pcm_projection` rather than re-deriving them.  Only the
    # user-facing aliases set those.  This matters because the W phase recurses with
    # the already-normalised `test`, so a remapping alias would silently reset the
    # variant mid-run (e.g. "GCM: quadratic" turning into lgbm inside the W phase).
    _test_key = test.lower().strip()
    if _test_key in {"gcm: quadratic", "quadratic"}:
        test, nuisance = "gcm", "poly2"
    elif _test_key in {"gcm: lgbm", "lgbm"}:
        test, nuisance = "gcm", "lgbm"
    elif _test_key == "gcm":
        test = "gcm"
    elif _test_key == "pcm":
        test = "pcm"
    elif _test_key in {"pcm: quadratic", "pcm: poly"}:
        test, nuisance, pcm_projection = "pcm", "poly2", "poly"
    elif _test_key == "pcm: lgbm":
        # LightGBM *projection*, but poly2 nuisances: the PCM half-split leaves n/2
        # for Ê[φ|Z^S] and Ê[f̂|Z^S], and fully nonparametric nuisances at that size
        # are not calibrated (measured size 0.080 at α=0.05 on the check design vs
        # 0.049 with poly2, at identical power).  Pass nuisance="lgbm" explicitly
        # to override.
        test, nuisance, pcm_projection = "pcm", "poly2", "lgbm"
    elif _test_key != "linear":
        raise ValueError("test must be 'linear', 'GCM: quadratic', 'GCM: lgbm', "
                         "'PCM: quadratic', or 'PCM: lgbm'")

    # rho=0 is treated as rho=None (gate 2 disabled).
    if rho is not None and rho == 0:
        rho = None

    # Backward gate selection.  "standard" keeps the published alpha/s rule;
    # "pathwise" corrects over every backward hypothesis reachable on any path.
    _bwd = str(backward_gate).lower().strip()
    if _bwd not in {"standard", "pathwise"}:
        raise ValueError("backward_gate must be 'standard' or 'pathwise'")
    _weight_fn = (resolve_alpha_spending(alpha_spending, max_depth=max_rounds)
                  if _bwd == "pathwise" else None)
    backward_log: List[Dict[str, float]] = []

    y_arr = np.asarray(y, dtype=float).reshape(-1)
    t_arr = np.asarray(t, dtype=float).reshape(-1)
    Z = np.asarray(z, dtype=float)
    n, m = Z.shape

    if w is not None and w_names is None:
        raise ValueError("w_names is required when w is provided")
    if w is not None and pvalue_fn is not None:
        # pvalue_fn is defined against the column layout of Z; the W phase runs on a
        # different matrix, so silently reusing it there would mislabel candidates.
        raise ValueError("pvalue_fn is not supported together with w; pass the W "
                         "columns inside z instead (the w_candidates=True layout)")

    # ── Phase 1: W selection ──────────────────────────────────────────────────
    S_w: List[int] = []
    k = 0
    if w is not None:
        W = np.asarray(w, dtype=float)
        if W.ndim == 1:
            W = W[:, None]
        result_w = nexis(
            y=y_arr, t=t_arr, z=W, alpha=alpha, max_rounds=max_rounds,
            test=test, nuisance=nuisance, n_splits=n_splits,
            n_estimators=n_estimators, max_depth=max_depth,
            pcm_projection=pcm_projection, pcm_order=pcm_order,
            pcm_screen_top=pcm_screen_top, pcm_combine=pcm_combine,
            rho=rho, adjust=adjust, cluster=None, hc1=hc1,
            backward=backward, verbose=verbose,
        )
        S_w = result_w.selected
        k = len(S_w)
        if k > 0:
            # Prepend selected W columns to Z; forward step will only add Z columns
            Z = np.hstack([W[:, S_w], Z])
            if verbose:
                print(f"  [W phase] selected {k} features: {S_w}", flush=True)

    total = Z.shape[1]  # k + m (or m when k=0)

    gcm_kwargs: dict = (dict(nuisance=nuisance, n_splits=n_splits,
                             n_estimators=n_estimators, max_depth=max_depth)
                        if test in ("gcm", "pcm") else {})
    pcm_kwargs: dict = (dict(order=pcm_order, projection=pcm_projection,
                             screen_top=pcm_screen_top, combine=pcm_combine)
                        if test == "pcm" else {})

    def _pvalues(S_cur, candidates, return_tstats=False):
        if pvalue_fn is not None:
            return pvalue_fn(y=y_arr, t=t_arr, z=Z, S=S_cur,
                             candidates=candidates,
                             return_tstats=return_tstats)
        if test == "pcm":
            return conditional_interaction_pvalues_pcm(
                y=y_arr, t=t_arr, z=Z, S=S_cur, candidates=candidates,
                return_tstats=return_tstats, **gcm_kwargs, **pcm_kwargs,
            )
        if test == "gcm":
            return conditional_interaction_pvalues_gcm(
                y=y_arr, t=t_arr, z=Z, S=S_cur, candidates=candidates,
                return_tstats=return_tstats, **gcm_kwargs,
            )
        return conditional_interaction_pvalues(
            y=y_arr, t=t_arr, z=Z, S=S_cur, candidates=candidates,
            return_tstats=return_tstats, cluster=cluster, hc1=hc1,
        )

    # W features (0..k-1) seed S; all features compete symmetrically from here
    selected: List[int] = list(range(k))
    last_pvals = np.ones(total, dtype=float)
    selected_pvals = np.ones(total, dtype=float)
    t_selected: List[float] = []

    need_tstats = rho is not None

    S_prev: List[int] = [-1]  # sentinel
    round_num = 0
    _gate2_stop = False
    n_rejections = 0

    while selected != S_prev and not _gate2_stop:
        if max_rounds is not None and round_num >= max_rounds:
            break
        S_prev = list(selected)

        # ── Forward step ──────────────────────────────────────────────────────
        remaining = [j for j in range(total) if j not in selected]
        if remaining:
            if need_tstats:
                pvals, tstats = _pvalues(selected, remaining, return_tstats=True)
            else:
                pvals = _pvalues(selected, remaining)
                tstats = None
            last_pvals = pvals.copy()

            # Gate 1: significance filter
            _adj = adjust.upper() if adjust is not None else None
            if _adj == "FDR":
                pv_rem = np.array([pvals[j] for j in remaining])
                order_rem = np.argsort(pv_rem)
                m_rem = len(remaining)
                bh_thr = (np.arange(1, m_rem + 1) / m_rem) * alpha
                below_bh = pv_rem[order_rem] <= bh_thr
                if below_bh.any():
                    kstar_rem = int(np.where(below_bh)[0].max())
                    eligible = [remaining[order_rem[i]] for i in range(kstar_rem + 1)]
                else:
                    eligible = []
                gate_fwd = float("nan")  # no single threshold for verbose
            else:
                gate_fwd = alpha if _adj is None else alpha / len(remaining)
                eligible = [j for j in remaining if pvals[j] <= gate_fwd]

            if eligible:
                if tstats is not None:
                    j_star = max(eligible, key=lambda j: abs(tstats[j]))
                else:
                    j_star = min(eligible, key=lambda j: pvals[j])
                p_star = float(pvals[j_star])

                if verbose:
                    t_str = (f" |t|={abs(tstats[j_star]):.2f}" if tstats is not None else "")
                    gate_str = "FDR" if _adj == "FDR" else f"{gate_fwd:.2e}"
                    print(f"  round {round_num+1:2d} fwd | remaining={len(remaining):5d} "
                          f"gate={gate_str} eligible={len(eligible)} "
                          f"| best=j{j_star} p={p_star:.2e}{t_str}", flush=True)

                # Gate 2: relative stopping (rho)
                if rho is not None and tstats is not None and len(t_selected) > 0:
                    t_new = float(abs(tstats[j_star]))
                    t_min_found = min(t_selected)
                    if t_min_found > 0 and t_new < rho * t_min_found:
                        if verbose:
                            print(f"    → auto-stopped (Gate 2): "
                                  f"|t_new|={t_new:.2f} < f={rho} × "
                                  f"t_min={t_min_found:.2f}", flush=True)
                        _gate2_stop = True

                if not _gate2_stop:
                    selected_pvals[j_star] = p_star
                    selected.append(j_star)
                    if tstats is not None:
                        t_selected.append(float(abs(tstats[j_star])))
                    if verbose:
                        print(f"    → added j{j_star}  S={selected}", flush=True)
            elif verbose:
                print(f"  round {round_num+1:2d} fwd | no eligible candidate "
                      f"(gate={gate_fwd:.2e})", flush=True)

        if _gate2_stop:
            break

        # ── Backward step ─────────────────────────────────────────────────────
        if not backward:
            round_num += 1
            continue

        if _adj == "FDR":
            # Batch: compute all backward p-values with the current S, then BH.
            js = list(selected)
            back_pv = [float(_pvalues([s for s in selected if s != j], [j])[j]) for j in js]
            pv_arr = np.array(back_pv)
            order_back = np.argsort(pv_arr)
            m_sel = len(js)
            bh_thr = (np.arange(1, m_sel + 1) / m_sel) * alpha
            below = pv_arr[order_back] <= bh_thr
            keep = (
                {js[order_back[i]] for i in range(int(np.where(below)[0].max()) + 1)}
                if below.any() else set()
            )
            for idx, j in enumerate(js):
                if j not in keep:
                    selected.remove(j)
                    n_rejections += 1
                    if verbose:
                        print(f"  round {round_num+1:2d} bwd | removed j{j} "
                              f"(BH) p={back_pv[idx]:.2e}  S={selected}", flush=True)
        else:
            for j in list(selected):
                if not selected:
                    break
                if j not in selected:
                    continue
                S_minus_j = [s for s in selected if s != j]
                pvals_back = _pvalues(S_minus_j, [j])
                p_j = float(pvals_back[j])
                s_cur = len(selected)

                if _bwd == "pathwise":
                    # g_s = alpha * w_s / (m * C(m-1, s-1)); m = candidates entering NEXIS
                    gate_bwd, log10_gate, log10_ns = pathwise_backward_gate(
                        alpha, total, s_cur, _weight_fn)
                    depth_budget = alpha * _weight_fn(s_cur)
                    retained = _passes_log10(p_j, log10_gate)
                else:
                    gate_bwd = alpha if _adj is None else alpha / s_cur
                    log10_gate = log10(gate_bwd) if gate_bwd > 0 else -np.inf
                    log10_ns = log10(s_cur) if s_cur > 0 else -np.inf
                    depth_budget = alpha
                    retained = p_j <= gate_bwd

                backward_log.append({
                    "round": float(round_num + 1), "j": float(j), "s": float(s_cur),
                    "log10_N_s": float(log10_ns), "depth_budget": float(depth_budget),
                    "gate": float(gate_bwd), "log10_gate": float(log10_gate),
                    "p": p_j, "retained": bool(retained),
                })

                if verbose:
                    extra = (f" N_s=1e{log10_ns:.2f} budget={depth_budget:.2e}"
                             if _bwd == "pathwise" else "")
                    print(f"  round {round_num+1:2d} bwd | j={j} s={s_cur} p={p_j:.2e} "
                          f"gate=1e{log10_gate:.2f}{extra}", flush=True)

                if not retained:
                    selected.remove(j)
                    n_rejections += 1
                    if verbose:
                        print(f"    → removed j{j}  S={selected}", flush=True)

        round_num += 1

    # Recompute final conditional p-values: p(j | S \ {j}) for every selected j.
    # This gives meaningful values for W-seeded features (which never pass through
    # the forward step and would otherwise be reported as 1.0).
    for j in list(selected):
        S_minus_j = [s for s in selected if s != j]
        selected_pvals[j] = float(_pvalues(S_minus_j, [j])[j])

    out_pvals = last_pvals.copy()
    for j in selected:
        out_pvals[j] = selected_pvals[j]

    # Feature names: w_{name} for prior features, z_{j} for neural features
    z_labels = [
        f"z_{z_names[j]}" if (z_names is not None and j < len(z_names) and z_names[j])
        else f"z_{j}"
        for j in range(m)
    ]
    if k > 0:
        w_labels = [
            f"w_{w_names[S_w[i]]}" if w_names else f"w_{S_w[i]}"
            for i in range(k)
        ]
        feature_names = w_labels + z_labels
    else:
        feature_names = z_labels

    if test == "gcm":
        test_label = "gcm_quadratic" if nuisance == "poly2" else "gcm_lgbm"
    elif test == "pcm":
        test_label = f"pcm_{pcm_projection}"
    else:
        test_label = test
    method_str = f"nexis_{test_label}"
    if w is not None:
        method_str = "w_" + method_str
    if not backward:
        method_str += "_fwd"
    if _bwd == "pathwise":
        method_str += "_pwbwd"
    if rho is not None:
        method_str += f"_sg{rho}"
    if _adj is None:
        method_str += "_noadj"
    elif _adj == "FDR":
        method_str += "_fdr"

    meta: Dict[str, object] = {
        "m": float(total),
        "steps": float(len(selected)),
        "rejections": float(n_rejections),
        "rounds": float(round_num),
        "test": test,
        "backward": backward,
        "backward_gate": _bwd,
        "backward_tests": float(len(backward_log)),
        "backward_removed": float(sum(1 for r in backward_log if not r["retained"])),
        "backward_log": backward_log,
    }
    if test in ("gcm", "pcm"):
        meta.update({"nuisance": nuisance, "n_splits": float(n_splits),
                     "n_estimators": float(n_estimators)})
    if test == "pcm":
        meta.update({"pcm_projection": pcm_projection, "pcm_order": float(pcm_order),
                     "pcm_combine": pcm_combine,
                     "pcm_screen_top": float(pcm_screen_top)})
    if rho is not None:
        meta["rho"] = float(rho)

    return SelectionResult(
        selected=selected,
        pvalues=out_pvals,
        method=method_str,
        alpha=alpha,
        metadata=meta,
        feature_names=feature_names,
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def iou_score(selected: Sequence[int], truth: Sequence[int]) -> float:
    S = set(int(x) for x in selected)
    T = set(int(x) for x in truth)
    union = S | T
    if len(union) == 0:
        return 1.0
    return len(S & T) / len(union)


def evaluate_methods_on_dataset(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    truth: Sequence[int],
    alpha: float = 0.05,
    max_rounds: Optional[int] = None,
    rho: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    out = {}
    truth_set = set(int(x) for x in truth)
    n_truth = len(truth_set)

    def _metrics(selected: Sequence[int]) -> Dict[str, float]:
        selected_set = set(int(x) for x in selected)
        tp = float(len(selected_set & truth_set))
        fp = float(len(selected_set - truth_set))
        n_selected = float(len(selected_set))
        recall = tp / n_truth if n_truth > 0 else 1.0
        precision = (tp / n_selected) if n_selected > 0 else (1.0 if n_truth == 0 else 0.0)
        return {
            "iou": iou_score(selected_set, truth_set),
            "n_selected": n_selected,
            "tp": tp,
            "fp": fp,
            "recall": float(recall),
            "precision": float(precision),
        }

    res = nexis(y=y, t=t, z=z, alpha=alpha, max_rounds=max_rounds,
                      test="linear")
    out["NEXIS (linear)"] = _metrics(res.selected)

    res = nexis(y=y, t=t, z=z, alpha=alpha, max_rounds=max_rounds,
                      test="linear", rho=rho)
    out["NEXIS (auto) (linear)"] = _metrics(res.selected)

    res = nexis(y=y, t=t, z=z, alpha=alpha, max_rounds=max_rounds,
                      test="gcm", nuisance="poly2")
    out["NEXIS (poly2)"] = _metrics(res.selected)

    res = nexis(y=y, t=t, z=z, alpha=alpha, max_rounds=max_rounds,
                      test="gcm", nuisance="poly2", rho=rho)
    out["NEXIS (auto) (poly2)"] = _metrics(res.selected)

    res = marginal_select(y=y, t=t, z=z, alpha=alpha, adjust="FWER")
    out["Marginal Testing (FWER)"] = _metrics(res.selected)

    res = marginal_select(y=y, t=t, z=z, alpha=alpha, adjust="FDR")
    out["Marginal Testing (FDR)"] = _metrics(res.selected)

    res = marginal_select(y=y, t=t, z=z, alpha=alpha, adjust=None)
    out["Marginal Testing"] = _metrics(res.selected)

    return out
