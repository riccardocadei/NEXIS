
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Dict
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
) -> np.ndarray:
    """K-fold cross-fitted predictions from any sklearn-compatible model."""
    from sklearn.model_selection import KFold
    pred = np.zeros_like(y, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr, te in kf.split(X):
        m = model_factory()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


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

    # Conditioning input for nuisance model: Z^S columns (intercept handled internally)
    Z_S_fit = Z[:, S_list] if S_list else np.zeros((n, 1))

    # R-learner pseudo-outcome: phi = (Y - m_hat) * (T - e) / (e*(1-e))
    m_hat = _crossfit(Z_S_fit, y, model_fn, n_splits=n_splits)
    phi = (y - m_hat) * (t - e) / (e * (1 - e))

    # Residualize phi on Z^S to remove remaining dependence
    phi_resid = phi - _crossfit(Z_S_fit, phi, model_fn, n_splits=n_splits)

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


@dataclass
class SelectionResult:
    selected: List[int]
    pvalues: np.ndarray
    method: str
    alpha: float
    metadata: Dict[str, float]
    selected_groups: List[str] = field(default_factory=list)  # group per selection


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
    controls: Optional[np.ndarray] = None,
    main_controls: Optional[np.ndarray] = None,
    interaction_only: Optional[set] = None,
    return_tstats: bool = False,
):
    """
    Vectorized p-values for H0(j|S) over j in candidates.
    Working model:
      Y = beta0 + betaT T + beta_S' Z_S + beta_j Z_j + gamma_S'(T*Z_S) + gamma_j (T*Z_j)
          + beta_W' W + gamma_W'(T*W) + e
    Tests gamma_j = 0 for each j.

    controls: optional (n, q) matrix W always included in the nuisance design as
    [W, T*W].  Removes W-based heterogeneity (main effect + interaction) from
    residuals without consuming Bonferroni budget.

    main_controls: optional (n, q) matrix W included in the nuisance design as [W]
    only (main effects, NOT T*W).  Use this when W's interaction with T should be
    tested as a candidate rather than partialled out unconditionally.

    interaction_only: optional set of candidate column indices for which only
    T*Z_j is tested (1-regressor, 1 dof) rather than the standard [Z_j, T*Z_j]
    (2-regressor, 2 dof).  Use for W candidates when W main effects are already
    in main_controls: Z_j = W_k would residualize to ~0 against D (which already
    contains W_k), making the 2-regressor system degenerate.  The 1-regressor
    test correctly recovers H0: gamma_k = 0 with W_k partialled out via D.

    Uses Frisch-Waugh-Lovell residualization against D=[1, T, W, T*W, Z_S, T*Z_S],
    then a 2-regressor OLS per candidate on [Z_j, T*Z_j] (after residualization),
    or a 1-regressor OLS on [T*Z_j] for interaction_only candidates.
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

    # Split candidates into regular (2-reg) and interaction-only (1-reg)
    io_set = interaction_only if interaction_only is not None else set()
    reg_cand = np.array([j for j in cand if j not in io_set], dtype=int)
    io_cand  = np.array([j for j in cand if j in io_set],     dtype=int)

    # Common nuisance design D = [1, T, W, T*W, main_W, Z_S, T*Z_S]
    D_cols = [np.ones(n), t]
    if controls is not None:
        W = np.asarray(controls, dtype=float)
        for col in range(W.shape[1]):
            D_cols.append(W[:, col])
        for col in range(W.shape[1]):
            D_cols.append(t * W[:, col])
    if main_controls is not None:
        Wm = np.asarray(main_controls, dtype=float)
        for col in range(Wm.shape[1]):
            D_cols.append(Wm[:, col])
    for k in S:
        D_cols.append(Z[:, k])
    for k in S:
        D_cols.append(t * Z[:, k])
    D = np.column_stack(D_cols) if len(D_cols) > 0 else np.empty((n, 0), dtype=float)

    y_tilde = _residualize_against(D, y)  # (n,)
    yy = np.sum(y_tilde ** 2)

    # ── Standard 2-regressor test for regular candidates ──────────────────────
    if reg_cand.size > 0:
        Z_c = Z[:, reg_cand]                      # (n, K)
        TZ_c = (t[:, None] * Z_c)                 # (n, K)
        Z_tilde = _residualize_against(D, Z_c)
        X_tilde = _residualize_against(D, TZ_c)

        zz = np.sum(Z_tilde * Z_tilde, axis=0)
        xx = np.sum(X_tilde * X_tilde, axis=0)
        zx = np.sum(Z_tilde * X_tilde, axis=0)
        zy = np.sum(Z_tilde * y_tilde[:, None], axis=0)
        xy = np.sum(X_tilde * y_tilde[:, None], axis=0)

        # 2x2 OLS algebra for coeff on X_tilde (interaction term)
        det = zz * xx - zx * zx
        valid = det > 1e-12

        # Degrees of freedom = n - (#columns in full model)
        # full model columns = dim(D) + 2  [Z_j and T*Z_j]
        p_full = D.shape[1] + 2
        dof = n - p_full

        if dof > 0:
            beta_x = np.zeros_like(det)
            beta_z = np.zeros_like(det)
            beta_x[valid] = (zz[valid] * xy[valid] - zx[valid] * zy[valid]) / det[valid]
            beta_z[valid] = (xx[valid] * zy[valid] - zx[valid] * xy[valid]) / det[valid]

            # RSS = y'y - beta'X'y ; here X=[z,x]
            rss = np.full_like(det, np.nan, dtype=float)
            rss[valid] = yy - beta_z[valid] * zy[valid] - beta_x[valid] * xy[valid]
            rss = np.maximum(rss, 0.0)

            sigma2 = np.full_like(det, np.nan, dtype=float)
            sigma2[valid] = rss[valid] / dof

            # Var(beta_x) = sigma^2 * (G^{-1})_{22} = sigma^2 * zz/det
            var_bx = np.full_like(det, np.nan, dtype=float)
            var_bx[valid] = sigma2[valid] * (zz[valid] / det[valid])

            ok = valid & np.isfinite(var_bx) & (var_bx > 0)
            tstat = np.zeros_like(det, dtype=float)
            tstat[ok] = beta_x[ok] / np.sqrt(var_bx[ok])

            p = np.ones_like(det, dtype=float)
            p[ok] = 2.0 * stats.t.sf(np.abs(tstat[ok]), df=dof)
            p = np.clip(np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)
            pvals[reg_cand] = p
            all_tstats[reg_cand] = tstat

    # ── 1-regressor test for interaction_only candidates ──────────────────────
    # Tests H0: gamma_j = 0 using only T*Z_j, with Z_j already in D (main_controls).
    # D already contains W_k main effects, so T*W_k is the only free regressor.
    if io_cand.size > 0:
        TX_c = t[:, None] * Z[:, io_cand]          # (n, K_io)
        TX_tilde = _residualize_against(D, TX_c)   # (n, K_io)

        # dof = n - dim(D) - 1  (only 1 free regressor per candidate)
        dof1 = n - (D.shape[1] + 1)
        if dof1 > 0:
            tx_ty = np.sum(TX_tilde * y_tilde[:, None], axis=0)  # (K_io,)
            tx_tx = np.sum(TX_tilde * TX_tilde,         axis=0)  # (K_io,)
            valid_io = tx_tx > 1e-12

            beta_io = np.zeros_like(tx_tx)
            beta_io[valid_io] = tx_ty[valid_io] / tx_tx[valid_io]

            rss_io = np.full_like(tx_tx, yy)
            rss_io[valid_io] -= beta_io[valid_io] * tx_ty[valid_io]
            rss_io = np.maximum(rss_io, 0.0)

            sigma2_io = rss_io / dof1
            var_io = np.full_like(tx_tx, np.nan)
            var_io[valid_io] = sigma2_io[valid_io] / tx_tx[valid_io]

            ok_io = valid_io & np.isfinite(var_io) & (var_io > 0)
            tstat_io = np.zeros_like(tx_tx)
            tstat_io[ok_io] = beta_io[ok_io] / np.sqrt(var_io[ok_io])

            p_io = np.ones_like(tx_tx)
            p_io[ok_io] = 2.0 * stats.t.sf(np.abs(tstat_io[ok_io]), df=dof1)
            p_io = np.clip(np.nan_to_num(p_io, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)
            pvals[io_cand] = p_io
            all_tstats[io_cand] = tstat_io

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
    interaction_only: Optional[set] = None,
) -> np.ndarray:
    return conditional_interaction_pvalues(y=y, t=t, z=z, S=[],
                                           interaction_only=interaction_only)


def marginal_select(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    alpha: float = 0.05,
    adjust: str = "none",  # "none" or "bonferroni"
    controls: Optional[np.ndarray] = None,
    main_controls: Optional[np.ndarray] = None,
    groups: Optional[Dict[str, List[int]]] = None,
    interaction_only: Optional[set] = None,
) -> SelectionResult:
    """Marginal interaction test with optional multiple-testing adjustment.

    groups: if provided and adjust='bonferroni', each group gets its own
            Bonferroni budget (α / group_size), matching nems_select_grouped.
            Without groups, a single α/M correction is applied over all M
            candidates (standard Bonferroni).
    """
    pvals = conditional_interaction_pvalues(y=y, t=t, z=z, S=[], controls=controls,
                                            main_controls=main_controls,
                                            interaction_only=interaction_only)
    m = len(pvals)

    if adjust.lower() in {"none", "raw", "unadjusted"}:
        selected = np.where(pvals <= alpha)[0].tolist()
        method = "marginal_raw"
        metadata: Dict[str, float] = {"threshold": float(alpha), "m": float(m)}
    elif adjust.lower() in {"bonf", "bonferroni"}:
        if groups is not None:
            # Per-group Bonferroni: each group corrects for its own size only.
            mask = np.zeros(m, dtype=bool)
            for gname, gidxs in groups.items():
                thr_g = alpha / max(len(gidxs), 1)
                for j in gidxs:
                    if pvals[j] <= thr_g:
                        mask[j] = True
            selected = np.where(mask)[0].tolist()
            method = "marginal_bonferroni_grouped"
            metadata = {"m": float(m), **{
                f"thr_{g}": alpha / max(len(idxs), 1)
                for g, idxs in groups.items()
            }}
        else:
            thr = alpha / max(m, 1)
            selected = np.where(pvals <= thr)[0].tolist()
            method = "marginal_bonferroni"
            metadata = {"threshold": float(thr), "m": float(m)}
    else:
        raise ValueError("adjust must be 'none' or 'bonferroni'")

    return SelectionResult(
        selected=selected,
        pvalues=pvals,
        method=method,
        alpha=alpha,
        metadata=metadata,
    )


def participation_ratio(Z: np.ndarray) -> float:
    """Participation Ratio of the column space of Z.

    PR = (Σ λ_i)² / Σ λ_i²  where λ_i are eigenvalues of the covariance matrix.
    Equals the number of dimensions that carry equal variance — a scalar
    measure of effective dimensionality in [1, rank(Z)].

    Uses the gram matrix trick: when n < d, eigenvalues of Z @ Z.T (n×n)
    equal those of Z.T @ Z (d×d), so we pick whichever is smaller — O(min(n,d)³).
    """
    Z = np.asarray(Z, dtype=float)
    Z = Z - Z.mean(axis=0)
    n, d = Z.shape
    if n < d:
        G = (Z @ Z.T) / max(n - 1, 1)   # (n, n) gram matrix
    else:
        G = (Z.T @ Z) / max(n - 1, 1)   # (d, d) covariance matrix
    lam = np.linalg.eigvalsh(G)
    lam = lam[lam > 0]
    if lam.size == 0:
        return 1.0
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def nems_select(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    alpha: float = 0.05,
    max_steps: Optional[int] = None,
    correction: str = "bonferroni",  # "bonferroni" | "pr" | "bon_aem" | "none"
    pr0: Optional[float] = None,     # precomputed participation ratio for correction="pr"
    m_eff: Optional[int] = None,     # fixed effective M for correction="bonferroni"
    z_for_corr: Optional[np.ndarray] = None,  # override Z used to build R in "bon_aem"
    controls: Optional[np.ndarray] = None,
    main_controls: Optional[np.ndarray] = None,
    interaction_only: Optional[set] = None,
    auto_f: Optional[float] = None,  # relative stopping threshold f = 1/(C_pa * C_br)
    verbose: bool = False,
) -> SelectionResult:
    """Sequential conditional interaction selection.

    correction:
      "bonferroni" — gate = α / |remaining|  (adaptive); if m_eff is set,
                     gate = α / m_eff  (fixed effective M, ignores remaining count)
      "pr"         — gate = α / max(PR₀ − step, 1)  where PR₀ = participation
                     ratio of Z computed once at step 0; decremented by 1 each step
      "bon_aem"    — Adaptive Effective Multiplicity; gate = α / M_eff(A) where
                     M_eff(A) = |A|² / Σ_{j,k∈A} ρ_{jk}²  (trace-ratio on the
                     active correlation submatrix).  The full correlation matrix R
                     is computed once from z_for_corr (if provided) or z; D_A is
                     updated in O(|A|) per step via:
                     D_{A\{r}} = D_A − 1 − 2·Σ_{j∈A\{r}} ρ²_{rj}
      "none"       — gate = α  (no multiple-testing correction)
    """
    Z = np.asarray(z, dtype=float)
    m = Z.shape[1]
    selected: List[int] = []
    last_pvals = np.ones(m, dtype=float)

    # Store the p-value at the step each feature is selected (before it moves into S
    # and gets excluded from subsequent computations, which would reset it to 1.0).
    selected_pvals = np.ones(m, dtype=float)

    # PR-based correction: use precomputed value if provided, else compute from Z
    if correction == "pr":
        pr0 = float(pr0) if pr0 is not None else participation_ratio(Z)
    else:
        pr0 = float(m)

    # AEM: compute full correlation matrix once, initialise D_A = Σ R²_{jk}
    # Use z_for_corr if provided (e.g. continuous precodes when z is sparse codes).
    if correction == "bon_aem":
        Z_corr = np.asarray(z_for_corr, dtype=float) if z_for_corr is not None else Z
        Z_c = Z_corr - Z_corr.mean(axis=0)
        col_stds = Z_c.std(axis=0)
        col_stds = np.where(col_stds < 1e-8, 1.0, col_stds)   # safe for const cols
        Z_norm = Z_c / col_stds                                 # (n, m) unit-std
        R = (Z_norm.T @ Z_norm) / Z_corr.shape[0]              # (m, m) correlation
        np.fill_diagonal(R, 1.0)                                # exact 1s on diagonal
        R_sq = R ** 2                                           # (m, m) squared correlations
        aem_active = np.ones(m, dtype=bool)                     # which features are still in A
        aem_D = float(R_sq.sum())                               # Σ_{j,k∈A} ρ²_{jk}
        aem_meff_traj: List[float] = []
    else:
        R_sq = None
        aem_active = None
        aem_D = 0.0
        aem_meff_traj = []

    # Auto stopping: track |t-stat| of each selected feature for Gate 2.
    t_selected: List[float] = []

    step = 0
    while True:
        if max_steps is not None and step >= max_steps:
            break

        remaining = [j for j in range(m) if j not in selected]
        if not remaining:
            break

        # Request t-stats when auto_f is set (needed for Gate 2 and tie-breaking)
        if auto_f is not None:
            pvals, tstats = conditional_interaction_pvalues(
                y=y, t=t, z=Z, S=selected, candidates=remaining,
                controls=controls, main_controls=main_controls,
                interaction_only=interaction_only, return_tstats=True)
        else:
            pvals = conditional_interaction_pvalues(
                y=y, t=t, z=Z, S=selected, candidates=remaining,
                controls=controls, main_controls=main_controls,
                interaction_only=interaction_only)
            tstats = None
        last_pvals = pvals.copy()

        # Select best candidate: by |t-stat| (breaks p-value ties at machine epsilon)
        # when tstats available, otherwise by minimum p-value.
        if tstats is not None:
            rem_abs_t = np.abs(tstats[remaining])
            j_star = remaining[int(np.argmax(rem_abs_t))]
        else:
            rem_p = pvals[remaining]
            j_star = remaining[int(np.argmin(rem_p))]
        p_star = pvals[j_star]

        if correction == "bonferroni":
            gate = alpha / (m_eff if m_eff is not None else len(remaining))
        elif correction == "pr":
            _pr_meff = max(pr0 - step, 1.0)
            gate = alpha / _pr_meff
        elif correction == "bon_aem":
            n_active = int(aem_active.sum())
            _aem_meff = (n_active ** 2) / max(aem_D, 1e-12)
            aem_meff_traj.append(_aem_meff)
            gate = alpha / max(_aem_meff, 1.0)
        else:  # "none"
            gate = alpha

        n_pass = int(np.sum(pvals[remaining] <= gate))

        if verbose:
            t_str = (f" |t|={abs(tstats[j_star]):.2f}" if tstats is not None else "")
            print(f"    step {step+1:2d} | remaining={len(remaining):5d} "
                  f"gate={gate:.2e} | passing={n_pass:4d} "
                  f"| best=j{j_star} p={p_star:.2e}{t_str}", flush=True)

        if p_star <= gate:
            # Gate 2: relative stopping criterion (auto_f).
            # Stop if new candidate's |t| < auto_f * min |t| of already-selected features.
            # Only active once at least one feature has been selected.
            if auto_f is not None and tstats is not None and len(t_selected) > 0:
                t_new = float(abs(tstats[j_star]))
                t_min_found = min(t_selected)
                if t_min_found > 0 and t_new < auto_f * t_min_found:
                    if verbose:
                        print(f"    → auto-stopped (Gate 2): "
                              f"|t_new|={t_new:.2f} < f={auto_f} × t_min={t_min_found:.2f} "
                              f"= {auto_f * t_min_found:.2f}")
                    break

            selected_pvals[j_star] = p_star   # record before j_star enters S
            selected.append(j_star)
            if auto_f is not None and tstats is not None:
                t_selected.append(float(abs(tstats[j_star])))
            step += 1

            # AEM: efficient O(|A|) update of D_A after removing j_star
            # D_{A\{r}} = D_A - 1 - 2·Σ_{j∈A\{r}} ρ²_{r,j}
            if correction == "bon_aem":
                cross = float(R_sq[j_star, aem_active].sum()) - 1.0  # subtract ρ²_{rr}=1
                aem_D -= (1.0 + 2.0 * cross)
                aem_D = max(aem_D, 1e-12)
                aem_active[j_star] = False
        else:
            if verbose:
                print(f"    → stopped (Gate 1): best p={p_star:.2e} > gate={gate:.2e}")
            break

    # Merge: selected features use their selection-step p-value; others use last step.
    out_pvals = last_pvals.copy()
    for j in selected:
        out_pvals[j] = selected_pvals[j]

    _method_map = {
        "bonferroni": "nems_bonferroni",
        "pr":         "nems_pr",
        "bon_aem":    "nems_bon_aem",
        "none":       "nems_none",
    }
    if correction == "bonferroni" and m_eff is not None:
        method_str = f"nems_bonferroni_meff{m_eff}"
    else:
        method_str = _method_map.get(correction, "nems")
    if auto_f is not None:
        method_str += f"_auto{auto_f}"

    extra_meta: dict = {}
    if m_eff is not None and correction == "bonferroni":
        extra_meta["m_eff"] = float(m_eff)
    if correction == "bon_aem":
        extra_meta["aem_meff_trajectory"] = aem_meff_traj
    if auto_f is not None:
        extra_meta["auto_f"] = float(auto_f)

    return SelectionResult(
        selected=selected,
        pvalues=out_pvals,
        method=method_str,
        alpha=alpha,
        metadata={"m": float(m), "steps": float(len(selected)),
                  "correction": correction, "pr0": float(pr0),
                  **extra_meta},
    )


def nems_select_grouped(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    groups: Dict[str, List[int]],
    alpha: float = 0.05,
    max_steps: Optional[int] = None,
    controls: Optional[np.ndarray] = None,
    main_controls: Optional[np.ndarray] = None,
    interaction_only: Optional[set] = None,
    priority_groups: Optional[List[str]] = None,
    verbose: bool = False,
) -> SelectionResult:
    """Stratified NEMS: each group gets its own Bonferroni budget.

    At every forward step all candidates are scored jointly (same residuals),
    but the gate for candidate j is  α / |remaining in j's group|  rather than
    α / |all remaining|.  The candidate with the lowest p-value that clears its
    group gate is selected.  This controls FWER within each group at level α
    without forcing the small W pool to compete against 3072 SAE features.

    groups: dict mapping group name → list of column indices in z.
            Every column index should appear in exactly one group.

    priority_groups: optional list of group names that get first pick.  If any
            candidate in a priority group clears its gate, the best such
            candidate is selected — even if a non-priority group has a lower
            p-value.  Falls back to the normal (lowest-p-across-all-groups)
            rule only when no priority-group candidate passes its gate.
    """
    Z = np.asarray(z, dtype=float)
    m = Z.shape[1]

    # Build reverse lookup: column index → group name
    col_to_group: Dict[int, str] = {}
    for gname, idxs in groups.items():
        for idx in idxs:
            col_to_group[idx] = gname

    priority_set: set = set(priority_groups) if priority_groups else set()

    selected: List[int] = []
    selected_groups_list: List[str] = []
    last_pvals = np.ones(m, dtype=float)
    selected_pvals = np.ones(m, dtype=float)
    step = 0

    while True:
        if max_steps is not None and step >= max_steps:
            break

        remaining = [j for j in range(m) if j not in selected]
        if not remaining:
            break

        pvals = conditional_interaction_pvalues(
            y=y, t=t, z=Z, S=selected, candidates=remaining,
            controls=controls, main_controls=main_controls,
            interaction_only=interaction_only,
        )
        last_pvals = pvals.copy()

        # For each group find its best remaining candidate and check its gate
        priority_best_j: Optional[int] = None
        priority_best_p: float = 1.0
        fallback_best_j: Optional[int] = None
        fallback_best_p: float = 1.0
        group_stats: List[str] = []
        for gname, gidxs in groups.items():
            rem_g = [j for j in gidxs if j not in selected]
            if not rem_g:
                group_stats.append(f"{gname}:exhausted")
                continue
            gate = alpha / len(rem_g)
            j_g = min(rem_g, key=lambda j: pvals[j])
            p_g = pvals[j_g]
            n_pass_g = int(sum(1 for j in rem_g if pvals[j] <= gate))
            group_stats.append(f"{gname}:rem={len(rem_g)},pass={n_pass_g},best=p{p_g:.2e}")
            if p_g <= gate:
                if gname in priority_set and p_g < priority_best_p:
                    priority_best_p = p_g
                    priority_best_j = j_g
                elif p_g < fallback_best_p:
                    fallback_best_p = p_g
                    fallback_best_j = j_g

        if verbose:
            print(f"    step {step+1:2d} | S={selected} | " + "  ".join(group_stats),
                  flush=True)

        # Priority groups take precedence; fall back to best overall if none pass
        if priority_best_j is not None:
            best_j, best_p = priority_best_j, priority_best_p
        elif fallback_best_j is not None:
            best_j, best_p = fallback_best_j, fallback_best_p
        else:
            if verbose:
                print(f"    → stopped: no group cleared its gate")
            break

        if verbose:
            print(f"      → selected j={best_j} (group={col_to_group[best_j]}) "
                  f"p={best_p:.2e}", flush=True)
        selected_pvals[best_j] = best_p
        selected.append(best_j)
        selected_groups_list.append(col_to_group[best_j])
        step += 1

    out_pvals = last_pvals.copy()
    for j in selected:
        out_pvals[j] = selected_pvals[j]

    group_sizes = {gname: len(idxs) for gname, idxs in groups.items()}
    return SelectionResult(
        selected=selected,
        pvalues=out_pvals,
        method="nems_grouped",
        alpha=alpha,
        metadata={"m": float(m), "steps": float(len(selected)), **{
            f"m_{g}": float(s) for g, s in group_sizes.items()
        }},
        selected_groups=selected_groups_list,
    )


def neis_select(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    alpha: float = 0.05,
    max_rounds: Optional[int] = 20,
    controls: Optional[np.ndarray] = None,
    main_controls: Optional[np.ndarray] = None,
    interaction_only: Optional[set] = None,
    auto_f: Optional[float] = None,  # Gate 2: stop if |t_new| < auto_f * min(|t_selected|)
    verbose: bool = False,
) -> SelectionResult:
    """Forward-backward selection (NEIS — Neural Exposure Interaction Search).

    Each round:
      1. Forward: add the best remaining j if p_j(S) ≤ α/|S̃|  (Bonferroni over remaining).
         Gate 2 (auto_f): stop before adding if |t_new| < auto_f * min(|t_selected|).
      2. Backward: for every j ∈ S, remove j if p_j(S \\ {j}) > α/|S|.
    Repeats until S is unchanged (fixed point).

    Gate for the backward check uses the live |S| at each test, matching the
    reference implementation in the paper (Algorithm 1 / Appendix B.2).
    """
    Z = np.asarray(z, dtype=float)
    m = Z.shape[1]
    selected: List[int] = []
    last_pvals = np.ones(m, dtype=float)
    selected_pvals = np.ones(m, dtype=float)
    t_selected: List[float] = []  # |t| of each selected feature (for Gate 2)

    S_prev: List[int] = [-1]  # sentinel — guaranteed != [] on first entry
    round_num = 0
    _gate2_stop = False

    while selected != S_prev and not _gate2_stop:
        if max_rounds is not None and round_num >= max_rounds:
            break
        S_prev = list(selected)

        # ── Forward step ──────────────────────────────────────────────────────
        remaining = [j for j in range(m) if j not in selected]
        if remaining:
            if auto_f is not None:
                pvals, tstats = conditional_interaction_pvalues(
                    y=y, t=t, z=Z, S=selected, candidates=remaining,
                    controls=controls, main_controls=main_controls,
                    interaction_only=interaction_only, return_tstats=True,
                )
            else:
                pvals = conditional_interaction_pvalues(
                    y=y, t=t, z=Z, S=selected, candidates=remaining,
                    controls=controls, main_controls=main_controls,
                    interaction_only=interaction_only,
                )
                tstats = None
            last_pvals = pvals.copy()

            if tstats is not None:
                j_star = remaining[int(np.argmax(np.abs(tstats[remaining])))]
            else:
                j_star = remaining[int(np.argmin(pvals[remaining]))]
            p_star = float(pvals[j_star])
            gate_fwd = alpha / len(remaining)

            if verbose:
                t_str = (f" |t|={abs(tstats[j_star]):.2f}" if tstats is not None else "")
                print(f"  round {round_num+1:2d} fwd | remaining={len(remaining):5d} "
                      f"gate={gate_fwd:.2e} | best=j{j_star} p={p_star:.2e}{t_str}", flush=True)

            if p_star <= gate_fwd:
                if auto_f is not None and tstats is not None and len(t_selected) > 0:
                    t_new = float(abs(tstats[j_star]))
                    t_min_found = min(t_selected)
                    if t_min_found > 0 and t_new < auto_f * t_min_found:
                        if verbose:
                            print(f"    → auto-stopped (Gate 2): "
                                  f"|t_new|={t_new:.2f} < f={auto_f} × "
                                  f"t_min={t_min_found:.2f}", flush=True)
                        _gate2_stop = True
                if not _gate2_stop:
                    selected_pvals[j_star] = p_star
                    selected.append(j_star)
                    if tstats is not None:
                        t_selected.append(float(abs(tstats[j_star])))
                    if verbose:
                        print(f"    → added j{j_star}  S={selected}", flush=True)

        if _gate2_stop:
            break

        # ── Backward step ─────────────────────────────────────────────────────
        for j in list(selected):
            if not selected:
                break
            S_minus_j = [k for k in selected if k != j]
            pvals_back = conditional_interaction_pvalues(
                y=y, t=t, z=Z, S=S_minus_j, candidates=[j],
                controls=controls, main_controls=main_controls,
                interaction_only=interaction_only,
            )
            p_j = float(pvals_back[j])
            gate_bwd = alpha / len(selected)  # live |S|, matching Algorithm 1

            if verbose:
                print(f"  round {round_num+1:2d} bwd | j={j} p={p_j:.2e} "
                      f"gate={gate_bwd:.2e}", flush=True)

            if p_j > gate_bwd:
                selected.remove(j)
                if verbose:
                    print(f"    → removed j{j}  S={selected}", flush=True)

        round_num += 1

    out_pvals = last_pvals.copy()
    for j in selected:
        out_pvals[j] = selected_pvals[j]

    method_str = "neis"
    if auto_f is not None:
        method_str += f"_auto{auto_f}"

    return SelectionResult(
        selected=selected,
        pvalues=out_pvals,
        method=method_str,
        alpha=alpha,
        metadata={"m": float(m), "steps": float(len(selected)),
                  "rounds": float(round_num),
                  **({"auto_f": float(auto_f)} if auto_f is not None else {})},
    )


def neis_select_gcm(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    alpha: float = 0.05,
    max_rounds: Optional[int] = 20,
    nuisance: str = "poly2",
    n_splits: int = 5,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    auto_f: Optional[float] = None,  # Gate 2: stop if |t_new| < auto_f * min(|t_selected|)
    verbose: bool = False,
) -> SelectionResult:
    """NEIS with GCM-hybrid test (nonparametric φ̂, linear Z^j residualization).

    Same forward-backward logic as neis_select, using conditional_interaction_pvalues_gcm
    for each p-value.  The nuisance model for φ̂ = (Y−m̂(Z^S))(T−e)/(e(1−e)) is
    controlled by `nuisance`:

      "poly2"  ~1.2× vs linear  Ridge on poly(degree=2) features of Z^S.
                                 Handles quadratic main-effect nonlinearity.
                                 Default: best speed/quality trade-off.
      "lgbm"   ~3× vs linear    LightGBM shallow trees; fully nonparametric.
      "rf"     ~27× vs linear   Random Forest; most robust; use for final results.
    """
    Z = np.asarray(z, dtype=float)
    m = Z.shape[1]
    selected: List[int] = []
    last_pvals = np.ones(m, dtype=float)
    selected_pvals = np.ones(m, dtype=float)
    t_selected: List[float] = []  # |z| of each selected feature (for Gate 2)

    gcm_kwargs = dict(nuisance=nuisance, n_splits=n_splits,
                      n_estimators=n_estimators, max_depth=max_depth)

    S_prev: List[int] = [-1]
    round_num = 0
    _gate2_stop = False

    while selected != S_prev and not _gate2_stop:
        if max_rounds is not None and round_num >= max_rounds:
            break
        S_prev = list(selected)

        # ── Forward step ──────────────────────────────────────────────────────
        remaining = [j for j in range(m) if j not in selected]
        if remaining:
            if auto_f is not None:
                pvals, tstats = conditional_interaction_pvalues_gcm(
                    y=y, t=t, z=Z, S=selected, candidates=remaining,
                    return_tstats=True, **gcm_kwargs,
                )
            else:
                pvals = conditional_interaction_pvalues_gcm(
                    y=y, t=t, z=Z, S=selected, candidates=remaining, **gcm_kwargs,
                )
                tstats = None
            last_pvals = pvals.copy()

            if tstats is not None:
                j_star = remaining[int(np.argmax(np.abs(tstats[remaining])))]
            else:
                j_star = remaining[int(np.argmin(pvals[remaining]))]
            p_star = float(pvals[j_star])
            gate_fwd = alpha / len(remaining)

            if verbose:
                t_str = (f" |z|={abs(tstats[j_star]):.2f}" if tstats is not None else "")
                print(f"  round {round_num+1:2d} fwd | remaining={len(remaining):5d} "
                      f"gate={gate_fwd:.2e} | best=j{j_star} p={p_star:.2e}{t_str}", flush=True)

            if p_star <= gate_fwd:
                if auto_f is not None and tstats is not None and len(t_selected) > 0:
                    t_new = float(abs(tstats[j_star]))
                    t_min_found = min(t_selected)
                    if t_min_found > 0 and t_new < auto_f * t_min_found:
                        if verbose:
                            print(f"    → auto-stopped (Gate 2): "
                                  f"|z_new|={t_new:.2f} < f={auto_f} × "
                                  f"z_min={t_min_found:.2f}", flush=True)
                        _gate2_stop = True
                if not _gate2_stop:
                    selected_pvals[j_star] = p_star
                    selected.append(j_star)
                    if tstats is not None:
                        t_selected.append(float(abs(tstats[j_star])))
                    if verbose:
                        print(f"    → added j{j_star}  S={selected}", flush=True)

        if _gate2_stop:
            break

        # ── Backward step ─────────────────────────────────────────────────────
        for j in list(selected):
            if not selected:
                break
            S_minus_j = [k for k in selected if k != j]
            pvals_back = conditional_interaction_pvalues_gcm(
                y=y, t=t, z=Z, S=S_minus_j, candidates=[j], **gcm_kwargs,
            )
            p_j = float(pvals_back[j])
            gate_bwd = alpha / len(selected)

            if verbose:
                print(f"  round {round_num+1:2d} bwd | j={j} p={p_j:.2e} "
                      f"gate={gate_bwd:.2e}", flush=True)

            if p_j > gate_bwd:
                selected.remove(j)
                if verbose:
                    print(f"    → removed j{j}  S={selected}", flush=True)

        round_num += 1

    out_pvals = last_pvals.copy()
    for j in selected:
        out_pvals[j] = selected_pvals[j]

    method_str = f"neis_gcm_{nuisance}"
    if auto_f is not None:
        method_str += f"_auto{auto_f}"

    return SelectionResult(
        selected=selected,
        pvalues=out_pvals,
        method=method_str,
        alpha=alpha,
        metadata={"m": float(m), "steps": float(len(selected)),
                  "rounds": float(round_num), "nuisance": nuisance,
                  "n_splits": float(n_splits), "n_estimators": float(n_estimators),
                  **({"auto_f": float(auto_f)} if auto_f is not None else {})},
    )


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
    nems_max_steps: Optional[int] = None,
    pr0: Optional[float] = None,   # unused, kept for API compatibility
    z_precode: Optional[np.ndarray] = None,  # continuous precodes for AEM M_eff
    include_gcm: bool = False,     # add NEIS (GCM): ~27× slower, submit separately
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
        if n_selected > 0:
            precision = tp / n_selected
        else:
            precision = 1.0 if n_truth == 0 else 0.0
        return {
            "iou": iou_score(selected_set, truth_set),
            "n_selected": n_selected,
            "tp": tp,
            "fp": fp,
            "recall": float(recall),
            "precision": float(precision),
        }

    res_auto = nems_select(y=y, t=t, z=z, alpha=alpha, max_steps=nems_max_steps,
                           correction="bonferroni", auto_f=0.5)
    out["NEMS (auto)"] = _metrics(res_auto.selected)

    res_bonf = marginal_select(y=y, t=t, z=z, alpha=alpha, adjust="bonferroni")
    out["Marginal (Bon)"] = _metrics(res_bonf.selected)

    res_marginal = marginal_select(y=y, t=t, z=z, alpha=alpha, adjust="none")
    out["Marginal"] = _metrics(res_marginal.selected)

    res_neis = neis_select(y=y, t=t, z=z, alpha=alpha, max_rounds=nems_max_steps)
    out["NEIS (linear)"] = _metrics(res_neis.selected)

    res_neis_auto = neis_select(y=y, t=t, z=z, alpha=alpha, max_rounds=nems_max_steps,
                                auto_f=0.5)
    out["NEIS (auto) (linear)"] = _metrics(res_neis_auto.selected)

    res_neis_poly = neis_select_gcm(y=y, t=t, z=z, alpha=alpha, nuisance="poly2",
                                    max_rounds=nems_max_steps)
    out["NEIS (poly2)"] = _metrics(res_neis_poly.selected)

    res_neis_auto_poly = neis_select_gcm(y=y, t=t, z=z, alpha=alpha, nuisance="poly2",
                                         max_rounds=nems_max_steps, auto_f=0.5)
    out["NEIS (auto) (poly2)"] = _metrics(res_neis_auto_poly.selected)

    return out
