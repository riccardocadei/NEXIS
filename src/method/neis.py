
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Dict, Tuple
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
            Bonferroni budget (α / group_size).
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


# ── NEIS ──────────────────────────────────────────────────────────────────────

def neis_select(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    alpha: float = 0.05,
    max_rounds: Optional[int] = 20,
    test: str = "linear",
    nuisance: str = "poly2",       # gcm only: "poly2" | "lgbm" | "rf"
    n_splits: int = 5,             # gcm only
    n_estimators: int = 100,       # gcm only
    max_depth: Optional[int] = None,  # gcm only
    controls: Optional[np.ndarray] = None,          # linear only
    main_controls: Optional[np.ndarray] = None,      # linear only
    interaction_only: Optional[set] = None,          # linear only
    auto_f: Optional[float] = None,
    effect_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    backward: bool = True,
    verbose: bool = False,
) -> SelectionResult:
    """Forward(-backward) selection (NEIS — Neural Exposure Interaction Search).

    Each round:
      1. Forward: among candidates passing Gate 1 (p ≤ α/|remaining|) and Gate 3
         (effect_range), pick the best.  Gate 2 (auto_f): stop if the chosen
         candidate's |t| < auto_f * min(|t| of already-selected features).
      2. Backward (skipped when backward=False): for every j ∈ S, remove j if
         p_j(S \\ {j}) > α/|S|.
    Repeats until S is unchanged (fixed point).

    backward=False runs a pure greedy forward pass — useful for ablation.

    test:
      "linear"  — parametric interaction t-test; fast, assumes linear effects.
                  Supports controls / main_controls / interaction_only.
      "gcm"     — GCM-hybrid test (nonparametric φ̂, linear Z^j residualisation).
                  Speed: ~1.2× (poly2), ~3× (lgbm), ~27× (rf) vs linear.
                  controls / main_controls / interaction_only are ignored.

    effect_range:
      (lo, hi) in the same units as the interaction coefficient γ_j.  The code
      estimates σ̂ by residualising Y on [1, T] and converts the range to t-stat
      bounds:  t_lo = lo * √n / σ̂,  t_hi = hi * √n / σ̂.
      In the forward step only candidates with t_lo ≤ |t_j| ≤ t_hi are eligible,
      even if they pass the α gate.  Either bound can be None (unbounded).
    """
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    t_arr = np.asarray(t, dtype=float).reshape(-1)
    Z = np.asarray(z, dtype=float)
    n, m = Z.shape

    # Convert effect_range to t-stat bounds once using σ̂ from Y ~ [1, T]
    if effect_range is not None:
        D_base = np.column_stack([np.ones(n), t_arr])
        sigma_hat = float(_residualize_against(D_base, y_arr).std(ddof=1))
        sigma_hat = sigma_hat if sigma_hat > 1e-10 else 1e-10
        sqrt_n = float(np.sqrt(n))
        lo, hi = effect_range
        t_gate_lo = (lo * sqrt_n / sigma_hat) if lo is not None else 0.0
        t_gate_hi = (hi * sqrt_n / sigma_hat) if hi is not None else np.inf
    else:
        t_gate_lo, t_gate_hi = 0.0, np.inf

    gcm_kwargs: dict = (dict(nuisance=nuisance, n_splits=n_splits,
                             n_estimators=n_estimators, max_depth=max_depth)
                        if test == "gcm" else {})

    def _pvalues(S_cur, candidates, return_tstats=False):
        if test == "gcm":
            return conditional_interaction_pvalues_gcm(
                y=y_arr, t=t_arr, z=Z, S=S_cur, candidates=candidates,
                return_tstats=return_tstats, **gcm_kwargs,
            )
        return conditional_interaction_pvalues(
            y=y_arr, t=t_arr, z=Z, S=S_cur, candidates=candidates,
            controls=controls, main_controls=main_controls,
            interaction_only=interaction_only, return_tstats=return_tstats,
        )

    selected: List[int] = []
    last_pvals = np.ones(m, dtype=float)
    selected_pvals = np.ones(m, dtype=float)
    t_selected: List[float] = []

    need_tstats = (auto_f is not None) or (effect_range is not None)

    S_prev: List[int] = [-1]  # sentinel
    round_num = 0
    _gate2_stop = False

    while selected != S_prev and not _gate2_stop:
        if max_rounds is not None and round_num >= max_rounds:
            break
        S_prev = list(selected)

        # ── Forward step ──────────────────────────────────────────────────────
        remaining = [j for j in range(m) if j not in selected]
        if remaining:
            if need_tstats:
                pvals, tstats = _pvalues(selected, remaining, return_tstats=True)
            else:
                pvals = _pvalues(selected, remaining)
                tstats = None
            last_pvals = pvals.copy()

            gate_fwd = alpha / len(remaining)

            # Gate 1 + Gate 3: find eligible candidates
            def _eligible(j: int) -> bool:
                if pvals[j] > gate_fwd:
                    return False
                if tstats is not None:
                    t_abs = float(abs(tstats[j]))
                    if not (t_gate_lo <= t_abs <= t_gate_hi):
                        return False
                return True

            eligible = [j for j in remaining if _eligible(j)]

            if eligible:
                if tstats is not None:
                    j_star = max(eligible, key=lambda j: abs(tstats[j]))
                else:
                    j_star = min(eligible, key=lambda j: pvals[j])
                p_star = float(pvals[j_star])

                if verbose:
                    t_str = (f" |t|={abs(tstats[j_star]):.2f}" if tstats is not None else "")
                    print(f"  round {round_num+1:2d} fwd | remaining={len(remaining):5d} "
                          f"gate={gate_fwd:.2e} eligible={len(eligible)} "
                          f"| best=j{j_star} p={p_star:.2e}{t_str}", flush=True)

                # Gate 2: relative stopping (auto_f)
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
            elif verbose:
                print(f"  round {round_num+1:2d} fwd | no eligible candidate "
                      f"(gate={gate_fwd:.2e})", flush=True)

        if _gate2_stop:
            break

        # ── Backward step ─────────────────────────────────────────────────────
        if not backward:
            round_num += 1
            continue
        for j in list(selected):
            if not selected:
                break
            S_minus_j = [k for k in selected if k != j]
            pvals_back = _pvalues(S_minus_j, [j])
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

    method_str = f"neis_{test}"
    if test == "gcm":
        method_str += f"_{nuisance}"
    if not backward:
        method_str += "_fwd"
    if auto_f is not None:
        method_str += f"_auto{auto_f}"
    if effect_range is not None:
        lo_s = str(effect_range[0]) if effect_range[0] is not None else "0"
        hi_s = str(effect_range[1]) if effect_range[1] is not None else "inf"
        method_str += f"_range{lo_s}-{hi_s}"

    meta: Dict[str, object] = {
        "m": float(m),
        "steps": float(len(selected)),
        "rounds": float(round_num),
        "test": test,
        "backward": backward,
    }
    if test == "gcm":
        meta.update({"nuisance": nuisance, "n_splits": float(n_splits),
                     "n_estimators": float(n_estimators)})
    if auto_f is not None:
        meta["auto_f"] = float(auto_f)
    if effect_range is not None:
        meta["t_gate_lo"] = t_gate_lo
        meta["t_gate_hi"] = float(t_gate_hi) if np.isfinite(t_gate_hi) else None

    return SelectionResult(
        selected=selected,
        pvalues=out_pvals,
        method=method_str,
        alpha=alpha,
        metadata=meta,
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
    auto_f: float = 0.5,
    effect_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
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

    res = neis_select(y=y, t=t, z=z, alpha=alpha, max_rounds=max_rounds,
                      test="linear", effect_range=effect_range)
    out["NEIS (linear)"] = _metrics(res.selected)

    res = neis_select(y=y, t=t, z=z, alpha=alpha, max_rounds=max_rounds,
                      test="linear", auto_f=auto_f, effect_range=effect_range)
    out["NEIS (auto) (linear)"] = _metrics(res.selected)

    res = neis_select(y=y, t=t, z=z, alpha=alpha, max_rounds=max_rounds,
                      test="gcm", nuisance="poly2", effect_range=effect_range)
    out["NEIS (poly2)"] = _metrics(res.selected)

    res = neis_select(y=y, t=t, z=z, alpha=alpha, max_rounds=max_rounds,
                      test="gcm", nuisance="poly2", auto_f=auto_f,
                      effect_range=effect_range)
    out["NEIS (auto) (poly2)"] = _metrics(res.selected)

    res = marginal_select(y=y, t=t, z=z, alpha=alpha, adjust="bonferroni")
    out["Marginal (Bon)"] = _metrics(res.selected)

    res = marginal_select(y=y, t=t, z=z, alpha=alpha, adjust="none")
    out["Marginal"] = _metrics(res.selected)

    return out
