
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


@dataclass
class SelectionResult:
    selected: List[int]            # indices into the feature space nexis ran on
    pvalues: np.ndarray            # one entry per feature in that space
    method: str
    alpha: float
    metadata: Dict[str, float]
    feature_names: List[str] = field(default_factory=list)  # w_{name} / z_{j} labels


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
):
    """
    Vectorized p-values for H0(j|S) over j in candidates.
    Working model:
      Y = beta0 + betaT T + beta_S' Z_S + beta_j Z_j + gamma_S'(T*Z_S) + gamma_j (T*Z_j) + e
    Tests gamma_j = 0 for each j via FWL residualization against D=[1, T, Z_S, T*Z_S],
    then 2-regressor OLS per candidate on [Z_j, T*Z_j].

    cluster: reserved for future use. CR1S is not valid when z features are
      constant within clusters; pass pre-aggregated community-level data instead.
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
) -> np.ndarray:
    return conditional_interaction_pvalues(y=y, t=t, z=z, S=[])


def marginal_select(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    alpha: float = 0.05,
    adjust: Optional[str] = None,  # None | "FWER" | "FDR"
    groups: Optional[Dict[str, List[int]]] = None,
) -> SelectionResult:
    """Marginal interaction test with optional multiple-testing adjustment.

    adjust=None  : raw threshold at level alpha (no correction).
    adjust="FWER": Bonferroni correction (α/M).  groups splits the budget
                   per group instead of globally.
    adjust="FDR" : Benjamini-Hochberg step-up procedure at level alpha.
    """
    pvals = conditional_interaction_pvalues(y=y, t=t, z=z, S=[])
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
    test: str = "linear",            # "linear" | "quadratic" | "GCM"
    nuisance: str = "poly2",         # gcm only: "poly2" | "lgbm" | "rf"
    n_splits: int = 5,             # gcm only
    n_estimators: int = 100,       # gcm only
    max_depth: Optional[int] = None,  # gcm only
    cluster: Optional[np.ndarray] = None,  # CR1S cluster-robust SEs (linear test only)
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
    _test_key = test.lower().strip()
    if _test_key in {"gcm: quadratic", "quadratic"}:
        test, nuisance = "gcm", "poly2"
    elif _test_key in {"gcm: lgbm", "gcm", "lgbm"}:
        test, nuisance = "gcm", "lgbm"
    elif _test_key != "linear":
        raise ValueError("test must be 'linear', 'GCM: quadratic', or 'GCM: lgbm'")

    # rho=0 is treated as rho=None (gate 2 disabled).
    if rho is not None and rho == 0:
        rho = None

    y_arr = np.asarray(y, dtype=float).reshape(-1)
    t_arr = np.asarray(t, dtype=float).reshape(-1)
    Z = np.asarray(z, dtype=float)
    n, m = Z.shape

    if w is not None and w_names is None:
        raise ValueError("w_names is required when w is provided")

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
            rho=rho, adjust=adjust, cluster=cluster,
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
                        if test == "gcm" else {})

    def _pvalues(S_cur, candidates, return_tstats=False):
        if test == "gcm":
            return conditional_interaction_pvalues_gcm(
                y=y_arr, t=t_arr, z=Z, S=S_cur, candidates=candidates,
                return_tstats=return_tstats, **gcm_kwargs,
            )
        return conditional_interaction_pvalues(
            y=y_arr, t=t_arr, z=Z, S=S_cur, candidates=candidates,
            return_tstats=return_tstats, cluster=cluster,
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
                S_minus_j = [s for s in selected if s != j]
                pvals_back = _pvalues(S_minus_j, [j])
                p_j = float(pvals_back[j])
                gate_bwd = alpha if _adj is None else alpha / len(selected)

                if verbose:
                    print(f"  round {round_num+1:2d} bwd | j={j} p={p_j:.2e} "
                          f"gate={gate_bwd:.2e}", flush=True)

                if p_j > gate_bwd:
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
    else:
        test_label = test
    method_str = f"nexis_{test_label}"
    if w is not None:
        method_str = "w_" + method_str
    if not backward:
        method_str += "_fwd"
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
    }
    if test == "gcm":
        meta.update({"nuisance": nuisance, "n_splits": float(n_splits),
                     "n_estimators": float(n_estimators)})
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
