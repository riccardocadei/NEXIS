
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Dict
import numpy as np
from scipy import stats


@dataclass
class SelectionResult:
    selected: List[int]
    pvalues: np.ndarray
    method: str
    alpha: float
    metadata: Dict[str, float]


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
) -> np.ndarray:
    """
    Vectorized p-values for H0(j|S) over j in candidates.
    Working model:
      Y = beta0 + betaT T + beta_S' Z_S + beta_j Z_j + gamma_S'(T*Z_S) + gamma_j (T*Z_j) + e
    Tests gamma_j = 0 for each j.

    Uses Frisch-Waugh-Lovell residualization against D=[1, T, Z_S, T*Z_S],
    then a 2-regressor OLS per candidate on [Z_j, T*Z_j] (after residualization).
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
    if cand.size == 0:
        return pvals

    # Common nuisance design D = [1, T, Z_S, T*Z_S]
    D_cols = [np.ones(n), t]
    for k in S:
        D_cols.append(Z[:, k])
    for k in S:
        D_cols.append(t * Z[:, k])
    D = np.column_stack(D_cols) if len(D_cols) > 0 else np.empty((n, 0), dtype=float)

    y_tilde = _residualize_against(D, y)  # (n,)
    Z_c = Z[:, cand]                      # (n, K)
    TZ_c = (t[:, None] * Z_c)             # (n, K)
    Z_tilde = _residualize_against(D, Z_c)
    X_tilde = _residualize_against(D, TZ_c)

    yy = np.sum(y_tilde ** 2)
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
    if dof <= 0:
        return pvals

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

    pvals[cand] = p
    return pvals


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
    adjust: str = "none",  # "none" or "bonferroni"
) -> SelectionResult:
    pvals = marginal_interaction_pvalues(y=y, t=t, z=z)
    m = len(pvals)

    if adjust.lower() in {"none", "raw", "unadjusted"}:
        thr = alpha
        method = "marginal_raw"
    elif adjust.lower() in {"bonf", "bonferroni"}:
        thr = alpha / max(m, 1)
        method = "marginal_bonferroni"
    else:
        raise ValueError("adjust must be 'none' or 'bonferroni'")

    selected = np.where(pvals <= thr)[0].tolist()
    return SelectionResult(
        selected=selected,
        pvalues=pvals,
        method=method,
        alpha=alpha,
        metadata={"threshold": float(thr), "m": float(m)},
    )


def nems_select(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    alpha: float = 0.05,
    max_steps: Optional[int] = None,
) -> SelectionResult:
    Z = np.asarray(z, dtype=float)
    m = Z.shape[1]
    selected: List[int] = []
    last_pvals = np.ones(m, dtype=float)

    step = 0
    while True:
        if max_steps is not None and step >= max_steps:
            break

        remaining = [j for j in range(m) if j not in selected]
        if not remaining:
            break

        pvals = conditional_interaction_pvalues(y=y, t=t, z=Z, S=selected, candidates=remaining)
        last_pvals = pvals.copy()

        rem_p = pvals[remaining]
        j_star = remaining[int(np.argmin(rem_p))]
        p_star = pvals[j_star]
        gate = alpha / len(remaining)  # Bonferroni gate over remaining coordinates

        if p_star <= gate:
            selected.append(j_star)
            step += 1
        else:
            break

    return SelectionResult(
        selected=selected,
        pvalues=last_pvals,
        method="nems",
        alpha=alpha,
        metadata={"m": float(m), "steps": float(len(selected))},
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

    res_nems = nems_select(y=y, t=t, z=z, alpha=alpha, max_steps=nems_max_steps)
    out["NEMS"] = _metrics(res_nems.selected)

    res_raw = marginal_select(y=y, t=t, z=z, alpha=alpha, adjust="none")
    out["Marginal"] = _metrics(res_raw.selected)

    res_bonf = marginal_select(y=y, t=t, z=z, alpha=alpha, adjust="bonferroni")
    out["Marginal (Bonferroni)"] = _metrics(res_bonf.selected)

    return out
