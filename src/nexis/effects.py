"""Treatment-effect estimation: OLS with HC1 robust standard errors and the ATE."""

from __future__ import annotations

import numpy as np


# ── OLS with HC1 standard errors ──────────────────────────────────────────────

def _ols_hc1(y: np.ndarray, X: np.ndarray):
    """Return (beta, se) from OLS with HC1 heteroscedasticity-robust SEs."""
    n, k = X.shape
    try:
        XtXi = np.linalg.pinv(X.T @ X)
        beta  = XtXi @ X.T @ y
        resid = y - X @ beta
        Xe    = X * resid[:, None]
        meat  = Xe.T @ Xe
        V     = XtXi @ meat @ XtXi * (n / (n - k))
        se    = np.sqrt(np.maximum(np.diag(V), 0.0))
        return beta, se
    except np.linalg.LinAlgError:
        return np.full(k, np.nan), np.full(k, np.nan)


def ate_ols(y: np.ndarray, t: np.ndarray, W: np.ndarray | None):
    """OLS: y ~ 1 + t + W.  Returns (estimate, se, n)."""
    n = len(y)
    cols = [np.ones(n), t]
    if W is not None and W.shape[1] > 0:
        keep = np.std(W, axis=0) > 1e-8
        if keep.any():
            cols.append(W[:, keep])
    X = np.column_stack(cols)
    beta, se = _ols_hc1(y, X)
    return float(beta[1]), float(se[1]), n
