"""
Generic causal inference estimation utilities.

Provides OLS with HC1 robust standard errors, ATE estimation,
and GATE/CATE computation for binary/sparse/continuous effect modifiers.
"""

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


def ci95(est: float, se: float) -> tuple[float, float]:
    return est - 1.96 * se, est + 1.96 * se


def fmt_est(est: float, se: float) -> str:
    lo, hi = ci95(est, se)
    return f"{est:+.4f}  [{lo:+.4f}, {hi:+.4f}]  SE={se:.4f}"


# ── Feature classification ────────────────────────────────────────────────────

def classify_feature(z: np.ndarray):
    """Classify a feature vector and return a split threshold + labels.

    Returns one of:
      ('binary',     threshold, label_lo, label_hi)
      ('sparse',     0.0,       label_lo, label_hi)   — median=0, some nonzero
      ('continuous', median,    label_lo, label_hi)
    """
    unique_vals = np.unique(z[np.isfinite(z)])
    if len(unique_vals) <= 2:
        v0, v1 = unique_vals[0], unique_vals[1]
        return "binary", float(v0), \
               str(int(v0)) if v0 == int(v0) else f"{v0:.3f}", \
               str(int(v1)) if v1 == int(v1) else f"{v1:.3f}"

    median = float(np.median(z))
    frac_nonzero = float((z > 0).mean())

    if median == 0.0 and 0.02 < frac_nonzero < 0.98:
        return "sparse", 0.0, \
               f"inactive (Z=0, {1 - frac_nonzero:.0%})", \
               f"active   (Z>0, {frac_nonzero:.0%})"

    return "continuous", median, \
           f"≤{median:.4f} ({(z <= median).mean():.0%})", \
           f">{median:.4f}  ({(z > median).mean():.0%})"


# ── Per-feature GATE/CATE ─────────────────────────────────────────────────────

def feature_gate(
    y: np.ndarray,
    t: np.ndarray,
    W: np.ndarray | None,
    z: np.ndarray,
    feat_label: str,
    p_value: float,
    interp_label: str | None = None,
    vlm_label: str | None = None,
    vlm_confidence: str | None = None,
) -> dict:
    """Compute GATE/CATE for one effect modifier.

    Splits observations into low/high groups based on feature type,
    estimates ATE within each group via HC1-robust OLS.
    """
    ftype, threshold, lbl_lo, lbl_hi = classify_feature(z)

    mask_lo = (z == threshold) if ftype == "binary" else (z <= threshold)
    mask_hi = ~mask_lo

    W_lo = W[mask_lo] if W is not None else None
    W_hi = W[mask_hi] if W is not None else None

    gate_lo, se_lo, n_lo = ate_ols(y[mask_lo], t[mask_lo], W_lo)
    gate_hi, se_hi, n_hi = ate_ols(y[mask_hi], t[mask_hi], W_hi)

    return dict(
        label=feat_label, pvalue=p_value,
        interp=interp_label, vlm_label=vlm_label, vlm_confidence=vlm_confidence,
        ftype=ftype, threshold=threshold,
        lbl_lo=lbl_lo, lbl_hi=lbl_hi,
        gate_lo=gate_lo, se_lo=se_lo, n_lo=n_lo,
        gate_hi=gate_hi, se_hi=se_hi, n_hi=n_hi,
        diff=gate_hi - gate_lo,
    )
