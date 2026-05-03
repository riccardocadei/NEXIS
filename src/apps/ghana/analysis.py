"""Ghana LEAP 1000 — DiD estimation and covariate balance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ── Naive (means-based) DiD ───────────────────────────────────────────────────

def naive_did(df: pd.DataFrame, outcome: str = 'Y') -> pd.DataFrame:
    """Compute the naive (cell-means) DiD estimate.

    Returns a 3×3 table (Baseline / Endline / DiD) ×
    (Comparison / Treatment / Diff T−C).
    """
    means = df.groupby(['wave', 'T'])[outcome].mean().unstack('T')
    means.index = pd.Index(['Baseline', 'Endline'], name='Wave')
    means.columns = pd.Index(['Comparison', 'Treatment'], name='Arm')
    means['Diff (T−C)'] = means['Treatment'] - means['Comparison']
    means.loc['DiD'] = means.loc['Endline'] - means.loc['Baseline']
    return means


# ── Regression DiD (OLS with HC1 SEs) ────────────────────────────────────────

def regression_did(df: pd.DataFrame, outcome: str = 'Y',
                   cluster: str | None = 'comm') -> pd.DataFrame:
    """OLS DiD: Y = α + β₁·T + β₂·wave + δ·(T×wave) + ε.

    Returns a DataFrame with Coef, SE, t-stat, and 95% CI.
    Under the parallel trends assumption, δ = DiD = ITT.

    Parameters
    ----------
    cluster : column name to use for cluster-robust variance estimation, or
              None to fall back to HC1.  Defaults to 'comm' (162 communities),
              which gives far more reliable cluster SEs than the 5 available
              districts.  Note: comm mixes T and C households so it is not the
              exact randomisation unit, but geographic clustering is still the
              right correction for spatial correlation in residuals.
    """
    cols = ['T', 'wave', outcome] + ([cluster] if cluster else [])
    sub  = df.dropna(subset=cols).copy()
    y    = sub[outcome].values
    T    = sub['T'].values
    w    = sub['wave'].values
    D    = T * w  # DiD indicator

    X = np.column_stack([np.ones(len(y)), T, w, D])
    n, k = X.shape

    XtXi = np.linalg.inv(X.T @ X)
    beta  = XtXi @ X.T @ y
    resid = y - X @ beta

    if cluster is not None:
        # Cluster-robust (CRVE) sandwich estimator
        groups = sub[cluster].values
        unique_g = np.unique(groups)
        G = len(unique_g)
        meat = np.zeros((k, k))
        for g in unique_g:
            mask = groups == g
            Xg   = X[mask]
            eg   = resid[mask]
            sg   = Xg.T @ eg
            meat += np.outer(sg, sg)
        # Small-sample correction: (G / (G-1)) * (n-1) / (n-k)
        meat *= (G / (G - 1)) * ((n - 1) / (n - k))
        V  = XtXi @ meat @ XtXi
        se_label = f'SE (CRVE, G={G})'
    else:
        # HC1 sandwich
        Xe   = X * resid[:, None]
        meat = Xe.T @ Xe
        V    = XtXi @ meat @ XtXi * (n / (n - k))
        se_label = 'SE (HC1)'

    se = np.sqrt(np.diag(V))

    idx = [
        'Intercept  (C baseline mean)',
        'T          (arm difference at baseline)',
        'wave       (common time trend)',
        'T × wave   (DiD = ITT estimate)',
    ]
    return pd.DataFrame({
        'Coef':      beta,
        se_label:    se,
        't-stat':    beta / se,
        '95% CI lo': beta - 1.96 * se,
        '95% CI hi': beta + 1.96 * se,
    }, index=idx)


# ── Covariate balance ─────────────────────────────────────────────────────────

def balance_tests(df0: pd.DataFrame, w_cols: list[str],
                  labels: dict[str, str] | None = None,
                  smd_threshold: float = 0.1) -> pd.DataFrame:
    """Two-sample t-tests and SMDs for all covariates at baseline.

    Parameters
    ----------
    df0           : baseline-only DataFrame (wave == 0).
    w_cols        : list of covariate column names to test.
    labels        : optional dict mapping column name → display label.
    smd_threshold : |SMD| threshold for the practical-balance flag (default 0.1).

    Returns a DataFrame indexed by variable with columns:
        C mean, T mean, Diff (T−C), SMD, p-value, p-balanced, smd-balanced.

    Two balance flags are reported separately because with n > 1,000 even
    trivially small differences are statistically significant (p-value flag),
    while |SMD| < 0.1 captures practical importance regardless of sample size.
    Note: PMT score typically shows a huge SMD (≈ −3) despite a tiny absolute
    difference because all study households are compressed near the eligibility
    threshold, making the pooled SD near-zero.
    """
    rows = []
    for col in w_cols:
        s0 = df0.loc[df0['T'] == 0, col].dropna()
        s1 = df0.loc[df0['T'] == 1, col].dropna()
        _, pval   = stats.ttest_ind(s0, s1, equal_var=False)
        pooled_sd = np.sqrt((s0.var() + s1.var()) / 2)
        smd       = (s1.mean() - s0.mean()) / pooled_sd if pooled_sd > 0 else 0.0
        name = (labels or {}).get(col, col.replace('_', ' '))
        rows.append({
            'variable':     name,
            'C mean':       round(s0.mean(), 3),
            'T mean':       round(s1.mean(), 3),
            'Diff (T−C)':   round(s1.mean() - s0.mean(), 3),
            'SMD':          round(smd, 3),
            'p-value':      round(pval, 3),
            'p-bal':        '✓' if pval > 0.05 else '✗',
            'smd-bal':      '✓' if abs(smd) < smd_threshold else '✗',
        })
    return pd.DataFrame(rows).set_index('variable')
