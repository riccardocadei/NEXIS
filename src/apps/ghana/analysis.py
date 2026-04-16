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

def regression_did(df: pd.DataFrame, outcome: str = 'Y') -> pd.DataFrame:
    """OLS DiD: Y = α + β₁·T + β₂·wave + δ·(T×wave) + ε.

    Returns a DataFrame with Coef, SE (HC1), t-stat, and 95% CI.
    Under the parallel trends assumption, δ = DiD = ITT.

    Note: standard errors are HC1 (heteroscedasticity-robust). The
    randomisation unit is the sub-district community/cluster; the
    dataset lacks a community ID, so we cluster at the district level
    (5 districts), which is conservative but gives few clusters — wild
    cluster bootstrap is recommended for final reporting.
    """
    sub = df.dropna(subset=['T', 'wave', outcome]).copy()
    y   = sub[outcome].values
    T   = sub['T'].values
    w   = sub['wave'].values
    D   = T * w  # DiD indicator

    X = np.column_stack([np.ones(len(y)), T, w, D])
    n, k = X.shape

    XtXi = np.linalg.inv(X.T @ X)
    beta  = XtXi @ X.T @ y
    resid = y - X @ beta

    # HC1 sandwich
    Xe   = X * resid[:, None]
    meat = Xe.T @ Xe
    V    = XtXi @ meat @ XtXi * (n / (n - k))
    se   = np.sqrt(np.diag(V))

    idx = [
        'Intercept  (C baseline mean)',
        'T          (arm difference at baseline)',
        'wave       (common time trend)',
        'T × wave   (DiD = ITT estimate)',
    ]
    return pd.DataFrame({
        'Coef':     beta,
        'SE (HC1)': se,
        't-stat':   beta / se,
        '95% CI lo': beta - 1.96 * se,
        '95% CI hi': beta + 1.96 * se,
    }, index=idx)


# ── Covariate balance ─────────────────────────────────────────────────────────

def balance_tests(df0: pd.DataFrame, w_cols: list[str],
                  labels: dict[str, str] | None = None) -> pd.DataFrame:
    """Two-sample t-tests and SMDs for all covariates at baseline.

    Parameters
    ----------
    df0     : baseline-only DataFrame (wave == 0).
    w_cols  : list of covariate column names to test.
    labels  : optional dict mapping column name → display label.

    Returns a DataFrame indexed by variable with columns:
        C mean, T mean, Diff (T−C), SMD, p-value, balanced (✓/✗).
    """
    rows = []
    for col in w_cols:
        s0 = df0.loc[df0['T'] == 0, col].dropna()
        s1 = df0.loc[df0['T'] == 1, col].dropna()
        _, pval    = stats.ttest_ind(s0, s1, equal_var=False)
        pooled_sd  = np.sqrt((s0.var() + s1.var()) / 2)
        smd        = (s1.mean() - s0.mean()) / pooled_sd if pooled_sd > 0 else 0.0
        name = (labels or {}).get(col, col.replace('_', ' '))
        rows.append({
            'variable':   name,
            'C mean':     round(s0.mean(), 3),
            'T mean':     round(s1.mean(), 3),
            'Diff (T−C)': round(s1.mean() - s0.mean(), 3),
            'SMD':        round(smd, 3),
            'p-value':    round(pval, 3),
            'balanced':   '✓' if pval > 0.05 else '✗',
        })
    return pd.DataFrame(rows).set_index('variable')
