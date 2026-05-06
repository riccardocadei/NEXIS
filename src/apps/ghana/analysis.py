"""Ghana LEAP 1000 — DiD estimation, covariate balance, and GATE tests."""

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


# ── Interaction p-value (HC1 or cluster-robust) ───────────────────────────────

def interaction_pval(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    cluster: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """HC1 or cluster-robust p-value for the T×Z interaction.

    Runs OLS: y ~ 1 + T + Z + T*Z and returns (coef, SE, p-value) for T*Z.

    Parameters
    ----------
    cluster : array of group labels for cluster-robust (CRVE) SEs.
              Pass community IDs when Z is a community-level feature so the
              effective sample size reflects 162 communities rather than
              ~2 300 households.  Omit for household-level W features.
    """
    X = np.column_stack([np.ones(len(y)), t, z, t * z])
    try:
        n, k = X.shape
        XtXi  = np.linalg.inv(X.T @ X)
        beta  = XtXi @ X.T @ y
        resid = y - X @ beta
        if cluster is not None:
            groups   = np.asarray(cluster)
            unique_g = np.unique(groups)
            G        = len(unique_g)
            meat     = np.zeros((k, k))
            for g in unique_g:
                mask = groups == g
                sg   = X[mask].T @ resid[mask]
                meat += np.outer(sg, sg)
            meat *= (G / (G - 1)) * ((n - 1) / (n - k))
            V = XtXi @ meat @ XtXi
        else:
            Xe = X * resid[:, None]
            V  = XtXi @ (Xe.T @ Xe) @ XtXi * (n / (n - k))
        tstat = beta[3] / np.sqrt(V[3, 3])
        return float(beta[3]), float(np.sqrt(V[3, 3])), float(2 * stats.t.sf(abs(tstat), df=n - k))
    except Exception:
        return np.nan, np.nan, np.nan


# ── GATE effect-modification table ────────────────────────────────────────────

def gate_modification_table(
    y: np.ndarray,
    t: np.ndarray,
    Z: np.ndarray,
    z_names: list[str] | None = None,
    n_sae: int | None = None,
    binarize: bool = True,
    cluster: np.ndarray | None = None,
) -> pd.DataFrame:
    """For each column of Z, test for effect modification.

    When binarize=True (default):
        Binarize Z then run ΔY ~ 1 + T + Z_bin + T*Z_bin.
        Binarization rule:
          - First `n_sae` columns (SAE codes): active (> 0) vs inactive (= 0).
          - Remaining: above vs below community median.
          - Already-binary columns: used as-is.
        Returns GATE per group + interaction p-value.

    When binarize=False:
        Use continuous Z directly: ΔY ~ 1 + T + Z_j + T*Z_j.
        Returns the T*Z interaction coefficient and its p-value.

    Parameters
    ----------
    y         : (n,) first-difference outcome ΔY
    t         : (n,) treatment (0/1)
    Z         : (n, p) feature matrix
    z_names   : display names for Z columns
    n_sae     : number of leading SAE columns to binarize at 0; rest use median split
    binarize  : if False, skip binarization and use a linear interaction test
    cluster   : (n,) group labels for cluster-robust SEs (CRVE).  Pass community IDs
                when Z is community-level (e.g. SAE activations, spectral indices) so
                the SE reflects 162 community clusters instead of ~2 300 households.
                Omit for household-level W features (HC1 is appropriate there).

    Returns
    -------
    DataFrame sorted by p-value.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    Z = np.asarray(Z, dtype=float)
    p = Z.shape[1]
    names = z_names if z_names is not None else [f'z_{j}' for j in range(p)]
    clust = np.asarray(cluster) if cluster is not None else None

    rows = []
    for j in range(p):
        col = Z[:, j]

        if not binarize:
            coef, se, pval = interaction_pval(y, t, col, cluster=clust)
            rows.append({
                'feature':  names[j],
                'T×Z coef': round(coef, 3) if not np.isnan(coef) else np.nan,
                'SE':       round(se,   3) if not np.isnan(se)   else np.nan,
                'p-value':  round(pval, 4) if not np.isnan(pval) else np.nan,
            })
            continue

        unique_vals = np.unique(col)
        if len(unique_vals) <= 2:
            z_bin     = (col > 0).astype(float)
            threshold = '> 0'
        elif n_sae is not None and j < n_sae:
            z_bin     = (col > 0).astype(float)
            threshold = 'active (> 0)'
        else:
            med       = np.median(col)
            z_bin     = (col > med).astype(float)
            threshold = f'> median ({med:.3g})'

        def _gate(group):
            mask = z_bin == group
            y1 = y[mask & (t == 1)]
            y0 = y[mask & (t == 0)]
            if len(y1) == 0 or len(y0) == 0:
                return np.nan, int(mask.sum())
            return float(y1.mean() - y0.mean()), int(mask.sum())

        gate0, n0 = _gate(0)
        gate1, n1 = _gate(1)
        _, se, pval = interaction_pval(y, t, z_bin, cluster=clust)

        rows.append({
            'feature':   names[j],
            'threshold': threshold,
            'GATE(=0)':  round(gate0, 2) if not np.isnan(gate0) else np.nan,
            'n(=0)':     n0,
            'GATE(=1)':  round(gate1, 2) if not np.isnan(gate1) else np.nan,
            'n(=1)':     n1,
            'diff':      round(gate1 - gate0, 2) if not (np.isnan(gate0) or np.isnan(gate1)) else np.nan,
            'SE':        round(se, 2) if not np.isnan(se) else np.nan,
            'p-value':   round(pval, 4) if not np.isnan(pval) else np.nan,
        })

    return (
        pd.DataFrame(rows)
        .set_index('feature')
        .sort_values('p-value')
    )
