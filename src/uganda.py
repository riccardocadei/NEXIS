"""Uganda YOP helper functions — shared across notebooks."""

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from geodatasets import get_path
from scipy import stats
from scipy.linalg import lstsq
from shapely.geometry import box

# ── Constants ─────────────────────────────────────────────────────────────────
FILL_THRESHOLD = 5000
PCT_LO, PCT_HI = 2, 98

# Language group codes → names (Blattman et al. 2014 — northern Uganda YOP)
# Verify against the original codebook if in doubt.
LANG_NAMES = {
    1: 'Acholi',
    2: 'Lango',
    3: 'Teso',
    4: 'Karamojong',
    5: 'Madi',
    6: 'Lugbara',
    7: 'Other',
}

# Outcome aliases: clean name → CSV column in UgandaDataProcessed.csv
OUTCOME_ALIASES: dict[str, str] = {
    # ── Labour outcomes (Jerzak et al. 2023 / Blattman et al. 2014 Table III) ──
    "log_skilled_hours":  "Yobs",                    # log(skilled-labor hrs/wk + 100) — primary outcome
    "skilled_employed":   "skilled_dummy_e",          # any skilled trade engagement (binary)
    "skilled_fulltime":   "fulltimeskill_e",          # ≥30 hrs/week in skilled trade (binary)
    "employ_hours":       "employhours_e",            # total employment hours/week (missing at endline)
    "log_training_hours": "training_hours_ln_e",      # log vocational training hours
    # ── Economic outcomes (Blattman et al. 2014 Tables IV & VI) ────────────────
    "log_earnings":       "profits4w_real_ln_e",      # log real 4-week cash earnings
    "log_biz_assets":     "bizasset_val_real_ln_e",   # log real business asset value
    "wealth_index":       "wealthindex_e",            # household wealth/durable assets index
    "wellbeing":          "wealthladder_e",           # subjective wellbeing ladder (1–9)
}

ALL_OUTCOMES = list(OUTCOME_ALIASES)

# Clean display names for outcomes (used in plot titles, narrative headers, etc.)
OUTCOME_DISPLAY: dict[str, str] = {
    "log_skilled_hours":  "log skilled labor hours",
    "skilled_employed":   "skilled employment (any)",
    "skilled_fulltime":   "full-time skilled employment",
    "employ_hours":       "total employment hours",
    "log_training_hours": "log vocational training hours",
    "log_earnings":       "log cash earnings",
    "log_biz_assets":     "log business asset value",
    "wealth_index":       "household wealth index",
    "wellbeing":          "subjective wellbeing",
}


def outcome_display(name: str) -> str:
    """Return a clean human-readable label for an outcome alias.

    Falls back to replacing underscores with spaces if the alias is unknown.
    """
    return OUTCOME_DISPLAY.get(name, name.replace("_", " "))

# ── W covariate display helpers ───────────────────────────────────────────────

_STATIC_W = {
    'age':          ('Age',                '< 30',       '≥ 30'),
    'female':       ('Sex',                'Male',       'Female'),
    'father_educ':  ("Father's education", 'Low / Med',  'High'),
    'mother_educ':  ("Mother's education", 'Low / Med',  'High'),
    'group_female': ('Female-only group',  'No',         'Yes'),
}


def w_display(label: str) -> tuple[str, str, str]:
    """Map a raw W column name to (display_name, tick_lo, tick_hi).

    Used for axis labels and tick annotations in GATE/CATE plots.
    """
    if label in _STATIC_W:
        return _STATIC_W[label]
    if label.startswith('lang_'):
        try:
            code = int(label.split('_')[1])
            lang = LANG_NAMES.get(code, label)
            if str(lang).lower() == 'other':
                return ('Minor local language', 'Not minor local language', 'Minor local language')
            return (f'{lang} language', f'Non-{lang} language', f'{lang} language')
        except ValueError:
            pass
    if label.startswith('district_'):
        dist = label.replace('district_', '').title()
        return (f'{dist} district', 'Other', dist)
    return (label.replace('_', ' '), '= 0', '= 1')


def resolve_outcome(name: str) -> str:
    """Return the CSV column name for an outcome alias (pass-through if already a column name)."""
    return OUTCOME_ALIASES.get(name, name)


# ── Satellite imagery ─────────────────────────────────────────────────────────

def load_image(key, img_dir):
    """Load a 3-band Landsat image as a false-color (NIR→R, Green→G, SWIR→B)
    float32 array of shape (H, W, 3) in [0, 1].  Returns None if files missing."""
    raw = {}
    for b in [1, 2, 3]:
        path = Path(img_dir) / f'GeoKey{key}_BAND{b}.csv'
        if not path.exists():
            return None
        arr = pd.read_csv(path, header=None).values.astype(np.float32)
        arr[arr >= FILL_THRESHOLD] = np.nan
        raw[b] = arr

    out = []
    for b in [2, 1, 3]:          # NIR → R, Green → G, SWIR → B
        ch    = raw[b]
        valid = ch[~np.isnan(ch)]
        lo    = np.percentile(valid, PCT_LO)
        hi    = np.percentile(valid, PCT_HI)
        s     = np.clip((ch - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        s[np.isnan(ch)] = 0.0
        out.append(s)
    return np.stack(out, axis=-1)


def geo_label(key, df):
    """Parish name + coordinates for a given geo_long_lat_key."""
    rows = df[df['geo_long_lat_key'] == key]
    if not len(rows):
        return f'Key {key}'
    r = rows.iloc[0]
    coords = f"({r['geo_long_center']:.2f}°E, {r['geo_lat_center']:.2f}°N)"
    if 'PNAME_VALUE' in df.columns:
        named = rows.dropna(subset=['PNAME_VALUE'])
        if len(named):
            return f"{named.iloc[0]['PNAME_VALUE']}\n{coords}"
    return f'Key {key}\n{coords}'


# ── Mapping ───────────────────────────────────────────────────────────────────

def load_basemap(map_data_dir, bbox=(29.0, -1.5, 35.5, 4.8)):
    """Download (once) and return (uganda_gdf, neighbors, lakes_c) for draw_base."""
    map_data_dir = Path(map_data_dir)
    map_data_dir.mkdir(exist_ok=True)

    ne_shp = map_data_dir / 'ne_10m_admin_0_countries.shp'
    if not ne_shp.exists():
        url = 'https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip'
        data = urllib.request.urlopen(url, timeout=60).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                if Path(name).suffix in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
                    (map_data_dir / Path(name).name).write_bytes(z.read(name))

    world      = gpd.read_file(ne_shp).to_crs('EPSG:4326')
    uganda_gdf = world[world['NAME'] == 'Uganda']
    neighbors  = world[world.geometry.intersects(
                     uganda_gdf.union_all().buffer(0.3)) & (world['NAME'] != 'Uganda')]

    lakes   = gpd.read_file(get_path('naturalearth.lakes'))
    bbox_gdf = gpd.GeoDataFrame({'geometry': [box(*bbox)]}, crs='EPSG:4326')
    lakes_c  = lakes.clip(bbox_gdf)

    return uganda_gdf, neighbors, lakes_c


def draw_base(ax, uganda_gdf, neighbors, lakes_c,
              xlim=(29.1, 35.4), ylim=(-1.4, 4.6)):
    """Draw Uganda + neighbours + Lake Victoria on ax."""
    neighbors.plot(ax=ax, color='#e8e4dc', edgecolor='#bbb', lw=0.6, zorder=1)
    uganda_gdf.plot(ax=ax, color='#f5f0e8', edgecolor='#444', lw=1.4, zorder=2)
    lakes_c.plot(ax=ax, color='#a8d0e6', edgecolor='#7ab0cb', lw=0.5, zorder=3)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('Longitude', fontsize=9); ax.set_ylabel('Latitude', fontsize=9)
    ax.grid(alpha=0.2, ls='--', lw=0.5); ax.tick_params(labelsize=8)


# ── Causal inference helpers ──────────────────────────────────────────────────

def _cluster_ols(y, X, groups):
    """OLS with cluster-robust covariance (small-sample corrected)."""
    n, k    = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta    = XtX_inv @ X.T @ y
    resid   = y - X @ beta
    G       = len(np.unique(groups))
    meat    = sum(X[groups == g].T @ np.outer(resid[groups == g],
                                               resid[groups == g])
                  @ X[groups == g] for g in np.unique(groups))
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    return beta, V, G


def balance_pvalue(w, t, groups, categorical=False):
    """Cluster-robust balance test: returns (p_value, test_label).

    Numeric w  → OLS w ~ 1+T, t-test on T coefficient.
    Categorical → OLS T ~ dummies(w), joint F-test.
    """
    if categorical:
        dummies = pd.get_dummies(w, drop_first=True).values.astype(float)
        X       = np.column_stack([np.ones(len(t)), dummies])
        beta, V, G = _cluster_ols(t.astype(float), X, groups)
        beta_r, V_r = beta[1:], V[1:, 1:]
        k     = X.shape[1]
        F     = (beta_r @ np.linalg.inv(V_r) @ beta_r) / (k - 1)
        pval  = float(stats.f.sf(F, dfn=k - 1, dfd=G - 1))
        label = f'F cluster-robust (G={G})'
    else:
        X = np.column_stack([np.ones(len(w)), t])
        beta, V, G = _cluster_ols(w.astype(float), X, groups)
        t_stat = beta[1] / np.sqrt(V[1, 1])
        pval   = float(2 * stats.t.sf(abs(t_stat), df=G - 1))
        label  = f't cluster-robust (G={G})'
    return pval, label


def compute_cate(df_sub, col, label_map=None):
    """Raw subgroup CATEs with 95% CI. Returns DataFrame with label/cate/se/n1/n0."""
    rows = []
    for grp, gdf in df_sub.dropna(subset=[col]).groupby(col, observed=True):
        y1 = gdf.loc[gdf['T'] == 1, 'Y'].dropna()
        y0 = gdf.loc[gdf['T'] == 0, 'Y'].dropna()
        if len(y1) < 2 or len(y0) < 2:
            continue
        cate  = y1.mean() - y0.mean()
        se    = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
        label = label_map.get(grp, str(grp)) if label_map else str(grp)
        rows.append({'label': label, 'cate': cate, 'se': se,
                     'n1': len(y1), 'n0': len(y0)})
    return pd.DataFrame(rows)


def plot_districts(df, uganda_gdf, neighbors, lakes_c, ax=None):
    """Plot experimental sites colored by district on the Uganda basemap."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    need_cols = ['geo_long_center', 'geo_lat_center', 'district']
    sites = (
        df.dropna(subset=need_cols)
          .drop_duplicates('geo_long_lat_key')
          [['geo_long_lat_key', 'geo_long_center', 'geo_lat_center', 'district']]
          .copy()
    )

    districts = sorted(sites['district'].unique())
    cmap    = cm.get_cmap('tab20', len(districts))
    d_color = {d: cmap(i) for i, d in enumerate(districts)}

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    draw_base(ax, uganda_gdf, neighbors, lakes_c)
    for d, grp in sites.groupby('district'):
        ax.scatter(grp['geo_long_center'], grp['geo_lat_center'],
                   color=d_color[d], s=30, label=d, zorder=5,
                   edgecolors='white', linewidths=0.3, alpha=0.9)
        cx, cy = grp['geo_long_center'].mean(), grp['geo_lat_center'].mean()
        ax.text(cx, cy, d, fontsize=5.5, ha='center', va='bottom',
                color=d_color[d], zorder=6,
                bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.5, lw=0))

    return ax


def clean_w(df, educ_high_thresh=5, include_district=False):
    """Return a DataFrame of binarized pre-treatment covariates (all binary / one-hot).

    Columns produced:
        age_30plus         — 1 if age >= 30, else 0
        female             — as-is (already binary at individual level)
        father_educ_high   — 1 if father_educ >= educ_high_thresh (default 5 = secondary+)
        mother_educ_high   — 1 if mother_educ >= educ_high_thresh
        lang_{name}        — one-hot per language group using LANG_NAMES
        district_{name}    — one-hot per district (only if include_district=True)

    group_female and karamojan_district are discarded.
    """
    out = pd.DataFrame(index=df.index)

    if 'age' in df.columns:
        out['age_30plus'] = (df['age'] >= 30).astype(int)

    if 'female' in df.columns:
        out['female'] = df['female'].fillna(0).astype(int)

    if 'father_educ' in df.columns:
        out['father_educ_high'] = (df['father_educ'] >= educ_high_thresh).astype(int)

    if 'mother_educ' in df.columns:
        out['mother_educ_high'] = (df['mother_educ'] >= educ_high_thresh).astype(int)

    if 'lang_group' in df.columns:
        lang = df['lang_group'].astype(float)
        for code, name in LANG_NAMES.items():
            out[f'lang_{name.lower()}'] = (lang == float(code)).astype(int)

    if include_district and 'district' in df.columns:
        dummies = pd.get_dummies(df['district'], prefix='district').astype(int)
        out = pd.concat([out, dummies], axis=1)

    return out


def plot_cate_panels(df, w_panels, treat_color='#e07b39', ctrl_color='#5b8db8',
                     suptitle='Conditional ATEs by pre-treatment covariate (W)'):
    """Bar-chart CATE panels for a list of (title, col, label_map) specs.

    df must contain columns T, Y, and all columns referenced in w_panels.
    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    sub = df.dropna(subset=['Y', 'T']).copy()
    ctrl_y = sub.loc[sub['T'] == 0, 'Y']
    trt_y  = sub.loc[sub['T'] == 1, 'Y']
    ate     = trt_y.mean() - ctrl_y.mean()
    pct_ate = (np.exp(ate) - 1) * 100

    ncols = 4
    nrows = max(1, (len(w_panels) + ncols - 1) // ncols)
    fig, axes_arr = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.8 * nrows))
    axes_list = np.array(axes_arr).flatten().tolist()

    for ax_idx, (title, col, lmap) in enumerate(w_panels):
        ax = axes_list[ax_idx]
        if col not in sub.columns:
            ax.set_visible(False)
            continue
        g = compute_cate(sub, col, lmap)
        if g.empty:
            ax.set_visible(False)
            continue

        ci     = 1.96 * g['se']
        colors = [treat_color if v >= 0 else ctrl_color for v in g['cate']]
        x_pos  = np.arange(len(g))
        ax.bar(x_pos, g['cate'], color=colors, alpha=0.82, edgecolor='white',
               yerr=ci, capsize=5, error_kw={'ecolor': '#333', 'lw': 1.2, 'capthick': 1.2})
        ax.axhline(ate, color='#7b5ea7', lw=1.4, ls='--',
                   label=f'ATE={ate:+.3f} ({pct_ate:+.0f}%)')
        ax.axhline(0, color='black', lw=0.8)
        for xi, (_, row) in enumerate(g.iterrows()):
            pct = (np.exp(row['cate']) - 1) * 100
            off = row['se'] * 1.96 + 0.012
            ax.text(xi, row['cate'] + off, f'{pct:+.0f}%',
                    ha='center', va='bottom', fontsize=7.5, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(g['label'], rotation=20, ha='right', fontsize=9)
        ax.set_title(f'CATE by {title}', fontsize=10)
        ax.set_ylabel('ATE (log hrs)')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    for ax in axes_list[len(w_panels):]:
        ax.set_visible(False)

    plt.suptitle(suptitle + '\nerror bars = 95% CI  ·  labels ≈ % change in skilled-labor hours',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_languages(df, uganda_gdf, neighbors, lakes_c, ax=None):
    """Plot experimental sites colored by language group on the Uganda basemap.

    Each site's language is the modal lang_group among its individuals (7 sites
    have 2 language groups present; the rest are unambiguous).  A text label for
    each language group is placed at the centroid of its sites.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    need_cols = ['geo_long_center', 'geo_lat_center', 'lang_group']
    sites_all = df.dropna(subset=need_cols).copy()
    sites_all['lang_group'] = sites_all['lang_group'].astype(float)

    # Modal language per site (handles the ~7 mixed sites gracefully)
    modal = (
        sites_all.groupby('geo_long_lat_key')['lang_group']
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
        .rename(columns={'lang_group': 'lang_modal'})
    )
    sites = (
        sites_all.drop_duplicates('geo_long_lat_key')
        [['geo_long_lat_key', 'geo_long_center', 'geo_lat_center']]
        .merge(modal, on='geo_long_lat_key')
    )
    sites['lang_name'] = sites['lang_modal'].map(LANG_NAMES).fillna('Unknown')

    lang_order = [LANG_NAMES[k] for k in sorted(LANG_NAMES)]
    cmap    = cm.get_cmap('Set2', len(lang_order))
    l_color = {name: cmap(i) for i, name in enumerate(lang_order)}

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    draw_base(ax, uganda_gdf, neighbors, lakes_c)
    for lang, grp in sites.groupby('lang_name'):
        ax.scatter(grp['geo_long_center'], grp['geo_lat_center'],
                   color=l_color.get(lang, 'grey'), s=30, label=lang, zorder=5,
                   edgecolors='white', linewidths=0.3, alpha=0.9)
        cx = grp['geo_long_center'].mean()
        cy = grp['geo_lat_center'].mean()
        ax.text(cx, cy, lang, fontsize=6.5, ha='center', va='bottom', fontweight='bold',
                color=l_color.get(lang, 'grey'), zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.6, lw=0))

    ax.legend(title='Language', fontsize=7, title_fontsize=8,
              loc='lower right', framealpha=0.85)
    return ax


def w_interaction_tests(df, W_df=None, alpha=0.05):
    """Marginal T×W interaction tests for all W covariates.

    For each W_k tests H0: γ=0 in Y ~ 1 + T + W_k + T*W_k via OLS t-test.

    Parameters
    ----------
    df   : DataFrame with columns T, Y (and raw W if W_df is None).
    W_df : Optional pre-built W DataFrame (e.g. from clean_w()).  When supplied,
           df is only used for T and Y.  When None, falls back to the raw W
           variables (age, female, father_educ, mother_educ, group_female,
           karamojan_district, lang_group one-hots, district one-hots).

    Returns a DataFrame sorted by p-value with columns:
        W, γ (T×W), p-value, sig (bool, Bonferroni-corrected at alpha / n_cols).
    """
    sub = df.dropna(subset=['Y', 'T']).copy()

    if W_df is not None:
        W_df = W_df.loc[sub.index].astype(float)
    else:
        numeric_w = ['age', 'female', 'father_educ', 'mother_educ', 'group_female']
        numeric_w = [c for c in numeric_w if c in sub.columns]
        cat_dummies = pd.concat(
            [pd.get_dummies(sub[c], prefix=c).astype(float)
             for c in ('lang_group', 'district') if c in sub.columns],
            axis=1,
        )
        W_df = pd.concat([sub[numeric_w].astype(float), cat_dummies], axis=1)

    mask  = ~np.isnan(W_df.values).any(axis=1)
    Y     = sub['Y'].values[mask]
    T     = sub['T'].values[mask]
    W_arr = W_df.values[mask]

    gate = alpha / W_arr.shape[1]
    rows = []
    for j, name in enumerate(W_df.columns):
        w = W_arr[:, j]
        w = (w - w.mean()) / (w.std() + 1e-12)
        X = np.column_stack([np.ones(len(Y)), T, w, T * w])
        beta, _, _, _ = lstsq(X, Y)
        resid = Y - X @ beta
        s2    = (resid @ resid) / (len(Y) - 4)
        se    = np.sqrt(s2 * np.diag(np.linalg.pinv(X.T @ X)))
        t     = beta[3] / (se[3] + 1e-300)
        p     = float(2 * stats.t.sf(abs(t), df=len(Y) - 4))
        rows.append({'W': name, 'γ (T×W)': float(beta[3]), 'p-value': p, 'sig': p <= gate})

    return pd.DataFrame(rows).sort_values('p-value'), gate
