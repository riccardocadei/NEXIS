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
from shapely.geometry import box

# ── Constants ─────────────────────────────────────────────────────────────────
FILL_THRESHOLD = 5000
PCT_LO, PCT_HI = 2, 98


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
    rows = df[df['geo_long_lat_key'] == key].dropna(subset=['PNAME_VALUE'])
    if len(rows):
        r = rows.iloc[0]
        return f"{r['PNAME_VALUE']}\n({r['geo_long_center']:.2f}°E, {r['geo_lat_center']:.2f}°N)"
    return f'Key {key}'


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
