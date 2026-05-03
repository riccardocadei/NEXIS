"""Ghana LEAP 1000 — visualization utilities."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union

TREAT_COLOR = '#e07b39'
CTRL_COLOR  = '#5b8db8'
PAPER_BG    = '#FFF5EB'  # orange!8!white: 8%*(1,0.5,0) + 92%*(1,1,1)


def plot_ghana_map(data_dir: Path | str, ax=None, paper: bool = False,
                   df: 'pd.DataFrame | None' = None) -> plt.Axes:
    """Draw the Ghana LEAP 1000 study-area map.

    Highlights the trial regions (Northern / NorthEast / Upper East) and
    marks the five trial districts with annotated points.  If a dataframe
    with community GPS coordinates is supplied (via `df`), community
    centroids are overlaid as scatter points coloured by treatment arm.

    Parameters
    ----------
    paper : bool
        Paper-ready mode: no title, background set to orange!8!white.
    df : optional DataFrame returned by load_data(); must contain
        gps_latitude, gps_longitude, comm, T columns.  Baseline wave
        (wave == 0) is used to avoid double-counting panel households.
    """
    data_dir = Path(data_dir)
    gdf1 = gpd.read_file(data_dir / 'gadm41_GHA_1.json')
    gdf2 = gpd.read_file(data_dir / 'gadm41_GHA_2.json')

    # 10m lakes clipped to Ghana
    lakes_shp = data_dir / 'ne_10m_lakes.shp'
    if not lakes_shp.exists():
        url = 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes.zip'
        data = urllib.request.urlopen(url, timeout=60).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                if Path(name).suffix in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
                    (data_dir / Path(name).name).write_bytes(z.read(name))
    ghana_box = gpd.GeoDataFrame({'geometry': [box(*gdf1.total_bounds)]}, crs=gdf1.crs)
    lakes_gh  = gpd.read_file(lakes_shp).to_crs(gdf1.crs).clip(ghana_box)

    TRIAL_REGIONS = {'Northern', 'NorthEast', 'UpperEast'}
    gdf1['in_trial'] = gdf1['NAME_1'].isin(TRIAL_REGIONS)

    # Garu-Tempane was split into Garu + Tempane in the GADM dataset
    district_centroids = {
        'East Mamprusi': gdf2[gdf2['NAME_2'] == 'EastMamprusi'].geometry.iloc[0].centroid,
        'Karaga':        gdf2[gdf2['NAME_2'] == 'Karaga'].geometry.iloc[0].centroid,
        'Bongo':         gdf2[gdf2['NAME_2'] == 'Bongo'].geometry.iloc[0].centroid,
        'Yendi':         gdf2[gdf2['NAME_2'] == 'Yendi'].geometry.iloc[0].centroid,
        'Garu-Tempane':  unary_union(
                             gdf2[gdf2['NAME_2'].isin({'Garu', 'Tempane'})].geometry
                         ).centroid,
    }
    label_offsets = {
        'East Mamprusi': (-0.55, -0.18),
        'Karaga':        (-0.65,  0.05),
        'Bongo':         (-0.72,  0.12),
        'Yendi':         ( 0.15, -0.20),
        'Garu-Tempane':  ( 0.15,  0.12),
    }

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 5))

    if paper:
        ax.set_facecolor(PAPER_BG)
        ax.figure.set_facecolor(PAPER_BG)

    gdf1[~gdf1['in_trial']].plot(
        ax=ax, color='#d9d9d9', edgecolor='white', linewidth=0.6)
    gdf1[gdf1['in_trial']].plot(
        ax=ax, color=TREAT_COLOR, alpha=0.45, edgecolor='white', linewidth=0.6)
    if not lakes_gh.empty:
        lakes_gh.plot(ax=ax, color='#a8d0e6', edgecolor='#7ab0cb', lw=0.5, zorder=3)

    for name, pt in district_centroids.items():
        ax.scatter(pt.x, pt.y, s=55, color=TREAT_COLOR,
                   edgecolors='black', linewidths=0.5, zorder=5)
        dx, dy = label_offsets[name]
        ax.annotate(name,
                    xy=(pt.x, pt.y), xytext=(pt.x + dx, pt.y + dy),
                    fontsize=7, color='#222222',
                    arrowprops=dict(arrowstyle='-', color='#666666', lw=0.6),
                    va='center', ha='right' if dx < 0 else 'left')

    if df is not None:
        # Community centroids: one point per comm, using baseline wave only
        comm_df = (
            df[df['wave'] == 0]
            .dropna(subset=['gps_latitude', 'gps_longitude'])
            .groupby('comm', as_index=False)
            .agg(lat=('gps_latitude', 'first'),
                 lon=('gps_longitude', 'first'),
                 T=('T', lambda x: int(x.mode()[0])),
                 n=('T', 'count'))
        )
        for t_val, color, label in [
            (1, '#2ca02c', 'Treatment community'),
            (0, '#d62728', 'Comparison community'),
        ]:
            sub = comm_df[comm_df['T'] == t_val]
            ax.scatter(sub['lon'], sub['lat'],
                       s=8, c=color, marker='o',
                       edgecolors='none',
                       alpha=0.75, zorder=6, label=label)
        ax.legend(fontsize=7, loc='lower left',
                  framealpha=0.8, markerscale=1.2)

    if not paper:
        ax.set_title('LEAP 1000 trial districts', fontsize=11, pad=8)
    ax.axis('off')
    return ax


def plot_love(df0: pd.DataFrame, w_cols: list[str],
              labels: dict[str, str] | None = None,
              ax=None,
              treat_color: str = TREAT_COLOR,
              ctrl_color:  str = CTRL_COLOR) -> plt.Axes:
    """Love plot of standardised mean differences (SMDs) at baseline.

    Parameters
    ----------
    df0    : baseline-only DataFrame (wave == 0).
    w_cols : covariate columns to include.
    labels : optional dict mapping column name → display label.
    """
    rows = []
    for col in w_cols:
        s0 = df0.loc[df0['T'] == 0, col].dropna()
        s1 = df0.loc[df0['T'] == 1, col].dropna()
        pooled_sd = np.sqrt((s0.var() + s1.var()) / 2)
        smd       = (s1.mean() - s0.mean()) / pooled_sd if pooled_sd > 0 else 0.0
        name = (labels or {}).get(col, col.replace('_', ' '))
        rows.append({'variable': name, 'SMD': smd})

    smd_df = pd.DataFrame(rows).sort_values('SMD')

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    colors = [treat_color if abs(v) > 0.1 else ctrl_color for v in smd_df['SMD']]
    ax.barh(smd_df['variable'], smd_df['SMD'], color=colors, alpha=0.8)
    ax.axvline(0,    color='black', lw=0.8)
    ax.axvline( 0.1, color='gray', lw=1.0, ls='--', label='|SMD| = 0.1 threshold')
    ax.axvline(-0.1, color='gray', lw=1.0, ls='--')
    ax.set_xlabel('Standardised Mean Difference (Treatment − Comparison)')
    ax.set_title('Covariate Balance at Baseline (Love Plot)')
    ax.legend(fontsize=9)
    return ax


def plot_parallel_trends(df: pd.DataFrame, outcome: str = 'Y',
                         ylabel: str = 'Mean AE Expenditure (GH₵/month)',
                         ax=None,
                         treat_color: str = TREAT_COLOR,
                         ctrl_color:  str = CTRL_COLOR) -> plt.Axes:
    """The canonical DiD picture: mean outcome by arm and wave.

    Draws two connected points (Baseline → Endline) for each arm and
    annotates the DiD gap at endline.
    """
    means = df.groupby(['wave', 'T'])[outcome].mean().unstack('T')

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.plot([0, 1], means[0].values, 'o--', color=ctrl_color,  lw=2, ms=7, label='Comparison')
    ax.plot([0, 1], means[1].values, 'o-',  color=treat_color, lw=2, ms=7, label='Treatment')

    # Annotate the DiD gap
    did = (means[1][1] - means[1][0]) - (means[0][1] - means[0][0])
    gap_mid = (means[0][1] + means[1][1]) / 2
    ax.annotate(
        f'DiD ≈ {did:+.0f} GH₵',
        xy=(1, means[1][1]), xytext=(0.75, gap_mid + 5),
        fontsize=9,
        arrowprops=dict(arrowstyle='->', color='#444444', lw=0.8),
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Baseline (2015)', 'Endline (2017)'])
    ax.set_ylabel(ylabel)
    ax.set_title('Parallel Trends Visualisation')
    ax.legend(framealpha=0.9)
    return ax
