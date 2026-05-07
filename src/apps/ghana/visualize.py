"""Ghana LEAP 1000 — visualization utilities."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import rasterio
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

    # One colour per trial district
    DISTRICT_COLORS = {
        'East Mamprusi': '#e07b39',
        'Karaga':        '#5b8db8',
        'Bongo':         '#4caf7d',
        'Yendi':         '#9c6db7',
        'Garu-Tempane':  '#d4a017',
    }
    # GADM NAME_2 → study district label
    GADM_TO_DISTRICT = {
        'EastMamprusi': 'East Mamprusi',
        'Karaga':       'Karaga',
        'Bongo':        'Bongo',
        'Yendi':        'Yendi',
        'Garu':         'Garu-Tempane',
        'Tempane':      'Garu-Tempane',
    }
    label_offsets = {
        'East Mamprusi': (-0.55, -0.18),
        'Karaga':        (-0.65, -0.15),
        'Bongo':         (-0.72,  0.12),
        'Yendi':         ( 0.15, -0.20),
        'Garu-Tempane':  ( 0.00,  0.38),
    }

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 5))

    if paper:
        ax.set_facecolor(PAPER_BG)
        ax.figure.set_facecolor(PAPER_BG)

    # All regions in gray background first
    gdf1.plot(ax=ax, color='#d9d9d9', edgecolor='white', linewidth=0.6)

    # Trial districts: each in its own colour on top
    gdf2['district_label'] = gdf2['NAME_2'].map(GADM_TO_DISTRICT)
    for label, color in DISTRICT_COLORS.items():
        mask = gdf2['district_label'] == label
        if mask.any():
            gdf2[mask].plot(ax=ax, color=color, alpha=0.35,
                            edgecolor='white', linewidth=0.6)

    if not lakes_gh.empty:
        lakes_gh.plot(ax=ax, color='#a8d0e6', edgecolor='#7ab0cb', lw=0.5, zorder=3)

    # Crop to Ghana's actual extent with minimal padding
    minx, miny, maxx, maxy = gdf1.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # District name labels (no circle markers — names are district labels, not cities)
    for label, color in DISTRICT_COLORS.items():
        mask = gdf2['district_label'] == label
        if mask.any():
            pt = gdf2[mask].geometry.unary_union.centroid
        dx, dy = label_offsets[label]
        ax.annotate(label,
                    xy=(pt.x, pt.y), xytext=(pt.x + dx, pt.y + dy),
                    fontsize=14, color='#222222',
                    arrowprops=dict(arrowstyle='-', color='#666666', lw=0.6),
                    va='center', ha='right' if dx < 0 else ('center' if dx == 0 else 'left'))

    if df is not None:
        # Community centroids coloured by district; one point per comm.
        comm_df = (
            df[df['wave'] == 0]
            .dropna(subset=['gps_latitude', 'gps_longitude'])
            .groupby('comm', as_index=False)
            .agg(lat=('gps_latitude', 'first'),
                 lon=('gps_longitude', 'first'),
                 district=('district', 'first'))
        )
        for label, color in DISTRICT_COLORS.items():
            sub = comm_df[comm_df['district'] == label]
            ax.scatter(sub['lon'], sub['lat'],
                       s=8, c=color, marker='o',
                       edgecolors='none', alpha=0.9, zorder=6)

    ax.set_facecolor(PAPER_BG if paper else 'white')
    ax.figure.set_facecolor(PAPER_BG if paper else 'white')
    ax.axis('off')
    return ax


def plot_neuron_activation_map(
    data_dir: Path | str,
    df: 'pd.DataFrame',
    comm_ids: 'np.ndarray',
    activations: 'np.ndarray',
    neuron_idx: int,
    pvalue: float,
    ax=None,
    cmap: str = "YlOrRd",
    inactive_color: str = "#cccccc",
) -> plt.Axes:
    """Map of Ghana LEAP communities coloured by SAE neuron activation.

    Active communities (activation > 0) are coloured on a sequential scale;
    inactive communities are rendered in gray.

    Parameters
    ----------
    df         : DataFrame from load_data(); must contain comm, gps_latitude,
                 gps_longitude, wave columns.
    comm_ids   : (n_comm,) array of community IDs aligned with `activations`.
    activations: (n_comm,) activation values for this neuron.
    """
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable

    data_dir = Path(data_dir)
    gdf1 = gpd.read_file(data_dir / 'gadm41_GHA_1.json')
    gdf2 = gpd.read_file(data_dir / 'gadm41_GHA_2.json')

    lakes_shp = data_dir / 'ne_10m_lakes.shp'
    if lakes_shp.exists():
        ghana_box = gpd.GeoDataFrame({'geometry': [box(*gdf1.total_bounds)]}, crs=gdf1.crs)
        lakes_gh  = gpd.read_file(lakes_shp).to_crs(gdf1.crs).clip(ghana_box)
    else:
        lakes_gh = gpd.GeoDataFrame()

    GADM_TO_DISTRICT = {
        'EastMamprusi': 'East Mamprusi', 'Karaga': 'Karaga',
        'Bongo': 'Bongo', 'Yendi': 'Yendi',
        'Garu': 'Garu-Tempane', 'Tempane': 'Garu-Tempane',
    }

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 5))

    # Base map
    gdf1.plot(ax=ax, color='#e8e8e8', edgecolor='white', linewidth=0.6)
    gdf2['district_label'] = gdf2['NAME_2'].map(GADM_TO_DISTRICT)
    trial_mask = gdf2['district_label'].notna()
    gdf2[trial_mask].plot(ax=ax, color='#f0f0f0', edgecolor='white', linewidth=0.6)
    if not lakes_gh.empty:
        lakes_gh.plot(ax=ax, color='#a8d0e6', edgecolor='#7ab0cb', lw=0.5, zorder=3)

    # Community activations
    act_map = dict(zip(comm_ids.tolist(), activations.tolist()))
    comm_df = (
        df[df['wave'] == 0]
        .dropna(subset=['gps_latitude', 'gps_longitude'])
        .groupby('comm', as_index=False)
        .agg(lat=('gps_latitude', 'first'), lon=('gps_longitude', 'first'))
    )
    comm_df['activation'] = comm_df['comm'].map(act_map).fillna(0.0)

    active   = comm_df[comm_df['activation'] > 0]
    inactive = comm_df[comm_df['activation'] <= 0]

    # Inactive communities in gray
    if not inactive.empty:
        ax.scatter(inactive['lon'], inactive['lat'],
                   s=14, c=inactive_color, marker='o',
                   edgecolors='none', alpha=0.7, zorder=5, label='Inactive')

    # Active communities on a colormap
    if not active.empty:
        vmin, vmax = active['activation'].min(), active['activation'].max()
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sc = ax.scatter(active['lon'], active['lat'],
                        s=20, c=active['activation'], cmap=cmap, norm=norm,
                        marker='o', edgecolors='#333333', linewidths=0.3,
                        alpha=0.95, zorder=6)
        cbar = ax.figure.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=ax, fraction=0.03, pad=0.02, aspect=20
        )
        cbar.set_label('Activation', fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    n_active = int((activations > 0).sum())
    ax.set_title(
        f"Neuron {neuron_idx}  |  p = {pvalue:.4f}  |  "
        f"{n_active}/{len(comm_ids)} active communities",
        fontsize=8, pad=4,
    )
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


def show_neuron(
    fi: int,
    act: np.ndarray,
    live_idx: np.ndarray,
    sae_ids: np.ndarray,
    tif_dir: Path | str,
    comm_district: 'pd.Series',
    k: int = 5,
    ax_row=None,
) -> np.ndarray:
    """Show false-colour satellite chips for the top-k activated communities.

    Parameters
    ----------
    fi            : filtered neuron index (column index into `act`)
    act           : (n_comms, n_live) activation matrix (live neurons only)
    live_idx      : mapping from filtered index → full SAE neuron index
    sae_ids       : community IDs corresponding to act rows
    tif_dir       : directory containing ghana_comm{id:04d}.tif files
    comm_district : Series mapping comm id → district name
    k             : number of top communities to display

    Returns
    -------
    Array of the top-k community IDs shown.
    """
    tif_dir = Path(tif_dir)
    col     = act[:, fi]
    top_k   = np.argsort(col)[::-1][:k]
    comm_ids_top = sae_ids[top_k]
    acts_top     = col[top_k]

    if ax_row is None:
        _, axes = plt.subplots(1, k, figsize=(3 * k, 3))
    else:
        axes = ax_row

    for ax, comm_id, act_val in zip(axes, comm_ids_top, acts_top):
        tif_path = tif_dir / f'ghana_comm{int(comm_id):04d}.tif'
        with rasterio.open(tif_path) as src:
            r = src.read(4).astype(float)  # NIR
            g = src.read(2).astype(float)  # Green
            b = src.read(6).astype(float)  # SWIR2

        def _norm(band):
            lo, hi = (np.percentile(band[band > 0], [2, 98])
                      if (band > 0).any() else (0.0, 1.0))
            return np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)

        rgb = np.stack([_norm(r), _norm(g), _norm(b)], axis=-1)
        ax.imshow(rgb)
        district = comm_district.get(int(comm_id), '?')
        ax.set_title(f'comm {int(comm_id)}\n{district}\nact={act_val:.3f}', fontsize=7)
        ax.axis('off')

    return comm_ids_top


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
