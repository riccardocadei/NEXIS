"""
Generate and save figure_districts for Ghana LEAP 1000.
Labels sit close to clusters, connected by thin leader lines (no arrowhead).
No legend. Thin country border. No lines cross each other.
Full national map with neighbours — same style as Uganda figure_districts.

Usage:
    python src/apps/ghana/figure_maps.py
"""

from pathlib import Path
import io
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

ROOT     = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "ghana"
OUT_DIR  = ROOT / "results" / "ghana" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(ROOT / "src"))
from apps.ghana.data import load_data

plt.rcParams.update({
    "text.usetex":      False,
    "font.family":      "serif",
    "font.serif":       ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
})

# map extents (full Ghana + padding): xlim (-3.4, 1.4), ylim (4.5, 11.4)

# ── District anchors — placed so no lines cross ────────────────────────────────
# Centroids (lon,lat): Bongo(-0.822,10.922) GaruTempane(-0.148,10.838)
#            EastMamprusi(-0.306,10.407) Karaga(-0.469,10.064) Yendi(0.011,9.439)
DIST_ANCHOR = {
    "Bongo":         (-2.0, 10.7),   # left       from (-0.822, 10.922)
    "Garu-Tempane":  ( 0.9, 10.7),   # right      from (-0.148, 10.838)
    "East Mamprusi": ( 1.1, 10.3),   # right      from (-0.306, 10.407)
    "Karaga":        (-2.0, 10.0),   # left-down  from (-0.469, 10.064)
    "Yendi":         ( 0.9,  9.3),   # right-down from ( 0.011,  9.439)
}


def _load_basemap(bbox=(-3.5, 4.3, 1.5, 11.5)):
    """Return (ghana_gdf, neighbors, lakes_c, gdf1) for draw_base."""
    # Natural Earth admin-0 (Ghana outline + neighbours)
    ne_shp = DATA_DIR / 'ne_10m_admin_0_countries.shp'
    if not ne_shp.exists():
        url  = 'https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip'
        data = urllib.request.urlopen(url, timeout=60).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                if Path(name).suffix in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
                    (DATA_DIR / Path(name).name).write_bytes(z.read(name))

    world     = gpd.read_file(ne_shp).to_crs('EPSG:4326')
    ghana_gdf = world[world['NAME'] == 'Ghana']
    neighbors = world[world.geometry.intersects(
                    ghana_gdf.union_all().buffer(0.3)) & (world['NAME'] != 'Ghana')]

    # Lakes clipped to full Ghana bbox
    bbox_gdf = gpd.GeoDataFrame({'geometry': [box(*bbox)]}, crs='EPSG:4326')
    lakes_c  = gpd.read_file(DATA_DIR / 'ne_10m_lakes.shp').to_crs('EPSG:4326').clip(bbox_gdf)

    # GADM level-1 regions (sub-national, same role as Uganda's regions_gdf)
    gdf1 = gpd.read_file(DATA_DIR / 'gadm41_GHA_1.json').to_crs('EPSG:4326')

    return ghana_gdf, neighbors, lakes_c, gdf1


def _draw_base(ax, ghana_gdf, neighbors, lakes_c, regions_gdf,
               xlim=(-3.4, 1.4), ylim=(4.5, 11.4)):
    ax.set_facecolor('white')
    ax.figure.set_facecolor('white')
    # Explicit grey rectangle covering the full map extent (ocean + any gaps)
    ax.add_patch(mpatches.Rectangle(
        (xlim[0], ylim[0]), xlim[1] - xlim[0], ylim[1] - ylim[0],
        facecolor='#e8e4dc', edgecolor='none', zorder=0,
    ))
    neighbors.plot(ax=ax, color='#e8e4dc', edgecolor='#bbb', lw=0.6, zorder=1)
    regions_gdf.plot(ax=ax, color='#f5f0e8', edgecolor='#aaa', lw=0.4, zorder=2)
    ghana_gdf.plot(ax=ax, color='none', edgecolor='#444', lw=1.4, zorder=3)
    if not lakes_c.empty:
        lakes_c.plot(ax=ax, color='#a8d0e6', edgecolor='#7ab0cb', lw=0.5, zorder=4)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')


def _thin_border(ax, ghana_gdf, regions_gdf):
    regions_gdf.plot(ax=ax, color='none', edgecolor='#ccc', lw=0.3, zorder=10)
    ghana_gdf.plot(ax=ax, color='none', edgecolor='#888', lw=0.6, zorder=11)


def _leader(ax, cx, cy, ax_t, ay_t, text, fontsize=9):
    ax.annotate(
        text,
        xy=(cx, cy),
        xytext=(ax_t, ay_t),
        fontsize=fontsize,
        ha='center', va='center',
        color='black',
        fontweight='normal',
        zorder=12,
        arrowprops=dict(
            arrowstyle='-',
            color='#888',
            lw=0.7,
            shrinkA=0,
            shrinkB=3,
        ),
    )


def figure_districts(df, ghana_gdf, neighbors, lakes_c, regions_gdf):
    # One point per community (baseline wave only to avoid duplicate GPS)
    sites = (
        df[df['wave'] == 0]
        .dropna(subset=['gps_longitude', 'gps_latitude', 'district'])
        .groupby('comm', as_index=False)
        .agg(lon=('gps_longitude', 'first'),
             lat=('gps_latitude', 'first'),
             district=('district', 'first'))
    )

    districts = sorted(sites['district'].unique())
    cmap    = cm.get_cmap('tab20', len(districts))
    d_color = {d: cmap(i) for i, d in enumerate(districts)}

    fig, ax = plt.subplots(figsize=(6, 7), dpi=150)
    fig.patch.set_facecolor('white')
    _draw_base(ax, ghana_gdf, neighbors, lakes_c, regions_gdf)
    ax.axis('off')
    ax.figure.set_facecolor('white')
    _thin_border(ax, ghana_gdf, regions_gdf)

    for d, grp in sites.groupby('district'):
        color = d_color[d]
        cx = grp['lon'].mean()
        cy = grp['lat'].mean()
        ax.scatter(grp['lon'], grp['lat'],
                   color=color, s=20, zorder=6,
                   edgecolors='white', linewidths=0.3, alpha=0.92)
        ax_t, ay_t = DIST_ANCHOR.get(d, (cx + 0.4, cy + 0.4))
        _leader(ax, cx, cy, ax_t, ay_t, d, fontsize=9)

    fig.tight_layout(pad=0.1)
    return fig


def main():
    df = load_data(DATA_DIR)
    ghana_gdf, neighbors, lakes_c, regions_gdf = _load_basemap()

    fig_d = figure_districts(df, ghana_gdf, neighbors, lakes_c, regions_gdf)
    fig_d.savefig(OUT_DIR / "figure_districts.pdf", bbox_inches='tight', facecolor='white')
    fig_d.savefig(OUT_DIR / "figure_districts.png", bbox_inches='tight', facecolor='white')
    print(f"Saved → {OUT_DIR / 'figure_districts.pdf'}")
    plt.close(fig_d)


if __name__ == "__main__":
    main()
