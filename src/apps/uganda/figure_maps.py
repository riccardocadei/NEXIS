"""
Generate and save figure_districts and figure_languages for Uganda.
Labels sit close to clusters, connected by thin leader lines (no arrowhead).
No legend. Thin country border. No lines cross each other.

Usage:
    python src/apps/uganda/figure_maps.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "uganda"
MAP_DIR  = DATA_DIR / "map"
OUT_DIR  = ROOT / "results" / "uganda" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(ROOT / "src"))
from apps.uganda.data import load_basemap, draw_base, LANG_NAMES

plt.rcParams.update({
    "text.usetex":      False,
    "font.family":      "serif",
    "font.serif":       ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
})

# map extents: xlim (29.1, 35.4), ylim (-1.65, 4.45)

# ── Language anchors — placed so no lines cross ───────────────────────────────
# Centroids (lon,lat): Alur(31.20,2.47) Lugbara(31.11,3.12) Madi(31.73,3.48)
#            Karamojong(34.35,2.72) Teso(33.74,1.56) Langi(32.74,2.10) Other(33.73,1.19)
LANG_ANCHOR = {
    "Lango":      (29.7, 3.6),    # left-up   from (31.07, 3.20)
    "Teso":       (32.4, 4.2),    # up-right  from (31.70, 3.37)
    "Acholi":     (29.7, 2.3),    # left      from (31.15, 2.46)
    "Lugbara":    (31.3, 1.1),    # left-down from (32.74, 2.09)
    "Madi":       (34.8, 2.1),    # right-up  from (33.66, 1.57)
    "Other":      (34.8, 0.4),    # right-down from (33.75, 1.19)
    "Karamojong": (33.8, 4.0),    # up-left   from (34.39, 2.69) — stays inside
}

# ── District anchors — title-cased, short lines, no crossings ─────────────────
# Centroids (approx): YUMBE(31.22,3.46) MOYO(31.62,3.52) ADJUMANI(31.73,3.32)
# KOTIDO(34.08,3.36) ARUA(31.02,3.10) NEBBI(31.15,2.46) APAC(32.57,2.08)
# LIRA(32.95,2.11) KABERAMAIDO(33.19,1.74) SOROTI(33.58,1.65)
# MOROTO(34.49,2.46) NAKAPIRIPIRIT(34.71,1.98) KUMI(33.93,1.43) PALLISA(33.75,1.19)
DIST_ANCHOR = {
    "Yumbe":         (30.3, 4.2),   # top-left
    "Moyo":          (31.5, 4.2),   # top
    "Adjumani":      (32.5, 4.1),   # top-right
    "Kotido":        (34.7, 4.0),   # top-right
    "Arua":          (29.6, 3.1),   # left
    "Nebbi":         (29.6, 2.5),   # left-lower
    "Apac":          (31.5, 2.8),   # left-up  (diverges from Lira)
    "Lira":          (33.8, 2.8),   # right-up (diverges from Apac)
    "Kaberamaido":   (32.4, 0.8),   # down-left
    "Soroti":        (34.3, 0.9),   # down-right (opposite Kaberamaido)
    "Moroto":        (34.8, 3.1),   # right-up  — stays inside
    "Nakapiripirit": (34.1, 1.9),   # between Moroto and Kumi — avoids overlap
    "Kumi":          (35.1, 1.0),   # right-down
    "Pallisa":       (34.8, 0.1),   # right-down lower
}


def _thin_border(ax, uganda_gdf, regions_gdf):
    if regions_gdf is not None:
        regions_gdf.plot(ax=ax, color='none', edgecolor='#ccc', lw=0.3, zorder=10)
    uganda_gdf.plot(ax=ax, color='none', edgecolor='#888', lw=0.6, zorder=11)


def _leader(ax, cx, cy, ax_t, ay_t, text, fontsize=10):
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


def figure_districts(df, uganda_gdf, neighbors, lakes_c, regions_gdf):
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

    fig, ax = plt.subplots(figsize=(6, 7), dpi=150)
    fig.patch.set_facecolor('white')
    draw_base(ax, uganda_gdf, neighbors, lakes_c, cities=regions_gdf, paper=False)
    ax.axis('off')
    ax.set_facecolor('white')
    ax.figure.set_facecolor('white')
    _thin_border(ax, uganda_gdf, regions_gdf)

    for d, grp in sites.groupby('district'):
        color  = d_color[d]
        label  = d.title()
        cx = grp['geo_long_center'].mean()
        cy = grp['geo_lat_center'].mean()
        ax.scatter(grp['geo_long_center'], grp['geo_lat_center'],
                   color=color, s=20, zorder=6,
                   edgecolors='white', linewidths=0.3, alpha=0.92)
        ax_t, ay_t = DIST_ANCHOR.get(label, (cx + 0.4, cy + 0.4))
        _leader(ax, cx, cy, ax_t, ay_t, label, fontsize=9)

    fig.tight_layout(pad=0.1)
    return fig


def figure_languages(df, uganda_gdf, neighbors, lakes_c, regions_gdf):
    need_cols = ['geo_long_center', 'geo_lat_center', 'lang_group']
    sites_all = df.dropna(subset=need_cols).copy()
    sites_all['lang_group'] = sites_all['lang_group'].astype(float)

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

    fig, ax = plt.subplots(figsize=(6, 7), dpi=150)
    fig.patch.set_facecolor('white')
    draw_base(ax, uganda_gdf, neighbors, lakes_c, cities=regions_gdf, paper=False)
    ax.axis('off')
    ax.set_facecolor('white')
    ax.figure.set_facecolor('white')
    _thin_border(ax, uganda_gdf, regions_gdf)

    for lang, grp in sites.groupby('lang_name'):
        color = l_color.get(lang, 'grey')
        cx = grp['geo_long_center'].mean()
        cy = grp['geo_lat_center'].mean()
        ax.scatter(grp['geo_long_center'], grp['geo_lat_center'],
                   color=color, s=20, zorder=6,
                   edgecolors='white', linewidths=0.3, alpha=0.92)
        ax_t, ay_t = LANG_ANCHOR.get(lang, (cx + 0.5, cy + 0.5))
        _leader(ax, cx, cy, ax_t, ay_t, lang, fontsize=10)

    fig.tight_layout(pad=0.1)
    return fig


def main():
    df = pd.read_csv(DATA_DIR / "UgandaDataProcessed.csv", low_memory=False)
    uganda_gdf, neighbors, lakes_c, regions_gdf = load_basemap(MAP_DIR)

    fig_d = figure_districts(df, uganda_gdf, neighbors, lakes_c, regions_gdf)
    fig_d.savefig(OUT_DIR / "figure_districts.pdf", bbox_inches='tight', facecolor='white')
    fig_d.savefig(OUT_DIR / "figure_districts.png", bbox_inches='tight', facecolor='white')
    print(f"Saved → {OUT_DIR / 'figure_districts.pdf'}")
    plt.close(fig_d)

    fig_l = figure_languages(df, uganda_gdf, neighbors, lakes_c, regions_gdf)
    fig_l.savefig(OUT_DIR / "figure_languages.pdf", bbox_inches='tight', facecolor='white')
    fig_l.savefig(OUT_DIR / "figure_languages.png", bbox_inches='tight', facecolor='white')
    print(f"Saved → {OUT_DIR / 'figure_languages.pdf'}")
    plt.close(fig_l)


if __name__ == "__main__":
    main()
