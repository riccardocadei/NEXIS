#!/usr/bin/env python3
"""Uganda YOP figures of the paper, one subcommand each.

  maps    figure_districts.pdf and figure_languages.pdf (Appendix D): RCT sites coloured
          by district and by language group; labels next to the clusters with thin
          leader lines, no legend, thin country border
  neural  figure_neural_skilled_employed.pdf and figure_neural_log_biz_assets.pdf
          (Figure 4): one row per modifier, [map] | [2 most active tiles] | [2 inactive
          tiles], the sign of the GATE contrast in the title
  tile    river.pdf, the Uganda panel of Figure 1: RCT site 92 (Kamdini parish, Apac
          district), the top-activating site of SAE neuron 339 ("perennial river
          presence"), Landsat 7 2005-2007 median composite (the Ghana panel is
          `src/apps/ghana/figures.py tile`)

Outputs go to results/uganda/figures/ (maps, neural) and results/figures/figure1/ (tile),
or to --out-dir.

    python src/apps/uganda/figures.py maps
    python src/apps/uganda/figures.py neural
    python src/apps/uganda/figures.py tile
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import tifffile
from PIL import Image

ROOT      = Path(__file__).resolve().parents[3]
DATA_DIR  = ROOT / "data" / "uganda"
TIF_DIR   = DATA_DIR / "satellite" / "tif_rct"
MAP_DIR   = DATA_DIR / "map"
SPEC_PATH = DATA_DIR / "satellite" / "rct" / "spectral_indices.csv"
RES_DIR   = ROOT / "results" / "uganda" / "prithvi_l5_1024"
OUT_DIR   = ROOT / "results" / "uganda" / "figures"

sys.path.insert(0, str(ROOT / "src"))
from apps.uganda.data import load_basemap, draw_base, LANG_NAMES  # noqa: E402


def _paper_style() -> None:
    """Serif fonts of the maps and the neural figures (the tile keeps the defaults)."""
    plt.rcParams.update({
        "text.usetex":      False,
        "font.family":      "serif",
        "font.serif":       ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
    })


def _out_dir(argv: list[str], cmd: str, doc: str, default: Path = OUT_DIR) -> Path:
    ap = argparse.ArgumentParser(prog=f"figures.py {cmd}", description=doc)
    ap.add_argument("--out-dir", type=Path, default=default,
                    help="where to write the figures (default: %(default)s)")
    out_dir = ap.parse_args(argv).out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ===========================================================================
# maps
# ===========================================================================

# map extents: xlim (29.1, 35.4), ylim (-1.65, 4.45)

# ── Language anchors — placed so no lines cross ───────────────────────────────
# Centroids (lon,lat): Alur(31.20,2.47) Lugbara(31.11,3.12) Madi(31.73,3.48)
#            Karamojong(34.35,2.72) Teso(33.74,1.56) Langi(32.74,2.10) Other(33.73,1.19)
LANG_ANCHOR = {
    "Alur":       (29.7, 2.3),    # left      from (31.12, 2.45)
    "Lugbara":    (29.7, 3.6),    # left-up   from (31.07, 3.17)
    "Madi":       (32.5, 4.2),    # up-right  from (31.68, 3.40)
    "Karamojong": (33.8, 4.0),    # up-left   from (34.44, 2.60)
    "Teso":       (34.8, 2.1),    # right     from (33.66, 1.58)
    "Langi":      (31.3, 1.1),    # left-down from (32.74, 2.13)
    "Pallisa":    (34.8, 0.4),    # right-down from (33.79, 1.18)
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


def cmd_maps(argv: list[str]) -> None:
    """figure_districts and figure_languages (Appendix D maps)."""
    _paper_style()
    out_dir = _out_dir(argv, "maps", cmd_maps.__doc__)
    df = pd.read_csv(DATA_DIR / "UgandaDataProcessed.csv", low_memory=False)
    uganda_gdf, neighbors, lakes_c, regions_gdf = load_basemap(MAP_DIR)

    fig_d = figure_districts(df, uganda_gdf, neighbors, lakes_c, regions_gdf)
    fig_d.savefig(out_dir / "figure_districts.pdf", bbox_inches='tight', facecolor='white')
    fig_d.savefig(out_dir / "figure_districts.png", bbox_inches='tight', facecolor='white')
    print(f"Saved → {out_dir / 'figure_districts.pdf'}")
    plt.close(fig_d)

    fig_l = figure_languages(df, uganda_gdf, neighbors, lakes_c, regions_gdf)
    fig_l.savefig(out_dir / "figure_languages.pdf", bbox_inches='tight', facecolor='white')
    fig_l.savefig(out_dir / "figure_languages.png", bbox_inches='tight', facecolor='white')
    print(f"Saved → {out_dir / 'figure_languages.pdf'}")
    plt.close(fig_l)


# ===========================================================================
# neural
# ===========================================================================

# ── colours ───────────────────────────────────────────────────────────────────
C_ACTIVE   = "#C0392B"   # terracotta red — default activated (unused in current rows)
C_MAP_BG   = "#D5D8DC"   # light grey — all sites / inactive
C_LBL      = "#222222"   # near-black — z= labels above images
C_BLUE     = "#2E86C1"   # river blue  — perennial river presence (Z_339)
C_GREEN    = "#2E8B57"   # sea green — vegetation (Z_533, Z_820, NDVI)

# ── image loading ─────────────────────────────────────────────────────────────

def _norm(arr: np.ndarray) -> np.ndarray:
    valid = arr[arr > 0]
    if valid.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(valid, [2, 98])
    out = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    out[arr <= 0] = 0
    return out


def _neural_tile(key: int, size: int = 112) -> np.ndarray | None:
    """Return false-colour (NIR/Green/SWIR1) uint8 array (size, size, 3)."""
    path = TIF_DIR / f"uganda_rct{int(key):06d}.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        green = src.read(2).astype(np.float32)
        nir   = src.read(4).astype(np.float32)
        swir1 = src.read(5).astype(np.float32)
    arr = (np.stack([_norm(nir), _norm(green), _norm(swir1)], axis=-1) * 255).astype(np.uint8)
    img = Image.fromarray(arr).resize((size, size), Image.BICUBIC)
    return np.asarray(img)


# ── basemap ───────────────────────────────────────────────────────────────────

def _neural_basemap():
    uganda = gpd.read_file(MAP_DIR / "gadm41_UGA_1.json").to_crs("EPSG:4326")
    lakes  = gpd.read_file(MAP_DIR / "ne_10m_lakes.shp").to_crs("EPSG:4326")
    lakes  = lakes.clip(uganda.total_bounds)
    return uganda, lakes


def draw_map(ax, site_df, acts, uganda=None, lakes=None, acts2=None,
             continuous=False, color=C_ACTIVE):
    """Plot Uganda outline + activated sites.

    continuous=True: grey→color gradient proportional to acts (e.g. NDVI).
    continuous=False: binary — grey background, solid `color` for acts > 0.
    """
    from matplotlib.colors import to_rgb
    if uganda is not None:
        uganda.plot(ax=ax, color="#F0F0F0", edgecolor="#AAAAAA", linewidth=0.4)
    if lakes is not None:
        lakes.plot(ax=ax, color="#AED6F1", edgecolor="none")

    if continuous:
        grey = np.array(to_rgb(C_MAP_BG))
        hi_c = np.array(to_rgb(color))
        vals = np.asarray(acts, dtype=float)
        hi   = np.nanmax(vals)
        norm    = np.clip(vals / max(hi, 1e-9), 0.0, 1.0)  # 0 for ≤0, 1 at max
        colours = grey[None, :] * (1 - norm[:, None]) + hi_c[None, :] * norm[:, None]
        ax.scatter(site_df["geo_long_center"], site_df["geo_lat_center"],
                   c=colours, s=3, linewidths=0, zorder=3, alpha=0.95)
    else:
        ax.scatter(site_df["geo_long_center"], site_df["geo_lat_center"],
                   c=C_MAP_BG, s=2, linewidths=0, zorder=2)

        if acts2 is not None:
            active_mask = (acts > 0) | (acts2 > 0)
        else:
            active_mask = acts > 0

        if active_mask.any():
            ax.scatter(site_df.loc[active_mask, "geo_long_center"],
                       site_df.loc[active_mask, "geo_lat_center"],
                       c=color, s=4, linewidths=0, zorder=3, alpha=0.85)

    ax.set_axis_off()
    ax.set_aspect("equal")


# ── figure builder ────────────────────────────────────────────────────────────

def build_figure(rows: list[dict], out_path: Path,
                 uganda=None, lakes=None):
    """
    rows: list of dicts with keys:
        title      str   "Neuron 339: perennial river presence"
        raw_cols   list  [339]  or [698, 533]  (actual SAE columns in site_feats)
        colors     list  [C_ACTIVE]  or [C_MERGED_A, C_MERGED_B]
        top_keys   list of int (3 keys)
        top_acts   list of float
        bot_keys   list of int (3 keys)
        bot_acts   list of float
        all_acts   np.ndarray shape (n_sites,)   — for map
        all_acts2  np.ndarray or None            — second neuron for merged
        site_df    DataFrame with geo_long_center, geo_lat_center
    """
    n_rows = len(rows)

    # Layout: [map | img img img | divider | img img img]
    map_w   = 1.15
    img_w   = 1.1
    div_w   = 0.05
    pad_w   = 0.08
    row_h   = 1.28
    title_h = 0.22
    row_gap = 0.10   # inter-row spacing
    top_pad = 0.03
    bot_pad = 0.05

    total_w = map_w + 2*img_w + div_w + 2*img_w + 4*pad_w
    total_h = n_rows * (row_h + title_h) + max(n_rows - 1, 0) * row_gap + top_pad + bot_pad

    fig = plt.figure(figsize=(total_w, total_h), dpi=150)
    fig.patch.set_facecolor("white")

    for row_i, row in enumerate(rows):
        y_top = 1.0 - (top_pad + row_i*(row_h+title_h+row_gap)) / total_h

        # ── title ─────────────────────────────────────────────────────────────
        title_frac = title_h / total_h
        ax_title = fig.add_axes([0, y_top - title_frac, 1, title_frac])
        ax_title.set_axis_off()
        t = row["title"]
        if ": " in t:
            pre, desc = t.split(": ", 1)
            t = pre + ": " + desc[0].upper() + desc[1:]
        title_y = 0.72 if row_i == 0 else 0.58
        ax_title.text(0.01, title_y, t,
                      ha="left", va="center", fontsize=9,
                      fontweight="normal", transform=ax_title.transAxes)

        # ── image row axes ─────────────────────────────────────────────────────
        y_img_top = y_top - title_frac
        img_frac  = row_h / total_h

        col_widths = [map_w, pad_w, img_w, img_w, div_w, img_w, img_w, pad_w]
        x_starts = []
        x = pad_w / total_w
        for w in col_widths:
            x_starts.append(x)
            x += w / total_w

        map_frac   = map_w / total_w
        img_frac_w = img_w / total_w
        div_frac   = div_w / total_w

        # Map
        ax_map = fig.add_axes([x_starts[0], y_img_top - img_frac, map_frac, img_frac])
        draw_map(ax_map, row["site_df"], row["all_acts"],
                 uganda=uganda, lakes=lakes, acts2=row.get("all_acts2"),
                 continuous=row.get("continuous", False),
                 color=row.get("map_color", C_ACTIVE))

        def _fmt_z(act):
            return "0" if act == 0 else f"{act:.2f}".rstrip("0").rstrip(".")

        # Top-2 images (high activation)
        for j, (key, act) in enumerate(zip(row["top_keys"], row["top_acts"])):
            ax = fig.add_axes([x_starts[2+j], y_img_top - img_frac, img_frac_w, img_frac])
            img_arr = _neural_tile(int(key))
            if img_arr is not None:
                ax.imshow(img_arr)
            ax.set_axis_off()
            ax.set_title(f"Active ($z={_fmt_z(act)}$)", fontsize=8, color=C_LBL,
                         pad=2, fontweight="normal")
            ax.text(0.5, -0.03, f"Community {int(key)}", fontsize=6.5, color="#666666",
                    ha="center", va="top", transform=ax.transAxes, clip_on=False)

        # Divider
        ax_div = fig.add_axes([x_starts[4], y_img_top - img_frac, div_frac, img_frac])
        ax_div.set_axis_off()
        ax_div.axvline(0.5, color="#BBBBBB", linewidth=0.6, ymin=0.04, ymax=0.96)

        # Bottom-2 images (low / inactive)
        for j, (key, act) in enumerate(zip(row["bot_keys"], row["bot_acts"])):
            ax = fig.add_axes([x_starts[5+j], y_img_top - img_frac, img_frac_w, img_frac])
            img_arr = _neural_tile(int(key))
            if img_arr is not None:
                ax.imshow(img_arr)
            ax.set_axis_off()
            ax.set_title(f"Inactive ($z={_fmt_z(act)}$)", fontsize=8, color=C_LBL,
                         pad=2, fontweight="normal")
            ax.text(0.5, -0.03, f"Community {int(key)}", fontsize=6.5, color="#666666",
                    ha="center", va="top", transform=ax.transAxes, clip_on=False)

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def cmd_neural(argv: list[str]) -> None:
    """figure_neural_{skilled_employed,log_biz_assets} (Figure 4)."""
    _paper_style()
    out_dir = _out_dir(argv, "neural", cmd_neural.__doc__)
    # Shared data
    d = np.load(RES_DIR / "site_features.npz")
    site_feats = d["site_features"]   # (331, 1024)
    site_keys  = d["site_keys"]       # (331,)

    coords = (pd.read_csv(DATA_DIR / "UgandaDataProcessed.csv", low_memory=False,
                          usecols=["geo_long_lat_key","geo_long_center","geo_lat_center"])
              .drop_duplicates("geo_long_lat_key")
              .dropna()
              .set_index("geo_long_lat_key"))

    # Align coords to site_keys order
    site_df = coords.loc[site_keys].reset_index()

    uganda, lakes = _neural_basemap()

    def get_acts(raw_col):
        return site_feats[:, raw_col]

    def top_bot(acts, k=2):
        order = np.argsort(acts)[::-1]
        top_i = order[:k]
        bot_i = np.argsort(acts)[:k]
        return (site_keys[top_i].tolist(), acts[top_i].tolist(),
                site_keys[bot_i].tolist(), acts[bot_i].tolist())

    def top_bot_ndvi(ndvi_vals, k=2):
        """Top k by highest NDVI; bottom k from lowest *positive* NDVI only,
        ordered least-to-most sparse (highest→lowest among the bottom set)."""
        top_i = np.argsort(ndvi_vals)[::-1][:k]
        pos_idx = np.where(ndvi_vals > 0)[0]
        bot_pos = pos_idx[np.argsort(ndvi_vals[pos_idx])][:k]  # lowest positive, ascending
        bot_i   = bot_pos[::-1]   # reverse → least sparse first, most sparse last
        return (site_keys[top_i].tolist(), ndvi_vals[top_i].tolist(),
                site_keys[bot_i].tolist(), ndvi_vals[bot_i].tolist())

    # ── Compute GATE directions ────────────────────────────────────────────────
    df_full = pd.read_csv(DATA_DIR / "UgandaDataProcessed.csv", low_memory=False)
    site_key_to_idx = {k: i for i, k in enumerate(site_keys)}
    df_full["feat_idx"] = df_full["geo_long_lat_key"].map(site_key_to_idx)
    df_full = df_full.dropna(subset=["feat_idx", "Wobs"]).reset_index(drop=True)
    df_full["feat_idx"] = df_full["feat_idx"].astype(int)
    T_all = df_full["Wobs"].values.astype(float)

    def gate_sign(Y_col, active_mask):
        sub = df_full.dropna(subset=[Y_col])   # keep original index for correct alignment
        Y   = sub[Y_col].values.astype(float)
        T   = sub["Wobs"].values.astype(float)
        act = active_mask[sub.index.values]
        g1  = Y[act  & (T == 1)].mean() - Y[act  & (T == 0)].mean()
        g0  = Y[~act & (T == 1)].mean() - Y[~act & (T == 0)].mean()
        return "+" if (g1 - g0) >= 0 else "-"

    spec_df   = pd.read_csv(SPEC_PATH).set_index("site_key")
    ndvi_raw  = spec_df.reindex(site_keys)["ndvi_mean"].values.astype(np.float32)
    ndvi_med  = np.nanmedian(ndvi_raw)

    # ── Figure 1: skilled_employed ─────────────────────────────────────────────
    rows_se = []

    # Z_339 — perennial river presence
    a339  = get_acts(339)
    sign  = gate_sign("skilled_dummy_e", a339[df_full["feat_idx"].values] > 0)
    tk, ta, bk, ba = top_bot(a339)
    rows_se.append(dict(
        title     = f"Neuron 339: perennial river presence ({sign}impact)",
        raw_cols  = [339], colors = [C_BLUE],
        top_keys=tk, top_acts=ta, bot_keys=bk, bot_acts=ba,
        all_acts  = a339, all_acts2=None, site_df=site_df,
        map_color = C_BLUE,
    ))

    # Z_533 — vegetation spatial heterogeneity
    a533  = get_acts(533)
    sign  = gate_sign("skilled_dummy_e", a533[df_full["feat_idx"].values] > 0)
    tk, ta, bk, ba = top_bot(a533)
    rows_se.append(dict(
        title     = f"Neuron 533: vegetation spatial heterogeneity ({sign}impact)",
        raw_cols  = [533], colors = [C_GREEN],
        top_keys=tk, top_acts=ta, bot_keys=bk, bot_acts=ba,
        all_acts  = a533, all_acts2=None, site_df=site_df,
        map_color = C_GREEN,
    ))

    build_figure(rows_se, out_dir / "figure_neural_skilled_employed.pdf",
                 uganda=uganda, lakes=lakes)
    build_figure(rows_se, out_dir / "figure_neural_skilled_employed.png",
                 uganda=uganda, lakes=lakes)

    # ── Figure 2: log_biz_assets ───────────────────────────────────────────────
    rows_ba = []

    # NDVI mean — vegetation greenness (W spectral covariate)
    # (spec_df, ndvi_raw, ndvi_med already computed above for gate_sign)
    ndvi_centered = np.where(np.isnan(ndvi_raw - ndvi_med), 0.0, ndvi_raw - ndvi_med)
    ndvi_sign = gate_sign("bizasset_val_real_ln_e",
                          ndvi_raw[df_full["feat_idx"].values] > ndvi_med)
    tk, ta, bk, ba = top_bot_ndvi(ndvi_raw)
    rows_ba.append(dict(
        title      = f"NDVI: vegetation greenness ({ndvi_sign}impact)",
        raw_cols   = [], colors = [C_GREEN],
        top_keys=tk, top_acts=ta, bot_keys=bk, bot_acts=ba,
        all_acts   = ndvi_raw, all_acts2=None, site_df=site_df,
        continuous = True, map_color = C_GREEN,
    ))

    # Z_820 — structured agricultural landscape
    a820  = get_acts(820)
    sign  = gate_sign("bizasset_val_real_ln_e", a820[df_full["feat_idx"].values] > 0)
    tk, ta, bk, ba = top_bot(a820)
    # override inactive examples to avoid 219/230 (use 226, 228 instead)
    bk = [226, 228]
    ba = [float(a820[site_keys.tolist().index(k)]) for k in bk]
    rows_ba.append(dict(
        title     = f"Neuron 820: structured agricultural landscape ({sign}impact)",
        raw_cols  = [820], colors = [C_GREEN],
        top_keys=tk, top_acts=ta, bot_keys=bk, bot_acts=ba,
        all_acts  = a820, all_acts2=None, site_df=site_df,
        map_color = C_GREEN,
    ))

    build_figure(rows_ba, out_dir / "figure_neural_log_biz_assets.pdf",
                 uganda=uganda, lakes=lakes)
    build_figure(rows_ba, out_dir / "figure_neural_log_biz_assets.png",
                 uganda=uganda, lakes=lakes)


# ===========================================================================
# tile (Figure 1, Uganda panel)
# ===========================================================================

# Bands of the Uganda Landsat 7 tiles: SR_B1,B2,B3,B4,B5,B7 -> 1=Green 3=NIR 4=SWIR1.
TILE = dict(name="river",
            tif=ROOT / "data/uganda/satellite/tif_rct/uganda_rct000092.tif",
            swir_idx=4, country="Uganda", year=2007)
TILE_OUT = ROOT / "results" / "figures" / "figure1"


def district_of(name: str) -> str:
    """Look the district label up in the survey data rather than hard-coding it."""
    df = pd.read_csv(ROOT / "data/uganda/UgandaDataProcessed.csv",
                     usecols=["geo_long_lat_key", "district"])
    d = df.loc[df.geo_long_lat_key == 92, "district"].dropna().unique()
    assert len(d) == 1, f"ambiguous district for {name}: {d}"
    return str(d[0]).title()


def composite(path: Path, swir_idx: int) -> np.ndarray:
    """NIR / Green / SWIR false-colour composite, uint8, native resolution, every band
    stretched to its 2-98th percentile."""
    data = tifffile.imread(str(path)).astype(np.float32)  # (H, W, bands)
    rgb = np.stack([_norm(data[:, :, 3]), _norm(data[:, :, 1]),
                    _norm(data[:, :, swir_idx])], axis=-1)
    return (rgb * 255).astype(np.uint8)


def make_tile(spec: dict, out_dir: Path) -> None:
    img = composite(spec["tif"], spec["swir_idx"])
    label = f"{district_of(spec['name'])}, {spec['country']} ({spec['year']})"

    fig = plt.figure(figsize=(3, 3), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    ax.text(0.97, 0.97, label, transform=ax.transAxes, ha="right", va="top",
            fontsize=14, family="DejaVu Serif", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black",
                      alpha=0.5, edgecolor="none"))

    for ext in ("pdf", "png"):
        out = out_dir / f"{spec['name']}.{ext}"
        fig.savefig(out, bbox_inches="tight", pad_inches=0, dpi=200)
        print(f"Saved -> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    plt.close(fig)


def cmd_tile(argv: list[str]) -> None:
    """river.{pdf,png}: the Uganda tile of Figure 1, drawn at native resolution in a
    3x3 inch canvas with a "<district>, Uganda (2007)" label at the top right."""
    make_tile(TILE, _out_dir(argv, "tile", cmd_tile.__doc__, TILE_OUT))


COMMANDS = {"maps": cmd_maps, "neural": cmd_neural, "tile": cmd_tile}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if argv and argv[0] in ("-h", "--help") else 2)
    COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
