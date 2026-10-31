#!/usr/bin/env python3
"""Ghana LEAP 1000 figures of the paper, one subcommand each.

  maps         figure_districts.pdf (Appendix E): the 162 LEAP communities coloured by
               district on the full national map with neighbours; labels next to the
               clusters with thin leader lines, no legend
  neural       figure_neural_ghana_combined.pdf (Figure 5): neural discoveries (left,
               one row per certified neuron: [map | 2 active | 2 inactive tiles]) and the
               2015 / 2017 temporal grid of neuron 3821 (right)
  neural-1777  figure_neural_1777.pdf (Appendix E, exploratory analysis): neuron 1777
  tile         waterways.pdf, the Ghana panel of Figure 1: community 1265 (Bongo
               district), one of the 6 LEAP communities where SAE neuron 3821
               ("ephemeral waterways") is active (4th highest), Landsat 8 2015 median
               composite (the Uganda panel is `src/apps/uganda/figures.py tile`)

`neural` and `neural-1777` read the sign of each neuron's GATE contrast (the "(+impact)"
in the titles) from results/ghana/gate/gate_Z.csv: run `tables.py gate-neurons` first.
`neural` also reads the neutral-prompt VLM run
results/ghana/temporal/neuron_3821_temporal_neutral.json (pipeline/interpret_temporal_changes.py)
and boxes the local NDVI increase (ndvi_change.py).

Outputs go to results/ghana/figures/ (maps, neural, neural-1777) and
results/figures/figure1/ (tile), or to --out-dir.

    python src/apps/ghana/figures.py maps
    python src/apps/ghana/figures.py neural
    python src/apps/ghana/figures.py neural-1777
    python src/apps/ghana/figures.py tile
"""

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import tifffile
from matplotlib.patches import FancyArrowPatch
from PIL import Image
from PIL import Image as PILImage
from shapely.geometry import box

ROOT     = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"    / "ghana"
SAT_DIR  = DATA_DIR / "satellite"
TIF_NAT  = SAT_DIR  / "tif_national"
TIF_2015 = SAT_DIR  / "tif"
TIF_2017 = SAT_DIR  / "tif_2017"
TIF_DIR  = TIF_2015
RES_DIR  = ROOT / "results" / "ghana"
OUT_DIR  = RES_DIR  / "figures"

sys.path.insert(0, str(ROOT / "src"))
from apps.ghana.data import load_data  # noqa: E402
from apps.ghana.ndvi_change import largest_patches  # noqa: E402


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


def cmd_maps(argv: list[str]) -> None:
    """figure_districts.pdf (Appendix E): LEAP communities coloured by district."""
    _paper_style()
    out_dir = _out_dir(argv, "maps", cmd_maps.__doc__)
    df = load_data(DATA_DIR)
    ghana_gdf, neighbors, lakes_c, regions_gdf = _load_basemap()

    fig_d = figure_districts(df, ghana_gdf, neighbors, lakes_c, regions_gdf)
    fig_d.savefig(out_dir / "figure_districts.pdf", bbox_inches='tight', facecolor='white')
    fig_d.savefig(out_dir / "figure_districts.png", bbox_inches='tight', facecolor='white')
    print(f"Saved → {out_dir / 'figure_districts.pdf'}")
    plt.close(fig_d)


# ===========================================================================
# neural (Figure 5)
# ===========================================================================

C_MAP_BG = "#D5D8DC"
C_LBL    = "#222222"
C_BLUE   = "#2E86C1"
C_GREEN  = "#2E8B57"

TEMPORAL_COMMUNITIES = [
    {"comm_id": 951,  "activation": 2.8516},
    {"comm_id": 675,  "activation": 2.3856},
    {"comm_id": 395,  "activation": 2.0504},
    {"comm_id": 624,  "activation": 0.6847},
]

# Inactive example tiles, pinned to the published figure. All inactive
# communities tie at z = 0, so np.argsort would pick an arbitrary pair on a rerun.
INACTIVE_COMMUNITIES = {
    3821: [1613, 1604],
    2095: [1624, 1583],
}


def _load_temporal_changes() -> dict:
    """Read the neutral-prompt VLM run, neuron_3821_temporal_neutral.json.

    Returns {comm_id: ["+ cropland"]}, only for communities where the VLM lists
    a cropland increase. Other changes (the vegetation calls) are not shown.
    """
    path = RES_DIR / "temporal" / "neuron_3821_temporal_neutral.json"
    if not path.exists():
        return {}
    with open(path) as f:
        entries = json.load(f)

    changes = {}
    for cid, e in entries.items():
        if not any(c["symbol"] == "+" and c["label"] == "cropland" for c in e["changes"]):
            continue
        changes[int(cid)] = ["+ cropland"]
    return changes


# ── image loading ──────────────────────────────────────────────────────────────

def _norm(arr: np.ndarray) -> np.ndarray:
    valid = arr[arr > 0]
    if valid.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(valid, [2, 98])
    out = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    out[arr <= 0] = 0
    return out


def load_tile_national(grid_id: int, size: int = 112) -> np.ndarray | None:
    path = TIF_NAT / f"ghana_grid{int(grid_id):06d}.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        green = src.read(2).astype(np.float32)
        nir   = src.read(4).astype(np.float32)
        swir2 = src.read(6).astype(np.float32)
    arr = (np.stack([_norm(nir), _norm(green), _norm(swir2)], axis=-1) * 255).astype(np.uint8)
    return np.asarray(PILImage.fromarray(arr).resize((size, size), PILImage.BICUBIC))


def load_tile_community(tif_dir: Path, comm_id: int, size: int = 224) -> np.ndarray | None:
    path = tif_dir / f"ghana_comm{comm_id:04d}.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        green = src.read(2).astype(np.float32)
        nir   = src.read(4).astype(np.float32)
        swir2 = src.read(6).astype(np.float32)
    arr = (np.stack([_norm(nir), _norm(green), _norm(swir2)], axis=-1) * 255).astype(np.uint8)
    return np.asarray(PILImage.fromarray(arr).resize((size, size), PILImage.BICUBIC))


# ── NDVI-change boxes on the temporal tiles ─────────────────────────────────────

# White line with a thin, semi-transparent black halo; margin in 30 m pixels.
BOX_COLOUR, BOX_LW, BOX_HALO_LW, BOX_HALO_ALPHA = "white", 1.0, 1.6, 0.6
BOX_MARGIN_PX = 2


def draw_patch_box(ax, mask: np.ndarray, size: int) -> None:
    """Draw the patch's axis-aligned bounding box (+ margin), clipped to the tile."""
    ys, xs = np.nonzero(mask)
    sy, sx = size / mask.shape[0], size / mask.shape[1]
    y0 = max(ys.min() - BOX_MARGIN_PX, 0) * sy
    y1 = min(ys.max() + 1 + BOX_MARGIN_PX, mask.shape[0]) * sy
    x0 = max(xs.min() - BOX_MARGIN_PX, 0) * sx
    x1 = min(xs.max() + 1 + BOX_MARGIN_PX, mask.shape[1]) * sx
    # display pixel edges run from -0.5 to size - 0.5; inset by 1 px so a box on
    # the tile edge stays fully visible
    x0, y0 = max(x0 - 0.5, 0.5), max(y0 - 0.5, 0.5)
    x1, y1 = min(x1 - 0.5, size - 1.5), min(y1 - 0.5, size - 1.5)
    halo = [pe.Stroke(linewidth=BOX_HALO_LW, foreground="black", alpha=BOX_HALO_ALPHA),
            pe.Normal()]
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor=BOX_COLOUR, linewidth=BOX_LW, joinstyle="miter",
                               path_effects=halo, zorder=5))


# ── basemap ────────────────────────────────────────────────────────────────────

def load_basemap():
    ghana = gpd.read_file(DATA_DIR / "geo" / "gadm41_GHA_1.json").to_crs("EPSG:4326")
    lakes = gpd.read_file(DATA_DIR / "geo" / "ne_10m_lakes.shp").to_crs("EPSG:4326")
    lakes = lakes.clip(ghana.total_bounds)
    return ghana, lakes


def draw_map(ax, comm_df, acts, ghana=None, lakes=None, color=C_BLUE):
    if ghana is not None:
        ghana.plot(ax=ax, color="#F0F0F0", edgecolor="#AAAAAA", linewidth=0.4)
    if lakes is not None:
        lakes.plot(ax=ax, color="#AED6F1", edgecolor="none")
    ax.scatter(comm_df["gps_longitude"], comm_df["gps_latitude"],
               c=C_MAP_BG, s=5, linewidths=0, zorder=2)
    mask = acts > 0
    if mask.any():
        ax.scatter(comm_df.loc[mask, "gps_longitude"],
                   comm_df.loc[mask, "gps_latitude"],
                   c=color, s=8, linewidths=0, zorder=3, alpha=0.9)
    ax.set_axis_off()
    ax.set_aspect("equal")


# ── layout constants (inches) ──────────────────────────────────────────────────

# Neural section
map_w   = 1.15
img_w   = 1.1
div_w   = 0.05
pad_w   = 0.08
row_h   = 1.28
title_h = 0.22
row_gap = 0.10
top_pad = 0.03
bot_pad = 0.05

# Gap between neural and temporal sections
section_gap = 0.55

# Temporal section
t_year_w = 0.28
t_img_w  = 1.1
t_gap_w  = 0.0
t_n_cols = len(TEMPORAL_COMMUNITIES)

n_rows = 2  # neuron 3821 (row 0) + neuron 2095 (row 1)

# Total figure dimensions (inches)
neural_w   = pad_w + map_w + pad_w + 2*img_w + div_w + 2*img_w + pad_w
temporal_w = t_year_w + t_n_cols * t_img_w + (t_n_cols - 1) * t_gap_w
total_w    = neural_w + section_gap + temporal_w
total_h    = n_rows*(row_h + title_h) + (n_rows - 1)*row_gap + top_pad + bot_pad


def xf(x_in: float) -> float:
    """Inches from left → figure x-fraction."""
    return x_in / total_w


def yf(y_from_top_in: float) -> float:
    """Inches from top → figure y-fraction (matplotlib: 0=bottom)."""
    return 1.0 - y_from_top_in / total_h


# ── helpers ────────────────────────────────────────────────────────────────────

def _fmt_z(act: float) -> str:
    return "0" if act == 0 else f"{act:.2f}".rstrip("0").rstrip(".")


def _row_y(row_i: int):
    """Return (y_title_top, y_img_top, y_img_bot, y_img_mid) in figure fractions."""
    y_title_top_in = top_pad + row_i * (row_h + title_h + row_gap)
    y_img_top_in   = y_title_top_in + title_h
    y_img_bot_in   = y_img_top_in + row_h
    y_img_mid_in   = y_img_top_in + row_h / 2
    return (yf(y_title_top_in), yf(y_img_top_in),
            yf(y_img_bot_in),   yf(y_img_mid_in))


# ── main ───────────────────────────────────────────────────────────────────────

def cmd_neural(argv: list[str]) -> None:
    """figure_neural_ghana_combined.pdf (Figure 5)."""
    _paper_style()
    out_dir = _out_dir(argv, "neural", cmd_neural.__doc__)
    ndvi_patches = {c["comm_id"]: largest_patches(c["comm_id"]) for c in TEMPORAL_COMMUNITIES}
    for cid, patches in ndvi_patches.items():
        for p in patches:
            print(f"NDVI-increase box, community {cid}: {p['area_ha']:.1f} ha, "
                  f"local dNDVI {p['local_dndvi']:+.3f}")

    temporal_changes = _load_temporal_changes()
    if temporal_changes:
        print(f"Loaded temporal changes for {len(temporal_changes)} communities from JSON.")
    else:
        print("WARNING: neuron_3821_temporal_neutral.json not found — arrows will be omitted.")

    df = pd.read_stata(DATA_DIR / "survey" / "LEAP1000 2015-2017 household data++.dta")
    comm_df = (df.groupby("comm")[["gps_latitude", "gps_longitude"]]
                 .mean().reset_index().set_index("comm"))

    sae_acts = np.load(SAT_DIR / "sae_activations.npy")
    comm_ids = np.load(SAT_DIR / "prithvi_comm_ids.npy")
    comm_df_aligned = comm_df.reindex(comm_ids).reset_index()

    interp_path = RES_DIR / "codes" / "nexis_fwer_crve" / "qwen25_72b" / "interpretations.json"
    with open(interp_path) as f:
        interps = json.load(f)

    ghana, lakes = load_basemap()

    # Load effect direction from gate results
    gate_df = pd.read_csv(RES_DIR / "gate" / "gate_Z.csv")
    gate_df["neuron_idx"] = gate_df["feature"].str.extract(r"neuron (\d+)").astype(float)
    direction_map = dict(zip(
        gate_df["neuron_idx"].dropna().astype(int),
        gate_df.loc[gate_df["neuron_idx"].notna(), "diff"],
    ))

    neural_rows = []
    colors = {3821: C_BLUE, 2095: C_GREEN}
    for entry in interps:
        neuron = entry["neuron_idx"]
        diff   = direction_map.get(neuron, 0)
        sign   = "+" if diff >= 0 else "-"

        all_acts    = sae_acts[:, neuron]
        sorted_idx  = np.argsort(all_acts)[::-1]
        nonzero_idx = sorted_idx[all_acts[sorted_idx] > 0]
        bot_keys    = INACTIVE_COMMUNITIES[neuron]
        bot_idx     = [int(np.flatnonzero(comm_ids == k)[0]) for k in bot_keys]

        neural_rows.append(dict(
            title     = f"Neuron {neuron}: {entry['label']} ({sign}impact)",
            top_keys  = comm_ids[nonzero_idx[:2]].tolist(),
            top_acts  = all_acts[nonzero_idx[:2]].tolist(),
            bot_keys  = bot_keys,
            bot_acts  = all_acts[bot_idx].tolist(),
            all_acts  = all_acts,
            comm_df   = comm_df_aligned,
            map_color = colors.get(neuron, C_BLUE),
        ))

    fig = plt.figure(figsize=(total_w, total_h), dpi=150)
    fig.patch.set_facecolor("white")

    # ── neural rows ───────────────────────────────────────────────────────────
    for row_i, row in enumerate(neural_rows):
        _, y_img_top, _, _ = _row_y(row_i)
        h_title = title_h / total_h
        h_img   = row_h   / total_h
        w_img   = img_w   / total_w
        w_map   = map_w   / total_w
        w_div   = div_w   / total_w

        # Title (spans neural section width only)
        ax_t = fig.add_axes([0, y_img_top, neural_w / total_w, h_title])
        ax_t.set_axis_off()
        t = row["title"]
        if ": " in t:
            pre, desc = t.split(": ", 1)
            t = pre + ": " + desc[0].upper() + desc[1:]
        ax_t.text(0.01, 0.72 if row_i == 0 else 0.58, t,
                  ha="left", va="center", fontsize=9,
                  fontweight="normal", transform=ax_t.transAxes)

        # Map
        fig.add_axes([xf(pad_w), y_img_top - h_img, w_map, h_img])
        draw_map(plt.gca(), row["comm_df"], row["all_acts"],
                 ghana=ghana, lakes=lakes, color=row["map_color"])

        # Active tiles
        x_act0_in = pad_w + map_w + pad_w
        for j, (key, act) in enumerate(zip(row["top_keys"], row["top_acts"])):
            ax = fig.add_axes([xf(x_act0_in + j*img_w), y_img_top - h_img, w_img, h_img])
            arr = load_tile_community(TIF_2015, int(key))
            if arr is not None:
                ax.imshow(arr)
            ax.set_axis_off()
            ax.set_title(f"Active ($z={_fmt_z(act)}$)", fontsize=8, color=C_LBL,
                         pad=2, fontweight="normal")
            ax.text(0.5, -0.03, f"Community {int(key)}", fontsize=6.5, color="#666666",
                    ha="center", va="top", transform=ax.transAxes, clip_on=False)

        # Divider
        x_div_in = pad_w + map_w + pad_w + 2*img_w
        ax_div = fig.add_axes([xf(x_div_in), y_img_top - h_img, w_div, h_img])
        ax_div.set_axis_off()
        ax_div.axvline(0.5, color="#BBBBBB", linewidth=0.6, ymin=0.04, ymax=0.96)

        # Inactive tiles
        x_inact0_in = x_div_in + div_w
        for j, (key, act) in enumerate(zip(row["bot_keys"], row["bot_acts"])):
            ax = fig.add_axes([xf(x_inact0_in + j*img_w), y_img_top - h_img, w_img, h_img])
            arr = load_tile_community(TIF_2015, int(key))
            if arr is not None:
                ax.imshow(arr)
            ax.set_axis_off()
            ax.set_title(f"Inactive ($z={_fmt_z(act)}$)", fontsize=8, color=C_LBL,
                         pad=2, fontweight="normal")
            ax.text(0.5, -0.03, f"Community {int(key)}", fontsize=6.5, color="#666666",
                    ha="center", va="top", transform=ax.transAxes, clip_on=False)

    # ── temporal grid ─────────────────────────────────────────────────────────
    t_x0_in = neural_w + section_gap

    for row_i, (year, tif_dir) in enumerate([(2015, TIF_2015), (2017, TIF_2017)]):
        _, y_img_top, _, _ = _row_y(row_i)
        h_img   = row_h / total_h
        h_title = title_h / total_h
        w_tile  = t_img_w / total_w

        # Community tiles
        for col, comm in enumerate(TEMPORAL_COMMUNITIES):
            cid = comm["comm_id"]
            x_tile_in = t_x0_in + t_year_w + col * (t_img_w + t_gap_w)
            ax = fig.add_axes([xf(x_tile_in), y_img_top - h_img, w_tile, h_img])
            img = load_tile_community(tif_dir, cid)
            if img is not None:
                ax.imshow(img)
                # Same boxes on both years: they mark where the change happened.
                for patch in ndvi_patches[cid]:
                    draw_patch_box(ax, patch["mask"], img.shape[0])
            else:
                ax.set_facecolor("#222")
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        color="white", fontsize=8, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_i == 1:
                ax.text(0.5, -0.03, f"Community {cid}", fontsize=6.5, color="#666666",
                        ha="center", va="top", transform=ax.transAxes, clip_on=False)

    # ── year labels at the gap line ───────────────────────────────────────────
    _, y_img_top_0, y_img_bot_0, _ = _row_y(0)
    _, y_img_top_1, _,            _ = _row_y(1)
    # 2015: centered over the V upper arm (from last inactive tile to first temporal tile)
    x_v_origin = xf(pad_w + map_w + pad_w + 2*img_w + div_w + 2*img_w)
    x_v_target = xf(t_x0_in + t_year_w)
    _, _, _, y_mid_0 = _row_y(0)
    fig.text(0.5 * (x_v_origin + x_v_target), y_mid_0 + 0.1/2.54/total_h, "2015", fontsize=10,
             fontweight="normal", ha="center", va="bottom", transform=fig.transFigure)
    # 2017: left of temporal section, shifted 1 cm further left then 0.2 cm right
    x_2017 = xf(t_x0_in) - (1.0 - 0.2) / 2.54 / total_w
    fig.text(x_2017, y_img_top_1 + 0.02, "2017", fontsize=10, fontweight="normal",
             ha="left", va="top", transform=fig.transFigure)

    # ── VLM change arrows (between 2015 and 2017 rows) ────────────────────────
    fig.canvas.draw()

    for col, comm in enumerate(TEMPORAL_COMMUNITIES):
        cid = comm["comm_id"]
        if cid not in temporal_changes:
            continue

        x_tile_in = t_x0_in + t_year_w + col * (t_img_w + t_gap_w)
        arrow_x   = xf(x_tile_in) + 0.12 * (t_img_w / total_w)
        p_top = np.array([arrow_x, y_img_bot_0])
        p_bot = np.array([arrow_x, y_img_top_1])

        fig.add_artist(FancyArrowPatch(
            posA=p_top, posB=p_bot,
            arrowstyle="-|>", mutation_scale=5,
            linewidth=0.8, color="#444",
            transform=fig.transFigure, zorder=10,
        ))

        mid_y   = 0.5 * (p_top[1] + p_bot[1])
        label_x = arrow_x + 0.003
        lines = [(t, 7.5) for t in temporal_changes[cid]]
        step  = 0.1 / total_h                      # 0.1 in between baselines
        for k, (line, size) in enumerate(lines):
            offset_y = -step * (k - (len(lines) - 1) / 2)
            fig.text(label_x, mid_y + offset_y, line,
                     ha="left", va="center", fontsize=size,
                     color="black", fontweight="normal",
                     transform=fig.transFigure)

    # ── V-shape connector ─────────────────────────────────────────────────────
    # Origin: right edge of last inactive tile in row 0, vertical centre
    x_origin_in = pad_w + map_w + pad_w + 2*img_w + div_w + 2*img_w
    _, _, _, y_mid_0 = _row_y(0)
    _, _, _, y_mid_1 = _row_y(1)

    # Target: left edge of first temporal tile
    x_target_in = t_x0_in + t_year_w

    x_o    = xf(x_origin_in)
    y_o    = y_mid_0
    x_t    = xf(x_target_in)
    for (xa, ya), (xb, yb) in [
        ((x_o, y_o), (x_t, y_mid_0)),   # upper arm → 2015
        ((x_o, y_o), (x_t, y_mid_1)),   # lower arm → 2017
    ]:
        fig.add_artist(plt.Line2D(
            [xa, xb], [ya, yb],
            transform=fig.transFigure,
            color="#888888", linewidth=1.2,
            linestyle="--", zorder=5,
        ))

    # ── save ──────────────────────────────────────────────────────────────────
    out_pdf = out_dir / "figure_neural_ghana_combined.pdf"
    out_png = out_dir / "figure_neural_ghana_combined.png"
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")


# ===========================================================================
# neural-1777
# ===========================================================================

NEURON_1777    = 1777
N_TILES_1777   = 3
C_RED_1777     = "#C0392B"   # burn scar




def load_leap_tile(comm_id: int, size: int = 112) -> np.ndarray | None:
    path = TIF_DIR / f"ghana_comm{int(comm_id):04d}.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        green = src.read(2).astype(np.float32)
        nir   = src.read(4).astype(np.float32)
        swir2 = src.read(6).astype(np.float32)
    arr = (np.stack([_norm(nir), _norm(green), _norm(swir2)], axis=-1) * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(arr).resize((size, size), Image.BICUBIC))


def _basemap_1777():
    ghana = gpd.read_file(DATA_DIR / "gadm41_GHA_1.json").to_crs("EPSG:4326")
    lakes = gpd.read_file(DATA_DIR / "ne_10m_lakes.shp").to_crs("EPSG:4326")
    lakes = lakes.clip(ghana.total_bounds)
    return ghana, lakes


def _draw_map_1777(ax, comm_df, acts, ghana, lakes):
    ghana.plot(ax=ax, color="#F0F0F0", edgecolor="#AAAAAA", linewidth=0.4)
    lakes.plot(ax=ax, color="#AED6F1", edgecolor="none")
    ax.scatter(comm_df["gps_longitude"], comm_df["gps_latitude"],
               c=C_MAP_BG, s=5, linewidths=0, zorder=2)
    mask = acts > 0
    if mask.any():
        ax.scatter(comm_df.loc[mask, "gps_longitude"],
                   comm_df.loc[mask, "gps_latitude"],
                   c=C_RED_1777, s=8, linewidths=0, zorder=3, alpha=0.9)
    ax.set_axis_off()
    ax.set_aspect("equal")




def cmd_neural_1777(argv: list[str]) -> None:
    """figure_neural_1777.pdf (Appendix E, exploratory analysis): neuron 1777, sparse burn
    scar presence: map | 3 active LEAP tiles | 3 inactive LEAP tiles."""
    _paper_style()
    out_dir = _out_dir(argv, "neural-1777", cmd_neural_1777.__doc__)
    # Load interpretation label
    interp_path = RES_DIR / "codes" / "nexis_no_adj" / "qwen25_72b" / "interpretations.json"
    with open(interp_path) as f:
        interps = json.load(f)
    entry = next(e for e in interps if e.get("neuron_idx") == NEURON_1777)
    label = entry["label"]
    pval  = entry["pvalue"]

    # Load effect direction
    gate_df = pd.read_csv(RES_DIR / "gate" / "gate_Z.csv")
    gate_df["neuron_idx"] = gate_df["feature"].str.extract(r"neuron (\d+)").astype(float)
    direction_map = dict(zip(
        gate_df["neuron_idx"].dropna().astype(int),
        gate_df.loc[gate_df["neuron_idx"].notna(), "diff"],
    ))
    sign = "+" if direction_map.get(NEURON_1777, 0) >= 0 else "-"

    # Community GPS centroids
    df = pd.read_stata(DATA_DIR / "LEAP1000 2015-2017 household data++.dta")
    comm_df = (df.groupby("comm")[["gps_latitude", "gps_longitude"]]
                 .mean().reset_index().set_index("comm"))

    # SAE activations
    sae_acts = np.load(SAT_DIR / "sae_activations.npy")
    comm_ids = np.load(SAT_DIR / "prithvi_comm_ids.npy")
    comm_df_aligned = comm_df.reindex(comm_ids).reset_index()

    all_acts    = sae_acts[:, NEURON_1777]
    sorted_idx  = np.argsort(all_acts)[::-1]
    nonzero_idx = sorted_idx[all_acts[sorted_idx] > 0]
    zero_idx    = sorted_idx[all_acts[sorted_idx] == 0]

    top_keys = comm_ids[nonzero_idx[:N_TILES_1777]].tolist()
    top_acts = all_acts[nonzero_idx[:N_TILES_1777]].tolist()
    bot_keys = comm_ids[zero_idx[:N_TILES_1777]].tolist()
    bot_acts = all_acts[zero_idx[:N_TILES_1777]].tolist()

    ghana, lakes = _basemap_1777()

    # ── layout (inches) ───────────────────────────────────────────────────────
    map_w   = 1.15
    img_w   = 1.1
    div_w   = 0.05
    pad_w   = 0.08
    row_h   = 1.28
    title_h = 0.22
    top_pad = 0.03
    bot_pad = 0.05

    total_w = pad_w + map_w + pad_w + N_TILES_1777*img_w + div_w + N_TILES_1777*img_w + pad_w
    total_h = row_h + title_h + top_pad + bot_pad

    fig = plt.figure(figsize=(total_w, total_h), dpi=150)
    fig.patch.set_facecolor("white")

    def xf(x): return x / total_w
    def yf(y): return 1.0 - y / total_h

    # Title
    y_title_top = top_pad
    ax_t = fig.add_axes([0, yf(y_title_top + title_h), 1, title_h / total_h])
    ax_t.set_axis_off()
    title_str = f"Neuron {NEURON_1777}: {label[0].upper() + label[1:]} ({sign}impact)"
    ax_t.text(0.01, 0.72, title_str, ha="left", va="center",
              fontsize=9, fontweight="normal", transform=ax_t.transAxes)

    y_img_top = yf(top_pad + title_h)
    h_img     = row_h / total_h
    w_map     = map_w / total_w
    w_img     = img_w / total_w
    w_div     = div_w / total_w

    # Map
    ax_map = fig.add_axes([xf(pad_w), y_img_top - h_img, w_map, h_img])
    _draw_map_1777(ax_map, comm_df_aligned, all_acts, ghana, lakes)

    # Active tiles
    x0_act = pad_w + map_w + pad_w
    for j, (key, act) in enumerate(zip(top_keys, top_acts)):
        ax = fig.add_axes([xf(x0_act + j*img_w), y_img_top - h_img, w_img, h_img])
        arr = load_leap_tile(int(key))
        if arr is not None:
            ax.imshow(arr)
        ax.set_axis_off()
        ax.set_title(f"Active ($z={_fmt_z(act)}$)", fontsize=8, color=C_LBL,
                     pad=2, fontweight="normal")
        ax.text(0.5, -0.03, f"Community {int(key)}", fontsize=6.5, color="#666666",
                ha="center", va="top", transform=ax.transAxes, clip_on=False)

    # Divider
    x_div = x0_act + N_TILES_1777 * img_w
    ax_div = fig.add_axes([xf(x_div), y_img_top - h_img, w_div, h_img])
    ax_div.set_axis_off()
    ax_div.axvline(0.5, color="#BBBBBB", linewidth=0.6, ymin=0.04, ymax=0.96)

    # Inactive tiles
    x0_bot = x_div + div_w
    for j, (key, act) in enumerate(zip(bot_keys, bot_acts)):
        ax = fig.add_axes([xf(x0_bot + j*img_w), y_img_top - h_img, w_img, h_img])
        arr = load_leap_tile(int(key))
        if arr is not None:
            ax.imshow(arr)
        ax.set_axis_off()
        ax.set_title(f"Inactive ($z={_fmt_z(act)}$)", fontsize=8, color=C_LBL,
                     pad=2, fontweight="normal")
        ax.text(0.5, -0.03, f"Community {int(key)}", fontsize=6.5, color="#666666",
                ha="center", va="top", transform=ax.transAxes, clip_on=False)

    for ext in ("pdf", "png"):
        out = out_dir / f"figure_neural_1777.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        print(f"Saved → {out}")
    plt.close(fig)


# ===========================================================================
# tile (Figure 1, Ghana panel)
# ===========================================================================

# Bands of the Ghana Landsat 8 tiles: B4,B3,B2,B5,B6,B7 -> 1=Green 3=NIR 5=SWIR2.
TILE = dict(name="waterways",
            tif=ROOT / "data/ghana/satellite/tif/ghana_comm1265.tif",
            swir_idx=5, country="Ghana", year=2015)
TILE_OUT = ROOT / "results" / "figures" / "figure1"


def district_of(name: str) -> str:
    """Look the district label up in the survey data rather than hard-coding it."""
    df = pd.read_stata(ROOT / "data/ghana/survey/LEAP1000 2015-2017 household data++.dta",
                       columns=["comm", "district"])
    d = df.loc[df.comm == 1265, "district"].dropna().astype(str).unique()
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
    """waterways.{pdf,png}: the Ghana tile of Figure 1, drawn at native resolution in a
    3x3 inch canvas with a "<district>, Ghana (2015)" label at the top right."""
    make_tile(TILE, _out_dir(argv, "tile", cmd_tile.__doc__, TILE_OUT))


COMMANDS = {"maps": cmd_maps, "neural": cmd_neural, "neural-1777": cmd_neural_1777,
            "tile": cmd_tile}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if argv and argv[0] in ("-h", "--help") else 2)
    COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
