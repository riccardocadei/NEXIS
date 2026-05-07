"""
Neural discovery figure for Ghana LEAP 1000 (NEXIS FWER, CR1S).

One row per discovery:
  [map] | [top-2 satellite images] | [divider] | [bot-2 satellite images]

Usage:
    python src/apps/ghana/figure_neural.py
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from PIL import Image

plt.rcParams.update({
    "text.usetex":      False,
    "font.family":      "serif",
    "font.serif":       ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
})

ROOT     = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "ghana"
SAT_DIR  = DATA_DIR / "satellite"
TIF_NAT  = SAT_DIR / "tif_national"
RES_DIR  = ROOT / "results" / "ghana"
OUT_DIR  = ROOT / "results" / "ghana" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_MAP_BG = "#D5D8DC"
C_LBL    = "#222222"
C_BLUE   = "#2E86C1"   # ephemeral waterways
C_GREEN  = "#2E8B57"   # closed-canopy forest


# ── image loading ──────────────────────────────────────────────────────────────

def _norm(arr: np.ndarray) -> np.ndarray:
    valid = arr[arr > 0]
    if valid.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(valid, [2, 98])
    out = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    out[arr <= 0] = 0
    return out


def load_tile(grid_id: int, size: int = 112) -> np.ndarray | None:
    """False-colour (NIR/Green/SWIR2) tile from the national pool."""
    path = TIF_NAT / f"ghana_grid{int(grid_id):06d}.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        green = src.read(2).astype(np.float32)   # B3
        nir   = src.read(4).astype(np.float32)   # B5
        swir2 = src.read(6).astype(np.float32)   # B7
    arr = (np.stack([_norm(nir), _norm(green), _norm(swir2)], axis=-1) * 255).astype(np.uint8)
    img = Image.fromarray(arr).resize((size, size), Image.BICUBIC)
    return np.asarray(img)


def load_leap_tile(comm_id: int, size: int = 112) -> np.ndarray | None:
    """False-colour (NIR/Green/SWIR2) tile from the LEAP RCT community pool."""
    path = SAT_DIR / "tif" / f"ghana_comm{int(comm_id):04d}.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        green = src.read(2).astype(np.float32)
        nir   = src.read(4).astype(np.float32)
        swir2 = src.read(6).astype(np.float32)
    arr = (np.stack([_norm(nir), _norm(green), _norm(swir2)], axis=-1) * 255).astype(np.uint8)
    img = Image.fromarray(arr).resize((size, size), Image.BICUBIC)
    return np.asarray(img)


# ── basemap ────────────────────────────────────────────────────────────────────

def load_basemap():
    ghana = gpd.read_file(DATA_DIR / "gadm41_GHA_1.json").to_crs("EPSG:4326")
    lakes = gpd.read_file(DATA_DIR / "ne_10m_lakes.shp").to_crs("EPSG:4326")
    lakes = lakes.clip(ghana.total_bounds)
    return ghana, lakes


def draw_map(ax, comm_df, acts, ghana=None, lakes=None, color=C_BLUE):
    if ghana is not None:
        ghana.plot(ax=ax, color="#F0F0F0", edgecolor="#AAAAAA", linewidth=0.4)
    if lakes is not None:
        lakes.plot(ax=ax, color="#AED6F1", edgecolor="none")

    ax.scatter(comm_df["gps_longitude"], comm_df["gps_latitude"],
               c=C_MAP_BG, s=5, linewidths=0, zorder=2)

    active_mask = acts > 0
    if active_mask.any():
        ax.scatter(comm_df.loc[active_mask, "gps_longitude"],
                   comm_df.loc[active_mask, "gps_latitude"],
                   c=color, s=8, linewidths=0, zorder=3, alpha=0.9)

    ax.set_axis_off()
    ax.set_aspect("equal")


# ── figure builder ─────────────────────────────────────────────────────────────

def build_figure(rows: list[dict], out_path: Path, ghana=None, lakes=None,
                 tile_loader=None):
    """
    rows: list of dicts with keys:
        title      str
        top_keys   list of int  (LEAP community IDs)
        top_acts   list of float
        bot_keys   list of int  (LEAP community IDs)
        bot_acts   list of float
        all_acts   np.ndarray shape (n_comm,)   — community-level, for map
        comm_df    DataFrame with gps_latitude, gps_longitude  (index aligned to all_acts)
        map_color  str
    tile_loader: callable(comm_id) -> np.ndarray | None
    """
    if tile_loader is None:
        tile_loader = load_leap_tile
    n_rows  = len(rows)
    map_w   = 1.15
    img_w   = 1.1
    div_w   = 0.05
    pad_w   = 0.08
    row_h   = 1.28
    title_h = 0.22
    row_gap = 0.10
    top_pad = 0.03
    bot_pad = 0.05

    total_w = map_w + 2*img_w + div_w + 2*img_w + 4*pad_w
    total_h = n_rows*(row_h+title_h) + max(n_rows-1, 0)*row_gap + top_pad + bot_pad

    fig = plt.figure(figsize=(total_w, total_h), dpi=150)
    fig.patch.set_facecolor("white")

    for row_i, row in enumerate(rows):
        y_top = 1.0 - (top_pad + row_i*(row_h+title_h+row_gap)) / total_h

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

        ax_map = fig.add_axes([x_starts[0], y_img_top - img_frac, map_frac, img_frac])
        draw_map(ax_map, row["comm_df"], row["all_acts"],
                 ghana=ghana, lakes=lakes, color=row.get("map_color", C_BLUE))

        def _fmt_z(act):
            return "0" if act == 0 else f"{act:.2f}".rstrip("0").rstrip(".")

        for j, (key, act) in enumerate(zip(row["top_keys"], row["top_acts"])):
            ax = fig.add_axes([x_starts[2+j], y_img_top - img_frac, img_frac_w, img_frac])
            arr = tile_loader(int(key))
            if arr is not None:
                ax.imshow(arr)
            ax.set_axis_off()
            ax.set_title(f"Active ($z={_fmt_z(act)}$)", fontsize=8, color=C_LBL,
                         pad=2, fontweight="normal")
            ax.text(0.5, -0.03, f"Community {int(key)}", fontsize=6.5, color="#666666",
                    ha="center", va="top", transform=ax.transAxes, clip_on=False)

        ax_div = fig.add_axes([x_starts[4], y_img_top - img_frac, div_frac, img_frac])
        ax_div.set_axis_off()
        ax_div.axvline(0.5, color="#BBBBBB", linewidth=0.6, ymin=0.04, ymax=0.96)

        for j, (key, act) in enumerate(zip(row["bot_keys"], row["bot_acts"])):
            ax = fig.add_axes([x_starts[5+j], y_img_top - img_frac, img_frac_w, img_frac])
            arr = tile_loader(int(key))
            if arr is not None:
                ax.imshow(arr)
            ax.set_axis_off()
            ax.set_title(f"Inactive ($z={_fmt_z(act)}$)", fontsize=8, color=C_LBL,
                         pad=2, fontweight="normal")
            ax.text(0.5, -0.03, f"Community {int(key)}", fontsize=6.5, color="#666666",
                    ha="center", va="top", transform=ax.transAxes, clip_on=False)

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    import json

    # Community coordinates (GPS centroid per community)
    df = pd.read_stata(DATA_DIR / "LEAP1000 2015-2017 household data++.dta")
    comm_df = (df.groupby("comm")[["gps_latitude", "gps_longitude"]]
                 .mean()
                 .reset_index()
                 .set_index("comm"))

    # SAE community activations (162 communities × 4096 neurons)
    sae_acts  = np.load(SAT_DIR / "sae_activations.npy")   # (162, 4096)
    comm_ids  = np.load(SAT_DIR / "prithvi_comm_ids.npy")  # (162,)

    # Align comm_df to comm_ids order
    comm_df_aligned = comm_df.reindex(comm_ids).reset_index()

    # Load FWER interpretations
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

    rows = []
    colors = {3821: C_BLUE, 2095: C_GREEN}
    for entry in interps:
        neuron = entry["neuron_idx"]
        label  = entry["label"]
        pval   = entry["pvalue"]
        color  = colors.get(neuron, C_BLUE)

        all_acts = sae_acts[:, neuron]

        # Use LEAP RCT community tiles, sorted by activation
        sorted_idx = np.argsort(all_acts)[::-1]
        nonzero_idx = sorted_idx[all_acts[sorted_idx] > 0]
        zero_idx    = sorted_idx[all_acts[sorted_idx] == 0]

        top_keys = comm_ids[nonzero_idx[:2]].tolist()
        top_acts = all_acts[nonzero_idx[:2]].tolist()
        bot_keys = comm_ids[zero_idx[:2]].tolist()
        bot_acts = all_acts[zero_idx[:2]].tolist()

        diff = direction_map.get(neuron, 0)
        sign  = "+" if diff >= 0 else "-"
        title = f"Neuron {neuron}: {label} ({sign}impact)"

        rows.append(dict(
            title     = title,
            top_keys  = top_keys,
            top_acts  = top_acts,
            bot_keys  = bot_keys,
            bot_acts  = bot_acts,
            all_acts  = all_acts,
            comm_df   = comm_df_aligned,
            map_color = color,
        ))

    build_figure(rows, OUT_DIR / "figure_neural_ghana.pdf",  ghana=ghana, lakes=lakes)
    build_figure(rows, OUT_DIR / "figure_neural_ghana.png",  ghana=ghana, lakes=lakes)


if __name__ == "__main__":
    main()
