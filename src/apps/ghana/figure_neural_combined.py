"""
Combined figure: neural discovery (left) + neuron 3821 temporal grid (right).

Layout
------
Row 0 ── Neuron 3821: [map | act×2 | div | inact×2] ─┐
                                                        V── [2015 grid × 4 communities]
Row 1 ── Neuron 2095: [map | act×2 | div | inact×2] ─┘    [2017 grid × 4 communities]

A V-shaped dashed connector runs from the last inactive tile of the 3821 row
to the left edge of both temporal rows.

Usage:
    python src/apps/ghana/figure_neural_combined.py
"""

from pathlib import Path

import json

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import FancyArrowPatch
from PIL import Image as PILImage

plt.rcParams.update({
    "text.usetex":      False,
    "font.family":      "serif",
    "font.serif":       ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
})

ROOT     = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"    / "ghana"
SAT_DIR  = DATA_DIR / "satellite"
TIF_NAT  = SAT_DIR  / "tif_national"
TIF_2015 = SAT_DIR  / "tif"
TIF_2017 = SAT_DIR  / "tif_2017"
RES_DIR  = ROOT / "results" / "ghana"
OUT_DIR  = RES_DIR  / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_MAP_BG = "#D5D8DC"
C_LBL    = "#222222"
C_BLUE   = "#2E86C1"
C_GREEN  = "#2E8B57"

TEMPORAL_COMMUNITIES = [
    {"comm_id": 951,  "activation": 2.8516},
    {"comm_id": 675,  "activation": 2.3856},
    {"comm_id": 1265, "activation": 1.9522},
    {"comm_id": 624,  "activation": 0.6847},
]

CHANGE_COLORS = {
    "cropland":   "#c0392b",
    "vegetation": "#27ae60",
    "bare soil":  "#e67e22",
    "settlement": "#8e44ad",
    "water":      "#2980b9",
    "burn scar":  "#7f8c8d",
}


def _load_temporal_changes() -> dict:
    """Parse VLM temporal descriptions from neuron_3821_temporal.json.

    Returns {comm_id: [(symbol, label, color), ...]} only for communities
    where the VLM detected meaningful intensification.
    """
    path = RES_DIR / "temporal" / "neuron_3821_temporal.json"
    if not path.exists():
        return {}
    with open(path) as f:
        entries = json.load(f)

    changes = {}
    for e in entries:
        cid     = int(e["comm_id"])
        overall = e.get("overall", "").lower()
        if "does not" in overall or "no sign" in overall or "stable" in overall:
            continue
        if "intensif" not in overall and "expand" not in overall:
            continue
        items = []
        ag  = e.get("agricultural_change", "").lower()
        veg = e.get("vegetation_change",   "").lower()
        if "expansion" in ag or "intensif" in ag:
            items.append(("+", "cropland",   CHANGE_COLORS["cropland"]))
        if "denser" in veg or "increase" in veg:
            items.append(("+", "vegetation", CHANGE_COLORS["vegetation"]))
        if items:
            changes[cid] = items
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

def main():
    temporal_changes = _load_temporal_changes()
    if temporal_changes:
        print(f"Loaded temporal changes for {len(temporal_changes)} communities from JSON.")
    else:
        print("WARNING: temporal_changes.json not found — arrows will be omitted.")

    df = pd.read_stata(DATA_DIR / "LEAP1000 2015-2017 household data++.dta")
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
        zero_idx    = sorted_idx[all_acts[sorted_idx] == 0]

        neural_rows.append(dict(
            title     = f"Neuron {neuron}: {entry['label']} ({sign}impact)",
            top_keys  = comm_ids[nonzero_idx[:2]].tolist(),
            top_acts  = all_acts[nonzero_idx[:2]].tolist(),
            bot_keys  = comm_ids[zero_idx[:2]].tolist(),
            bot_acts  = all_acts[zero_idx[:2]].tolist(),
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
        for k, (sym, label, _) in enumerate(temporal_changes[cid]):
            offset_y = 0.034 * (k - (len(temporal_changes[cid]) - 1) / 2)
            fig.text(label_x, mid_y + offset_y, f"{sym} {label}",
                     ha="left", va="center", fontsize=7.5,
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
    out_pdf = OUT_DIR / "figure_neural_ghana_combined.pdf"
    out_png = OUT_DIR / "figure_neural_ghana_combined.png"
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
