"""
Single-neuron figure for Neuron 1777 (Sparse burn scar presence).

Layout: [map | 3 active LEAP tiles | divider | 3 inactive LEAP tiles]

Usage:
    python src/apps/ghana/figure_neural_1777.py
"""

from pathlib import Path
import json

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
TIF_DIR  = SAT_DIR / "tif"
RES_DIR  = ROOT / "results" / "ghana"
OUT_DIR  = RES_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEURON    = 1777
N_TILES   = 3
C_MAP_BG  = "#D5D8DC"
C_LBL     = "#222222"
C_RED     = "#C0392B"   # burn scar


def _norm(arr: np.ndarray) -> np.ndarray:
    valid = arr[arr > 0]
    if valid.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(valid, [2, 98])
    out = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    out[arr <= 0] = 0
    return out


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


def load_basemap():
    ghana = gpd.read_file(DATA_DIR / "gadm41_GHA_1.json").to_crs("EPSG:4326")
    lakes = gpd.read_file(DATA_DIR / "ne_10m_lakes.shp").to_crs("EPSG:4326")
    lakes = lakes.clip(ghana.total_bounds)
    return ghana, lakes


def draw_map(ax, comm_df, acts, ghana, lakes):
    ghana.plot(ax=ax, color="#F0F0F0", edgecolor="#AAAAAA", linewidth=0.4)
    lakes.plot(ax=ax, color="#AED6F1", edgecolor="none")
    ax.scatter(comm_df["gps_longitude"], comm_df["gps_latitude"],
               c=C_MAP_BG, s=5, linewidths=0, zorder=2)
    mask = acts > 0
    if mask.any():
        ax.scatter(comm_df.loc[mask, "gps_longitude"],
                   comm_df.loc[mask, "gps_latitude"],
                   c=C_RED, s=8, linewidths=0, zorder=3, alpha=0.9)
    ax.set_axis_off()
    ax.set_aspect("equal")


def _fmt_z(act: float) -> str:
    return "0" if act == 0 else f"{act:.2f}".rstrip("0").rstrip(".")


def main():
    # Load interpretation label
    interp_path = RES_DIR / "codes" / "nexis_no_adj" / "qwen25_72b" / "interpretations.json"
    with open(interp_path) as f:
        interps = json.load(f)
    entry = next(e for e in interps if e.get("neuron_idx") == NEURON)
    label = entry["label"]
    pval  = entry["pvalue"]

    # Load effect direction
    gate_df = pd.read_csv(RES_DIR / "gate" / "gate_Z.csv")
    gate_df["neuron_idx"] = gate_df["feature"].str.extract(r"neuron (\d+)").astype(float)
    direction_map = dict(zip(
        gate_df["neuron_idx"].dropna().astype(int),
        gate_df.loc[gate_df["neuron_idx"].notna(), "diff"],
    ))
    sign = "+" if direction_map.get(NEURON, 0) >= 0 else "-"

    # Community GPS centroids
    df = pd.read_stata(DATA_DIR / "LEAP1000 2015-2017 household data++.dta")
    comm_df = (df.groupby("comm")[["gps_latitude", "gps_longitude"]]
                 .mean().reset_index().set_index("comm"))

    # SAE activations
    sae_acts = np.load(SAT_DIR / "sae_activations.npy")
    comm_ids = np.load(SAT_DIR / "prithvi_comm_ids.npy")
    comm_df_aligned = comm_df.reindex(comm_ids).reset_index()

    all_acts    = sae_acts[:, NEURON]
    sorted_idx  = np.argsort(all_acts)[::-1]
    nonzero_idx = sorted_idx[all_acts[sorted_idx] > 0]
    zero_idx    = sorted_idx[all_acts[sorted_idx] == 0]

    top_keys = comm_ids[nonzero_idx[:N_TILES]].tolist()
    top_acts = all_acts[nonzero_idx[:N_TILES]].tolist()
    bot_keys = comm_ids[zero_idx[:N_TILES]].tolist()
    bot_acts = all_acts[zero_idx[:N_TILES]].tolist()

    ghana, lakes = load_basemap()

    # ── layout (inches) ───────────────────────────────────────────────────────
    map_w   = 1.15
    img_w   = 1.1
    div_w   = 0.05
    pad_w   = 0.08
    row_h   = 1.28
    title_h = 0.22
    top_pad = 0.03
    bot_pad = 0.05

    total_w = pad_w + map_w + pad_w + N_TILES*img_w + div_w + N_TILES*img_w + pad_w
    total_h = row_h + title_h + top_pad + bot_pad

    fig = plt.figure(figsize=(total_w, total_h), dpi=150)
    fig.patch.set_facecolor("white")

    def xf(x): return x / total_w
    def yf(y): return 1.0 - y / total_h

    # Title
    y_title_top = top_pad
    ax_t = fig.add_axes([0, yf(y_title_top + title_h), 1, title_h / total_h])
    ax_t.set_axis_off()
    title_str = f"Neuron {NEURON}: {label[0].upper() + label[1:]} ({sign}impact)"
    ax_t.text(0.01, 0.72, title_str, ha="left", va="center",
              fontsize=9, fontweight="normal", transform=ax_t.transAxes)

    y_img_top = yf(top_pad + title_h)
    h_img     = row_h / total_h
    w_map     = map_w / total_w
    w_img     = img_w / total_w
    w_div     = div_w / total_w

    # Map
    ax_map = fig.add_axes([xf(pad_w), y_img_top - h_img, w_map, h_img])
    draw_map(ax_map, comm_df_aligned, all_acts, ghana, lakes)

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
    x_div = x0_act + N_TILES * img_w
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
        out = OUT_DIR / f"figure_neural_1777.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
