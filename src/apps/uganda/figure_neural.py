"""
Neural discovery figures for Uganda (skilled_employed & log_biz_assets).

Each figure has one row per neural discovery:
  [map] | [top-3 satellite images] | [divider] | [bottom-3 satellite images]

Usage:
    python src/apps/uganda/figure_neural.py
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from PIL import Image

plt.rcParams.update({
    "text.usetex":        False,
    "font.family":        "serif",
    "font.serif":         ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset":   "cm",
})

ROOT      = Path(__file__).resolve().parents[3]
DATA_DIR  = ROOT / "data" / "uganda"
TIF_DIR   = DATA_DIR / "satellite" / "tif_rct"
MAP_DIR   = DATA_DIR / "map"
SPEC_PATH = DATA_DIR / "satellite" / "rct" / "spectral_indices.csv"
RES_DIR   = ROOT / "results" / "uganda" / "prithvi_l5_1024"
OUT_DIR   = ROOT / "results" / "uganda" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


def load_tile(key: int, size: int = 112) -> np.ndarray | None:
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

def load_basemap():
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
            img_arr = load_tile(int(key))
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
            img_arr = load_tile(int(key))
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

def main():
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

    uganda, lakes = load_basemap()

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

    build_figure(rows_se, OUT_DIR / "figure_neural_skilled_employed.pdf",
                 uganda=uganda, lakes=lakes)
    build_figure(rows_se, OUT_DIR / "figure_neural_skilled_employed.png",
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

    build_figure(rows_ba, OUT_DIR / "figure_neural_log_biz_assets.pdf",
                 uganda=uganda, lakes=lakes)
    build_figure(rows_ba, OUT_DIR / "figure_neural_log_biz_assets.png",
                 uganda=uganda, lakes=lakes)


if __name__ == "__main__":
    main()
