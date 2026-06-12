"""
Teaser figure (Figure 1): satellite tiles for the two key discoveries.

Layout: 2 rows (Ghana / Uganda) × 2 columns (feature active / inactive).
Each cell: one satellite tile with a GATE badge (top-right) and a tiny
feature label (bottom-left). Left strip carries programme/outcome info.

Ghana  — ephemeral waterways (neuron 3821), expenditure
Uganda — perennial river presence (Z_339),  skilled employment

Usage:
    python src/apps/figure_teaser.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image as PILImage

mpl.rcParams.update({
    "text.usetex":                 False,
    "font.family":                 "serif",
    "font.serif":                  ["cmr10", "DejaVu Serif", "Times New Roman", "serif"],
    "mathtext.fontset":            "cm",
    "axes.formatter.use_mathtext": True,
})

ROOT = Path(__file__).resolve().parents[2]

# ── Paths ──────────────────────────────────────────────────────────────────────
GHANA_TIF  = ROOT / "data" / "ghana"    / "satellite" / "tif"
UGANDA_TIF = ROOT / "data" / "uganda"   / "satellite" / "tif_rct"
OUT_DIR    = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Tile selection ─────────────────────────────────────────────────────────────
# Ghana  neuron 3821 — ephemeral waterways
#   active:   community 951  (activation 2.85, highest in RCT pool)
#   inactive: community 14   (activation 0,  representative dry savannah)
# Uganda Z_339 — perennial river presence
#   active:   site 92  (highest activation)
#   inactive: site 217 (zero activation, typical dryland site)
CELLS = [
    # (country, col, tif_dir, file_fmt, file_id,
    #  gate_str, gate_units, feature_label, accent_color, swir_idx)
    # Ghana L8: bands SR_B4,B3,B2,B5,B6,B7 → idx 0=Red,1=Green,2=Blue,3=NIR,4=SWIR1,5=SWIR2
    # Uganda L5: bands SR_B1..B7 → idx 0=Blue,1=Green,2=Red,3=NIR,4=SWIR1,5=SWIR2
    ("ghana",  "active",   GHANA_TIF,  "ghana_comm{:04d}.tif",  1265,
     "+42.9", "GH¢/mo", "ephemeral waterways",   "#2E86C1", 5),
    ("ghana",  "inactive", GHANA_TIF,  "ghana_comm{:04d}.tif",  1141,
     "+6.0",  "GH¢/mo", "no waterways",           "#2E86C1", 5),
    ("uganda", "active",   UGANDA_TIF, "uganda_rct{:06d}.tif",   92,
     "+0.09", "skilled empl.", "perennial river",       "#C0392B", 4),
    ("uganda", "inactive", UGANDA_TIF, "uganda_rct{:06d}.tif",  217,
     "+0.33", "skilled empl.", "no river",              "#C0392B", 4),
]

# Row metadata (order matters: ghana first, uganda second)
ROWS = [
    dict(
        country  = "ghana",
        prog     = "LEAP 1000",
        region   = "N. Ghana",
        outcome  = "expenditure",
        feature  = "ephemeral waterways",
        color    = "#2E86C1",
    ),
    dict(
        country  = "uganda",
        prog     = "YOP",
        region   = "N. Uganda",
        outcome  = "skilled employment",
        feature  = "perennial river",
        color    = "#C0392B",
    ),
]


# ── Image loading ──────────────────────────────────────────────────────────────

def _norm(arr: np.ndarray) -> np.ndarray:
    valid = arr[arr > 0]
    if valid.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(valid, [2, 98])
    out = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    out[arr <= 0] = 0
    return out


def load_tile(path: Path, swir_idx: int = 5, size: int = 224) -> np.ndarray | None:
    """False-colour NIR/Green/SWIR composite (matches existing paper figures).

    tifffile reads multi-band GeoTIFFs as (H, W, bands) float64.
    Band layout (0-indexed):
      Ghana  L8 (SR_B4,B3,B2,B5,B6,B7): Green=1, NIR=3, SWIR2=5  → swir_idx=5
      Uganda L5 (SR_B1..B7):             Green=1, NIR=3, SWIR1=4  → swir_idx=4
    """
    if not path.exists():
        return None
    data = tifffile.imread(str(path)).astype(np.float32)  # (H, W, bands)
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    green = data[:, :, 1]
    nir   = data[:, :, 3]
    swir  = data[:, :, swir_idx]
    arr = (np.stack([_norm(nir), _norm(green), _norm(swir)], axis=-1) * 255).astype(np.uint8)
    return np.asarray(PILImage.fromarray(arr).resize((size, size), PILImage.Resampling.BICUBIC))


# ── Figure ─────────────────────────────────────────────────────────────────────

def make_figure():
    # ── layout constants (inches) ──────────────────────────────────────────────
    label_w  = 0.85   # left strip for row labels
    tile_w   = 1.90   # tile width
    tile_h   = 1.90   # tile height (square)
    col_gap  = 0.12   # gap between active / inactive
    row_gap  = 0.16   # gap between Ghana / Uganda rows
    hdr_h    = 0.28   # column header strip height
    pad_l    = 0.05   # outer left padding
    pad_r    = 0.10   # outer right padding
    pad_t    = 0.05   # outer top padding
    pad_b    = 0.08   # outer bottom padding

    n_rows = 2
    n_cols = 2
    total_w = pad_l + label_w + n_cols * tile_w + (n_cols - 1) * col_gap + pad_r
    total_h = pad_t + hdr_h + n_rows * tile_h + (n_rows - 1) * row_gap + pad_b

    def xf(x_in): return x_in / total_w
    def yf(y_in): return 1.0 - y_in / total_h   # top-down

    fig = plt.figure(figsize=(total_w, total_h), dpi=200)
    fig.patch.set_facecolor("white")

    # ── column headers ─────────────────────────────────────────────────────────
    col_labels = ["feature active", "feature inactive"]
    col_x_starts = [
        pad_l + label_w,
        pad_l + label_w + tile_w + col_gap,
    ]
    y_hdr_center = yf(pad_t + hdr_h * 0.55)
    for i, (lbl, x_in) in enumerate(zip(col_labels, col_x_starts)):
        cx = xf(x_in + tile_w / 2)
        fig.text(cx, y_hdr_center, lbl,
                 ha="center", va="center",
                 fontsize=8, color="#333333",
                 fontstyle="italic",
                 transform=fig.transFigure)

    # ── tiles ──────────────────────────────────────────────────────────────────
    col_order = ["active", "inactive"]

    for row_i, row_meta in enumerate(ROWS):
        y_top_in = pad_t + hdr_h + row_i * (tile_h + row_gap)
        y_bot_in = y_top_in + tile_h

        # Row label strip
        cx_label = xf(pad_l + label_w * 0.5)
        cy_tile  = yf(y_top_in + tile_h / 2)
        fig.text(cx_label, cy_tile,
                 f"{row_meta['prog']}\n{row_meta['region']}",
                 ha="center", va="center",
                 fontsize=7.5, fontweight="bold", color="#222222",
                 linespacing=1.5,
                 transform=fig.transFigure)
        fig.text(cx_label, cy_tile - 0.048,
                 f"({row_meta['outcome']})",
                 ha="center", va="center",
                 fontsize=6.5, color="#555555",
                 fontstyle="italic",
                 transform=fig.transFigure)

        # Thin accent bar on left edge of row
        bar_w_in = 0.045
        bar_x_in = pad_l + 0.02
        margin   = 0.06
        fig.add_artist(mpl.patches.Rectangle(
            (xf(bar_x_in), yf(y_bot_in - margin)),
            bar_w_in / total_w,
            (tile_h + 2 * margin) / total_h,
            color=row_meta["color"], alpha=0.60,
            transform=fig.transFigure, zorder=5,
            linewidth=0,
        ))

        for col_i, col_key in enumerate(col_order):
            x_left_in = pad_l + label_w + col_i * (tile_w + col_gap)

            # Find the matching cell spec
            cell = next(
                c for c in CELLS
                if c[0] == row_meta["country"] and c[1] == col_key
            )
            (_, _, tif_dir, file_fmt, file_id,
             gate_str, gate_units, feat_label, accent, swir_idx) = cell

            tif_path = tif_dir / file_fmt.format(file_id)
            img = load_tile(tif_path, swir_idx=swir_idx, size=256)

            ax = fig.add_axes([
                xf(x_left_in),
                yf(y_bot_in),
                tile_w / total_w,
                tile_h / total_h,
            ])
            if img is not None:
                ax.imshow(img, aspect="auto")
            else:
                ax.set_facecolor("#1a1a1a")
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        color="white", fontsize=9, transform=ax.transAxes)

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0)

            # ── GATE badge (top-right) ─────────────────────────────────────────
            gate_txt = f"GATE {gate_str}\n{gate_units}"
            ax.text(0.97, 0.97, gate_txt,
                    ha="right", va="top",
                    fontsize=6.8, color="white", fontweight="bold",
                    linespacing=1.3,
                    bbox=dict(
                        boxstyle="round,pad=0.22",
                        facecolor="#000000",
                        alpha=0.58,
                        edgecolor="none",
                    ),
                    transform=ax.transAxes, zorder=10)

            # ── feature label (bottom-left) ────────────────────────────────────
            ax.text(0.04, 0.04, feat_label,
                    ha="left", va="bottom",
                    fontsize=6.2, color="white",
                    fontstyle="italic",
                    bbox=dict(
                        boxstyle="round,pad=0.18",
                        facecolor="#000000",
                        alpha=0.48,
                        edgecolor="none",
                    ),
                    transform=ax.transAxes, zorder=10)

    # ── save ───────────────────────────────────────────────────────────────────
    for ext in ["pdf", "png"]:
        out = OUT_DIR / f"figure_teaser.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white",
                    dpi=200 if ext == "png" else None)
        print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
