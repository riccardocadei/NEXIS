"""Plot 2×6 grid of neuron 3821 (ephemeral waterways) communities: 2015 top, 2017 bottom.

Arrows between rows mark communities where VLM detected intensification (2015→2017).
Symbols next to each arrow indicate what changed.
"""

from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import rasterio
from PIL import Image as PILImage

mpl.rcParams.update({
    "text.usetex":   False,
    "font.family":   "serif",
    "font.serif":    ["cmr10", "DejaVu Serif", "Times New Roman", "serif"],
    "mathtext.fontset":              "cm",
    "axes.formatter.use_mathtext":   True,
})

ROOT     = Path(__file__).resolve().parents[2]
TIF_2015 = ROOT / "data"    / "ghana" / "satellite" / "tif"
TIF_2017 = ROOT / "data"    / "ghana" / "satellite" / "tif_2017"
OUT_PATH = ROOT / "results" / "ghana" / "temporal" / "neuron_3821_grid.png"
OUT_PDF  = ROOT / "results" / "ghana" / "temporal" / "neuron_3821_grid.pdf"

COMMUNITIES = [
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


def _load_changes() -> dict:
    """Load VLM-detected changes from neuron_3821_temporal.json.

    Returns {comm_id: [(symbol, label, color), ...]} for communities where
    the VLM detected meaningful intensification (non-stable overall verdict).
    """
    import json
    path = ROOT / "results" / "ghana" / "temporal" / "neuron_3821_temporal.json"
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
        if "cropland" in e.get("agricultural_change", "").lower() and (
            "expansion" in e.get("agricultural_change", "").lower()
            or "intensif" in e.get("agricultural_change", "").lower()
        ):
            items.append(("+", "cropland", CHANGE_COLORS["cropland"]))
        if "denser" in e.get("vegetation_change", "").lower() or (
            "increase" in e.get("vegetation_change", "").lower()
        ):
            items.append(("+", "vegetation", CHANGE_COLORS["vegetation"]))
        if items:
            changes[cid] = items
    return changes


def load_fc(tif_path: Path, size: int = 224) -> np.ndarray | None:
    if not tif_path.exists():
        return None
    with rasterio.open(tif_path) as src:
        green = src.read(2).astype(np.float32)
        nir   = src.read(4).astype(np.float32)
        swir2 = src.read(6).astype(np.float32)

    def norm(b):
        valid = b[b > 0]
        if valid.size == 0:
            return np.zeros_like(b)
        lo, hi = np.percentile(valid, [2, 98])
        out = np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1)
        out[b <= 0] = 0
        return out

    arr = (np.stack([norm(nir), norm(green), norm(swir2)], axis=-1) * 255).astype(np.uint8)
    return np.array(PILImage.fromarray(arr).resize((size, size), PILImage.BICUBIC))


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    CHANGES = _load_changes()
    print(f"Loaded changes for communities: {sorted(CHANGES.keys())}")

    n = len(COMMUNITIES)

    # Extra vertical space between rows for arrows
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 7.2),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.04})

    for col, comm in enumerate(COMMUNITIES):
        cid  = comm["comm_id"]
        name = f"ghana_comm{cid:04d}.tif"

        for row, tif_dir in enumerate([TIF_2015, TIF_2017]):
            ax  = axes[row, col]
            img = load_fc(tif_dir / name)

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

            if row == 1:
                ax.set_xlabel(f"Community {cid}", fontsize=12, labelpad=5)

    # Year labels on left
    for row, year in enumerate([2015, 2017]):
        axes[row, 0].set_ylabel(str(year), fontsize=15, fontweight="bold",
                                labelpad=8, rotation=90, va="center")

    # Arrows + symbols for communities with VLM-detected intensification
    fig.canvas.draw()  # needed to resolve transforms
    for col, comm in enumerate(COMMUNITIES):
        cid = comm["comm_id"]
        if cid not in CHANGES:
            continue

        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        # Convert axes coords → figure coords; shift arrow left to make room for labels
        arrow_x = 0.35  # left of centre in axes coords
        p_top = fig.transFigure.inverted().transform(
            ax_top.transAxes.transform((arrow_x, -0.02))
        )
        p_bot = fig.transFigure.inverted().transform(
            ax_bot.transAxes.transform((arrow_x,  1.02))
        )

        # Arrow in figure coordinates
        arrow = FancyArrowPatch(
            posA=p_top, posB=p_bot,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.8,
            color="#444",
            transform=fig.transFigure,
            zorder=10,
        )
        fig.add_artist(arrow)

        # Labels to the right of the arrow, in black
        mid_fig = 0.5 * (p_top + p_bot)
        label_x = p_top[0] + 0.012
        for k, (sym, label, _color) in enumerate(CHANGES[cid]):
            offset_y = 0.025 * (k - (len(CHANGES[cid]) - 1) / 2)
            fig.text(
                label_x, mid_fig[1] + offset_y,
                f"{sym} {label}",
                ha="left", va="center",
                fontsize=13, color="black", fontweight="bold",
                transform=fig.transFigure,
            )

    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_PDF,           bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")
    print(f"Saved → {OUT_PDF}")
    plt.show()


if __name__ == "__main__":
    main()
