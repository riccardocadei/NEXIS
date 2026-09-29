"""Local NDVI increase between the 2015 and 2017 community composites.

Used by `figure_neural_combined.py` (boxes on the temporal panel of Figure 5)
and `table_temporal.py` (column "Local NDVI increase (% area)").

Rule
----
NDVI = (NIR - Red) / (NIR + Red) on the Landsat-8 annual median composites
(`data/ghana/satellite/tif/` for 2015, `tif_2017/` for 2017; band order B4, B3,
B2, B5, B6, B7). Water is masked: MNDWI = (Green - SWIR1) / (Green + SWIR1) > 0
or NDVI < 0 in either year, dilated by WATER_BUFFER_PX. On the rest:

1. dNDVI = NDVI_2017 - NDVI_2015, minus the tile's median dNDVI. Every tile
   greens overall (median +0.015 to +0.043), a year-to-year shift of the
   composites, so only change beyond it counts as local.
2. Gaussian smoothing, sigma = SMOOTH_SIGMA_PX (30 m).
3. Pixels above THRESHOLD (about 2 robust SD of the smoothed local dNDVI,
   0.024, pooled over the six waterway-active communities).
4. 3 x 3 binary opening, then connected patches of at least MIN_PATCH_PX
   (1.8 ha) are kept.

`increase_area_pct` reports the kept patches as % of the valid (non-water) area.
`largest_patches` returns the largest patch, plus the second largest if it is
at least SECOND_PATCH_FRAC of the largest's area (the ones boxed in the figure).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi

ROOT     = Path(__file__).resolve().parents[3]
SAT_DIR  = ROOT / "data" / "ghana" / "satellite"
TIF_2015 = SAT_DIR / "tif"
TIF_2017 = SAT_DIR / "tif_2017"

RED, GREEN, NIR, SWIR1 = 0, 1, 3, 4   # TIF band order: B4, B3, B2, B5, B6, B7
THRESHOLD         = 0.05   # local dNDVI (beyond the tile median), after smoothing
SMOOTH_SIGMA_PX   = 1.0    # Gaussian sigma, 30 m pixels
MIN_PATCH_PX      = 20     # 20 px x 0.09 ha = 1.8 ha
WATER_BUFFER_PX   = 2
SECOND_PATCH_FRAC = 0.75   # keep the 2nd-largest patch if >= 75% of the largest
PX_HA             = 0.09   # one 30 m pixel


def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def local_change(comm_id: int) -> dict | None:
    """Per-pixel local dNDVI and the increase patches of one community.

    Returns {"valid", "local_dndvi", "labels", "sizes"} on the native 30 m
    grid (labels: patches >= MIN_PATCH_PX numbered 1..n, sizes in pixels),
    or None if a tile is missing.
    """
    p15 = TIF_2015 / f"ghana_comm{comm_id:04d}.tif"
    p17 = TIF_2017 / f"ghana_comm{comm_id:04d}.tif"
    if not (p15.exists() and p17.exists()):
        return None
    a, b = _read(p15), _read(p17)
    ndvi  = lambda x: (x[NIR] - x[RED]) / (x[NIR] + x[RED])
    mndwi = lambda x: (x[GREEN] - x[SWIR1]) / (x[GREEN] + x[SWIR1])
    n15, n17 = ndvi(a), ndvi(b)
    water = (mndwi(a) > 0) | (mndwi(b) > 0) | (n15 < 0) | (n17 < 0)
    valid = ~ndi.binary_dilation(water, iterations=WATER_BUFFER_PX)

    d = n17 - n15
    loc = np.where(valid, d - np.median(d[valid]), 0.0)
    # smooth over valid pixels only (normalised convolution)
    w = ndi.gaussian_filter(valid.astype(np.float32), SMOOTH_SIGMA_PX)
    sm = ndi.gaussian_filter(loc, SMOOTH_SIGMA_PX) / np.maximum(w, 1e-6)
    m = ndi.binary_opening((sm > THRESHOLD) & valid, np.ones((3, 3)))

    lab, n = ndi.label(m)
    sizes = np.asarray(ndi.sum(m, lab, range(1, n + 1))) if n else np.zeros(0)
    keep = np.flatnonzero(sizes >= MIN_PATCH_PX)
    order = keep[np.argsort(sizes[keep])[::-1]]          # largest first
    relabel = np.zeros(n + 1, dtype=np.int32)
    relabel[order + 1] = np.arange(1, len(order) + 1)
    return {"valid": valid, "local_dndvi": np.where(valid, loc, np.nan),
            "labels": relabel[lab], "sizes": sizes[order]}


def increase_area_pct(comm_id: int) -> float | None:
    """Area of all increase patches, % of the valid (non-water) area."""
    c = local_change(comm_id)
    if c is None:
        return None
    return 100.0 * (c["labels"] > 0)[c["valid"]].mean()


def largest_patches(comm_id: int) -> list[dict]:
    """[{"mask", "area_ha", "local_dndvi"}] for the patch(es) boxed in the figure.

    local_dndvi is the patch mean of the unsmoothed dNDVI minus the tile median.
    """
    c = local_change(comm_id)
    if c is None or len(c["sizes"]) == 0:
        return []
    n_keep = 2 if (len(c["sizes"]) > 1
                   and c["sizes"][1] >= SECOND_PATCH_FRAC * c["sizes"][0]) else 1
    out = []
    for k in range(1, n_keep + 1):
        mask = c["labels"] == k
        out.append({"mask": mask, "area_ha": float(mask.sum() * PX_HA),
                    "local_dndvi": float(np.nanmean(c["local_dndvi"][mask]))})
    return out
