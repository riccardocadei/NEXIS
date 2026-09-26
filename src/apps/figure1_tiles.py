"""
Figure 1 of the paper: two satellite tiles, one per anti-poverty programme.

  (a) Perennial river   -> results/figures/figure1/river.pdf
      Uganda YOP, RCT site 92 (Kamdini parish, Apac district), Landsat 7
      2005-2007 median composite. Site 92 is the top-activating RCT site of
      SAE neuron 339 ("perennial river presence") in
      results/uganda/prithvi_l5_1024/site_features.npz.
  (b) Ephemeral waterway -> results/figures/figure1/waterways.pdf
      Ghana LEAP 1000, community 1265 (Bongo district), Landsat 8 2015
      median composite. Community 1265 is one of the 6 LEAP communities where
      SAE neuron 3821 ("ephemeral waterways") is active (4th highest,
      data/ghana/satellite/sae_activations.npy).

Each tile is a false-colour composite (R = NIR, G = Green, B = SWIR; SWIR1 for
Uganda Landsat 7, SWIR2 for Ghana Landsat 8), every band stretched
to its 2-98th percentile, drawn at native resolution in a 3x3 inch canvas
with a "<district>, <country> (<year>)" label at the top right.

This reproduces paper/iclr27/figures/{uganda/river.pdf, ghana/waterways.pdf}
(first made on 2026-06-12 by an uncommitted script; the colour composite is the
one of src/apps/figure_teaser.py, which draws the website teaser).

Usage (from the repo root):
    python src/apps/figure1_tiles.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "figures" / "figure1"

# One entry per panel; the district label is read from the survey data.
TILES = [
    dict(name="river",
         tif=ROOT / "data/uganda/satellite/tif_rct/uganda_rct000092.tif",
         swir_idx=4,          # Uganda L7 bands: SR_B1,B2,B3,B4,B5,B7 -> 1=Green 3=NIR 4=SWIR1
         country="Uganda", year=2007),
    dict(name="waterways",
         tif=ROOT / "data/ghana/satellite/tif/ghana_comm1265.tif",
         swir_idx=5,          # Ghana L8 bands: B4,B3,B2,B5,B6,B7 -> 1=Green 3=NIR 5=SWIR2
         country="Ghana", year=2015),
]


def district_of(name: str) -> str:
    """Look the district label up in the survey data rather than hard-coding it."""
    if name == "river":
        df = pd.read_csv(ROOT / "data/uganda/UgandaDataProcessed.csv",
                         usecols=["geo_long_lat_key", "district"])
        d = df.loc[df.geo_long_lat_key == 92, "district"].dropna().unique()
    else:
        df = pd.read_stata(ROOT / "data/ghana/survey/LEAP1000 2015-2017 household data++.dta",
                           columns=["comm", "district"])
        d = df.loc[df.comm == 1265, "district"].dropna().astype(str).unique()
    assert len(d) == 1, f"ambiguous district for {name}: {d}"
    return str(d[0]).title()


def _norm(arr: np.ndarray) -> np.ndarray:
    valid = arr[arr > 0]
    if valid.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(valid, [2, 98])
    out = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    out[arr <= 0] = 0
    return out


def composite(path: Path, swir_idx: int) -> np.ndarray:
    """NIR / Green / SWIR false-colour composite, uint8, native resolution."""
    data = tifffile.imread(str(path)).astype(np.float32)  # (H, W, bands)
    rgb = np.stack([_norm(data[:, :, 3]), _norm(data[:, :, 1]),
                    _norm(data[:, :, swir_idx])], axis=-1)
    return (rgb * 255).astype(np.uint8)


def make_tile(spec: dict) -> None:
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"{spec['name']}.{ext}"
        fig.savefig(out, bbox_inches="tight", pad_inches=0, dpi=200)
        print(f"Saved -> {out.relative_to(ROOT)}")
    plt.close(fig)


if __name__ == "__main__":
    for spec in TILES:
        make_tile(spec)
