"""
Generate transparent treatment maps matching the website style exactly.
Website colors/approach replicated from docs/index.html canvas rendering.
Run: python3 animations/gen_treatment_maps.py
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MPath
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "assets"
DATA = ROOT / "data"

# Exact website colors (from index.html canvas rendering)
COUNTRY_FILL = "#e8ede4"
COUNTRY_EDGE = "#b0bba8"
REGION_FILL  = "#e8f4e2"
REGION_EDGE  = "#b8d9b0"
# Treatment gradient endpoints (website: LO=control, HI=treated)
_LO = (44, 123, 182)
_HI = (215, 48, 39)

DOT_S  = 18
ALPHA  = 0.92


def _grad_color(t: float) -> tuple:
    """Interpolate between _LO and _HI at position t ∈ [0, 1]."""
    return tuple((_LO[i] + (_HI[i] - _LO[i]) * t) / 255 for i in range(3))


def _polygon_patch(rings, facecolor, edgecolor, lw, zorder):
    """Single PathPatch for one polygon (exterior ring + optional hole rings).
    Using a compound path so fill works correctly with holes."""
    verts, codes = [], []
    for ring in rings:
        arr = np.array(ring)[:, :2]
        if len(arr) < 2:
            continue
        verts.append(arr)
        codes += [MPath.MOVETO] + [MPath.LINETO] * (len(arr) - 2) + [MPath.CLOSEPOLY]
    if not verts:
        return None
    path = MPath(np.concatenate(verts), codes)
    return mpatches.PathPatch(
        path, facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, zorder=zorder,
    )


def _add_geojson(ax, geojson_path, facecolor, edgecolor, lw, zorder):
    """Draw all (Multi)Polygon features from a GeoJSON file."""
    with open(geojson_path) as f:
        gj = json.load(f)
    features = gj.get("features", [{"geometry": gj}])
    for feat in features:
        geom = feat.get("geometry", feat)
        polys = geom["coordinates"]
        if geom["type"] == "Polygon":
            polys = [polys]          # wrap so we can iterate uniformly
        for rings in polys:
            p = _polygon_patch(rings, facecolor, edgecolor, lw, zorder)
            if p is not None:
                ax.add_patch(p)


def _base_ax(figsize, dpi):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.axis("off")
    return fig, ax


def _draw_basemap(ax, regions_json, country_json):
    """Replicate website order: country fill → region fills → country outline."""
    # 1. Country fill (background)
    _add_geojson(ax, country_json, COUNTRY_FILL, COUNTRY_EDGE, 0.7, zorder=1)
    # 2. Region fills + borders on top of country
    _add_geojson(ax, regions_json, REGION_FILL,  REGION_EDGE,  0.4, zorder=2)
    # 3. Country outline only (no fill) on top for clean border
    _add_geojson(ax, country_json, "none", COUNTRY_EDGE, 0.9, zorder=3)


# ── Uganda ────────────────────────────────────────────────────────────────────

def uganda_map():
    csv_path = DATA / "uganda/UgandaDataProcessed.csv"
    IDX_LON, IDX_LAT, IDX_TRT = 3, 4, 40
    IDX_KEY = 6010          # column index, not value
    agg = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) <= IDX_KEY:
                continue
            k   = row[IDX_KEY].strip()
            lon = row[IDX_LON].strip()
            lat = row[IDX_LAT].strip()
            trt = row[IDX_TRT].strip()
            if not (k and lon and lat and trt):
                continue
            try:
                lon, lat, trt = float(lon), float(lat), float(trt)
            except ValueError:
                continue
            if k not in agg:
                agg[k] = [lon, lat, 0.0, 0]
            agg[k][2] += trt
            agg[k][3] += 1

    sites = [(v[0], v[1], v[2] / v[3]) for v in agg.values() if v[3] > 0]
    lons = [s[0] for s in sites]
    lats = [s[1] for s in sites]

    fig, ax = _base_ax((4.8, 5.0), dpi=180)
    _draw_basemap(ax, DOCS / "uganda_regions.json", DOCS / "uganda_country.json")

    # Sort by pct_treated so treated dots render on top of control
    for lon, lat, pct in sorted(sites, key=lambda s: s[2]):
        col = _grad_color(max(0.0, min(1.0, pct)))
        ax.scatter(lon, lat, c=[col], s=DOT_S, alpha=ALPHA, zorder=4,
                   edgecolors=[(0, 0, 0, 0.18)], linewidths=0.4)

    mx, my = 0.4, 0.4
    ax.set_xlim(min(lons) - mx, max(lons) + mx)
    ax.set_ylim(min(lats) - my, max(lats) + my)
    ax.set_aspect("equal")
    fig.tight_layout(pad=0.0)
    return fig


# ── Ghana ─────────────────────────────────────────────────────────────────────

def ghana_map():
    with open(DOCS / "ghana_communities.json") as f:
        comms = json.load(f)

    lons = [c["lon"] for c in comms]
    lats = [c["lat"] for c in comms]

    fig, ax = _base_ax((4.0, 6.2), dpi=180)
    _draw_basemap(ax, DOCS / "ghana_regions.json", DOCS / "ghana_country.json")

    for c in sorted(comms, key=lambda x: x.get("pct_treated", 0)):
        pct = c.get("pct_treated", 0) or 0
        col = _grad_color(max(0.0, min(1.0, pct)))
        ax.scatter(c["lon"], c["lat"], c=[col], s=DOT_S, alpha=ALPHA, zorder=4,
                   edgecolors=[(0, 0, 0, 0.15)], linewidths=0.4)

    mx, my = 0.3, 0.4
    ax.set_xlim(min(lons) - mx, max(lons) + mx)
    ax.set_ylim(min(lats) - my, max(lats) + my)
    ax.set_aspect("equal")
    fig.tight_layout(pad=0.0)
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    def _save(fig, path):
        fig.savefig(path, dpi=180, bbox_inches="tight", transparent=True)
        print(f"  saved → {path}")
        plt.close(fig)

    print("Uganda treatment map …")
    _save(uganda_map(), DOCS / "uganda_treatment_map.png")
    print("Ghana treatment map …")
    _save(ghana_map(), DOCS / "ghana_treatment_map.png")
    print("Done.")
