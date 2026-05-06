"""
Find satellite tiles that visually resemble the letters N, E, X, I, S.

Improvements over v1:
- Cloud filtering (skip tiles dominated by bright reflectance)
- HOG descriptors (capture oriented stroke structure, not just edge density)
- False-color NIR-R-G rendering for better landscape contrast
- Deduplicated selection (each tile used at most once across letters)
- Top-20 candidates per letter
"""

import os
import glob
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from skimage.feature import hog
from skimage.transform import resize
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# ── config ─────────────────────────────────────────────────────────────────
DATA_DIRS   = [
    "data/ghana/satellite/tif_national",
    "data/uganda/satellite/tif_national",
]
OUT_DIR     = "outputs/ghana/nexis_teaser"
LETTERS     = list("NEXIS")
SIZE        = 128          # processing resolution
HOG_PPC     = 16           # pixels per HOG cell
HOG_CPB     = 2            # cells per HOG block
TOP_K       = 20           # candidates to keep per letter
CLOUD_THR   = 0.55         # skip tile if fraction of bright pixels > this
BRIGHT_THR  = 0.70         # per-pixel brightness threshold for cloud mask

os.makedirs(OUT_DIR, exist_ok=True)


# ── helpers ─────────────────────────────────────────────────────────────────

def pct_norm(arr, lo_p=2, hi_p=98):
    valid = arr[arr > -0.01]
    if valid.size < 10:
        return np.zeros_like(arr)
    lo, hi = np.percentile(valid, [lo_p, hi_p])
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)


def load_tile(path):
    """
    Returns (rgb_uint8, is_cloudy).
    rgb uses NIR-R-G false color for better landscape contrast.
    """
    with rasterio.open(path) as src:
        r   = src.read(1).astype(np.float32)
        g   = src.read(2).astype(np.float32)
        b   = src.read(3).astype(np.float32)   # Blue
        nir = src.read(4).astype(np.float32)

    # Cloud mask: clouds are bright in ALL visible bands
    visible_mean = (pct_norm(r) + pct_norm(g) + pct_norm(b)) / 3.0
    cloud_frac   = float((visible_mean > BRIGHT_THR).mean())
    is_cloudy    = cloud_frac > CLOUD_THR

    # False-color: NIR → R channel, R → G channel, G → B channel
    rgb = np.stack([pct_norm(nir), pct_norm(r), pct_norm(g)], axis=-1)
    return (rgb * 255).astype(np.uint8), is_cloudy


def tile_to_hog(rgb_uint8, size=SIZE):
    """Return HOG feature vector for a tile."""
    img = Image.fromarray(rgb_uint8).convert("L")
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = gaussian(arr, sigma=0.8)
    feat = hog(arr,
               pixels_per_cell=(HOG_PPC, HOG_PPC),
               cells_per_block=(HOG_CPB, HOG_CPB),
               orientations=9,
               feature_vector=True)
    return feat


def letter_to_hog(letter, size=SIZE, stroke_width=None):
    """Render a bold letter and return its HOG features."""
    img  = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    try:
        fs   = int(size * 0.82)
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), letter, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) // 2 - bbox[0]
    y = (size - th) // 2 - bbox[1]
    draw.text((x, y), letter, fill=255, font=font)

    arr  = np.array(img, dtype=np.float32) / 255.0
    feat = hog(arr,
               pixels_per_cell=(HOG_PPC, HOG_PPC),
               cells_per_block=(HOG_CPB, HOG_CPB),
               orientations=9,
               feature_vector=True)
    return feat


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ── build letter HOG templates ───────────────────────────────────────────────
print("Building letter HOG templates …")
templates = {L: letter_to_hog(L) for L in LETTERS}

# Visual sanity check of letter renders
fig, axes = plt.subplots(1, len(LETTERS), figsize=(len(LETTERS) * 2, 2.5))
for ax, L in zip(axes, LETTERS):
    img  = Image.new("L", (SIZE, SIZE), color=0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(SIZE * 0.82))
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), L, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    draw.text(((SIZE-tw)//2 - bbox[0], (SIZE-th)//2 - bbox[1]), L, fill=255, font=font)
    ax.imshow(np.array(img), cmap="gray");  ax.set_title(L, fontsize=14);  ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "letter_templates.png"), dpi=150);  plt.close()

# ── score all tiles ───────────────────────────────────────────────────────────
tile_paths  = sorted(
    p for d in DATA_DIRS for p in glob.glob(os.path.join(d, "*.tif"))
)
print(f"Scoring {len(tile_paths)} tiles (skipping cloudy ones) …")

scores      = {L: [] for L in LETTERS}   # (score, path, rgb)
n_cloudy    = 0

for path in tqdm(tile_paths):
    try:
        rgb, is_cloudy = load_tile(path)
    except Exception:
        continue
    if is_cloudy:
        n_cloudy += 1
        continue
    try:
        feat = tile_to_hog(rgb)
    except Exception:
        continue
    for L in LETTERS:
        s = cosine_sim(feat, templates[L])
        scores[L].append((s, path))

print(f"  Skipped {n_cloudy}/{len(tile_paths)} cloudy tiles "
      f"({100*n_cloudy/len(tile_paths):.1f}%)")

# ── deduplicated greedy selection ─────────────────────────────────────────────
print("\nSelecting best unique tile per letter …")
used_paths  = set()
best_paths  = {}   # letter → list[path] (top unique candidates)

# Sort letters by max available score (hardest letter first)
letter_order = sorted(LETTERS,
                      key=lambda L: scores[L][0][0] if scores[L] else 0,
                      reverse=False)

for L in letter_order:
    ranked = sorted(scores[L], key=lambda x: x[0], reverse=True)
    unique = [(s, p) for s, p in ranked if p not in used_paths]
    best_paths[L] = [p for _, p in unique[:TOP_K]]
    if best_paths[L]:
        used_paths.add(best_paths[L][0])
    sc = unique[0][0] if unique else float("nan")
    print(f"  {L}: score={sc:.4f}  path={best_paths[L][0] if best_paths[L] else 'none'}")

# ── save top-K strip per letter ───────────────────────────────────────────────
for L in LETTERS:
    ranked = sorted(scores[L], key=lambda x: x[0], reverse=True)
    unique = [(s, p) for s, p in ranked if p != best_paths.get(L, [None])[0] or True]
    # show top-K regardless (for browsing)
    top = [(s, p) for s, p in ranked][:TOP_K]
    k   = min(TOP_K, len(top))
    fig, axes = plt.subplots(1, k, figsize=(k * 2.5, 3))
    if k == 1:
        axes = [axes]
    for ax, (sc, p) in zip(axes, top):
        rgb, _ = load_tile(p)
        ax.imshow(rgb);  ax.set_title(f"{sc:.3f}", fontsize=7);  ax.axis("off")
    fig.suptitle(f"Letter  {L}  — top {k} (HOG, NIR false-color)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"top_{L}.png"), dpi=150);  plt.close()
    print(f"  Saved top_{L}.png")

# ── assemble teaser panel ────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 4))
gs  = gridspec.GridSpec(1, 5, wspace=0.05)

for i, L in enumerate(LETTERS):
    path = best_paths[L][0] if best_paths[L] else None
    ax   = fig.add_subplot(gs[i])
    if path:
        rgb, _ = load_tile(path)
        ax.imshow(rgb)
    ax.set_title(L, fontsize=24, fontweight="bold", pad=6, color="white",
                 bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5))
    ax.axis("off")

out_panel = os.path.join(OUT_DIR, "nexis_teaser.png")
plt.savefig(out_panel, dpi=200, bbox_inches="tight", facecolor="black")
plt.close()
print(f"\nTeaser panel → {out_panel}")
