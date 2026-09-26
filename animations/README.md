# Animations

Manim scenes for the project website (`docs/index.html`) and the README.

## Environment

The scenes run in the `manim` conda env (Manim Community v0.20.1), separate from the
`crl` env used by the rest of the repo:

```bash
MANIM=/nfs/scistore19/locatgrp/rcadei/miniconda3/envs/manim/bin
# or: conda run -n manim manim ...
```

Renders go to `animations/media/` (git-ignored, regenerable). `-qh` alone renders at
1080p**60**; the website assets are 1080p30, so pass `--fps 30`.

## Scenes

| File | Scene | What it shows | Website asset |
|---|---|---|---|
| `s01_pipeline.py` | `Pipeline` | From Tokens to Policy: satellite tile → Prithvi-EO → SAE atoms, plus survey features; NEXIS selects one atom and two language dummies, labelled with the certified YOP skilled-employment modifiers. | `docs/assets/s01_pipeline.mp4` (Method section) and `docs/assets/pipeline.gif` (README) |
| `s02_method.py` | `Selection` | NEXIS (paper Algorithm 1) on three candidate neurons: forward steps with gates α/\|S̄\| = 0.05/3, 0.05/2, 0.05/1 until the forward step stops, then one terminal backward step at α/m = 0.05/3. | `docs/assets/nexis_method.mp4` (Method section) |
| `s02_snapshots.py` | `Snapshot1`–`Snapshot4` | Still frames of `Selection`: forward steps 1–3 and the terminal backward step. | none |
| `s03_results.py` | `Results` | Summary of the two applications (YOP, LEAP 1000). Reads the treatment maps below. | none |
| `gen_treatment_maps.py` | (matplotlib) | Treatment-assignment maps used by `s03_results.py`. | `docs/assets/{uganda,ghana}_treatment_map.png` |

## Render commands

From the repo root:

```bash
cd animations

# Method animation -> docs/assets/nexis_method.mp4
$MANIM/manim -qh --fps 30 s02_method.py Selection
cp media/videos/s02_method/1080p30/Selection.mp4 ../docs/assets/nexis_method.mp4

# Pipeline animation -> docs/assets/s01_pipeline.mp4 and docs/assets/pipeline.gif
$MANIM/manim -qh --fps 30 s01_pipeline.py Pipeline
cp media/videos/s01_pipeline/1080p30/Pipeline.mp4 ../docs/assets/s01_pipeline.mp4
ffmpeg -y -i ../docs/assets/s01_pipeline.mp4 \
  -vf "fps=12,scale=860:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5" \
  ../docs/assets/pipeline.gif

# Snapshots (PNG, media/images/s02_snapshots/)
$MANIM/manim -sqh s02_snapshots.py Snapshot1 Snapshot2 Snapshot3 Snapshot4

# Results summary (not used on the site)
python3 gen_treatment_maps.py          # needs data/uganda/UgandaDataProcessed.csv; writes docs/assets/*_treatment_map.png
$MANIM/manim -qh --fps 30 s03_results.py Results
```

Use `-ql` (480p15) for quick previews. `s01_pipeline.py` reads `assets/sample_tile.png`;
if it is missing, it rebuilds it from a Ghana GeoTIFF under `data/` (needs `rasterio`) or
falls back to a plain square.
