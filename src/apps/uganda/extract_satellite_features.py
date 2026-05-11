"""Extract Prithvi-EO embeddings from Landsat 5 GeoTIFF tiles.

Run once on the RCT tiles and once on the national grid tiles:

  python scripts/uganda/extract_satellite_features.py \\
    --tif-dir data/uganda/satellite/tif_rct \\
    --out-dir data/uganda/satellite/rct

  python scripts/uganda/extract_satellite_features.py \\
    --tif-dir data/uganda/satellite/tif_national \\
    --out-dir data/uganda/satellite/national

Band order in our Landsat 5 TIF files (download order SR_B1…B7):
  0: Blue  (SR_B1)    1: Green (SR_B2)    2: Red   (SR_B3)
  3: NIR   (SR_B4)    4: SWIR-1 (SR_B5)   5: SWIR-2 (SR_B7)

Prithvi expects B02(Blue),B03(Green),B04(Red),B05(NIR),B06(SWIR1),B07(SWIR2)
— identical to our band order, so no reindexing is needed.

Outputs (written to --out-dir):
  spectral_indices.csv        — 12 hand-crafted index summaries per tile
  prithvi_embeddings.npy      — (N, 768) mean-pooled patch token embeddings
  prithvi_site_keys.npy       — (N,) numeric IDs parsed from filenames
                                  RCT tiles → geo_long_lat_key values
                                  national tiles → grid IDs
"""

import argparse
import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── Band indices (TIF file order: SR_B1→B7 = Blue…SWIR2) ─────────────────────
BLUE, GREEN, RED, NIR, SWIR1, SWIR2 = 0, 1, 2, 3, 4, 5

# Landsat 5 TIF order is already Blue→SWIR2 = Prithvi's expected order.
PRITHVI_BAND_ORDER = [BLUE, GREEN, RED, NIR, SWIR1, SWIR2]

# Prithvi training statistics (HLS global, ×10000 reflectance scale).
# Our data is in [0,1] reflectance (after GEE ×0.0000275 − 0.2 scaling), so ÷10000.
PRITHVI_MEAN = (np.array([775.2, 1080.9, 1228.6, 2497.2, 2204.2, 1610.8]) / 10000).astype(np.float32)
PRITHVI_STD  = (np.array([1281.5, 1270.0, 1399.5, 1368.3, 1291.7, 1154.5]) / 10000).astype(np.float32)

IMG_SIZE   = 224
BATCH_SIZE = 16


# ── Prithvi loader ─────────────────────────────────────────────────────────────

def load_prithvi(device: torch.device) -> torch.nn.Module:
    """Download Prithvi-EO-1.0-100M from HuggingFace and return the encoder."""
    from huggingface_hub import hf_hub_download
    import importlib.util

    repo = 'ibm-nasa-geospatial/Prithvi-EO-1.0-100M'
    print(f"  Loading Prithvi from {repo} …")

    code_path   = hf_hub_download(repo, 'prithvi_mae.py')
    weight_path = hf_hub_download(repo, 'Prithvi_EO_V1_100M.pt')

    spec   = importlib.util.spec_from_file_location('prithvi_mae', code_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    PrithviMAE = module.PrithviMAE

    model = PrithviMAE(
        img_size=IMG_SIZE,
        patch_size=(1, 16, 16),
        num_frames=3,
        in_chans=6,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        encoder_only=True,
    )

    state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys in Prithvi checkpoint: {missing[:5]}")
    print(f"  Loaded ({len(state_dict)} params). "
          f"Unexpected (decoder, expected): {len(unexpected)}")

    model.eval().to(device)
    return model


# ── Preprocessing ──────────────────────────────────────────────────────────────

def load_tile(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)   # (6, H, W)


def tile_id_from_path(p: Path) -> int:
    """Parse trailing digits from stem: uganda_rct000296 → 296."""
    m = re.search(r'(\d+)$', p.stem)
    if not m:
        raise ValueError(f"Cannot parse numeric ID from filename: {p.name}")
    return int(m.group(1))


def preprocess_for_prithvi(bands: np.ndarray) -> torch.Tensor:
    """(6,H,W) → (1,6,3,224,224): normalise, resize, repeat T=3."""
    x = bands[PRITHVI_BAND_ORDER]                               # (6, H, W)
    x = (x - PRITHVI_MEAN[:, None, None]) / (PRITHVI_STD[:, None, None] + 1e-8)
    t = torch.from_numpy(x).unsqueeze(0)                        # (1, 6, H, W)
    t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE),
                      mode='bilinear', align_corners=False)
    t = t.unsqueeze(2).expand(-1, -1, 3, -1, -1).contiguous()  # (1, 6, 3, 224, 224)
    return t


# ── Spectral indices ───────────────────────────────────────────────────────────

def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = a + b
    return np.where(np.abs(denom) > 1e-6, (a - b) / denom, 0.0)


def spectral_indices(bands: np.ndarray) -> dict[str, float]:
    red, green, blue = bands[RED], bands[GREEN], bands[BLUE]
    nir, swir1, _    = bands[NIR], bands[SWIR1], bands[SWIR2]

    ndvi  = _safe_ratio(nir, red)
    ndwi  = _safe_ratio(green, nir)
    mndwi = _safe_ratio(green, swir1)
    ndbi  = _safe_ratio(swir1, nir)
    evi   = 2.5 * (nir - red) / np.where(
                np.abs(nir + 6*red - 7.5*blue + 1) > 1e-6,
                nir + 6*red - 7.5*blue + 1, 1e-6)
    bsi   = _safe_ratio(swir1 + red, nir + blue)

    out = {}
    for name, arr in [('ndvi', ndvi), ('ndwi', ndwi), ('mndwi', mndwi),
                      ('ndbi', ndbi), ('evi', evi), ('bsi', bsi)]:
        out[f'{name}_mean'] = float(arr.mean())
        out[f'{name}_std']  = float(arr.std())
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tif-dir',  required=True,
                   help='Directory of .tif tiles to process')
    p.add_argument('--out-dir',  required=True,
                   help='Output directory for embeddings and spectral indices')
    p.add_argument('--no-embed', action='store_true',
                   help='Skip Prithvi embeddings (spectral indices only)')
    p.add_argument('--device',   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    return p.parse_args()


def main():
    args    = parse_args()
    tif_dir = Path(args.tif_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device(args.device)

    tif_paths = sorted(tif_dir.glob('*.tif'))
    if not tif_paths:
        print(f"No .tif files found in {tif_dir}")
        return
    print(f"Found {len(tif_paths)} tiles in {tif_dir}")

    # ── 1. Spectral indices ────────────────────────────────────────────────────
    print("\n[1/2] Computing spectral indices …")
    rows = []
    for path in tqdm(tif_paths):
        bands = load_tile(path)
        row   = {'site_key': tile_id_from_path(path)}
        row.update(spectral_indices(bands))
        rows.append(row)

    df_idx = pd.DataFrame(rows).sort_values('site_key').reset_index(drop=True)
    out_csv = out_dir / 'spectral_indices.csv'
    df_idx.to_csv(out_csv, index=False)
    print(f"  Saved {len(df_idx)} rows → {out_csv}")

    if args.no_embed:
        return

    # ── 2. Prithvi embeddings ─────────────────────────────────────────────────
    print(f"\n[2/2] Extracting Prithvi embeddings (device={device}) …")
    model = load_prithvi(device)

    site_keys  = []
    embeddings = []

    with torch.no_grad():
        batch_tiles, batch_keys = [], []
        for i, path in enumerate(tqdm(tif_paths)):
            bands = load_tile(path)
            batch_tiles.append(preprocess_for_prithvi(bands))
            batch_keys.append(tile_id_from_path(path))

            if len(batch_tiles) == args.batch_size or i == len(tif_paths) - 1:
                x = torch.cat(batch_tiles).to(device)   # (B, 6, 3, 224, 224)

                block_outs = model.encoder.forward_features(x)
                tokens     = block_outs[-1]              # (B, N_tokens+1, 768)
                patch_embs = tokens[:, 1:, :].mean(dim=1)  # (B, 768)

                embeddings.append(patch_embs.cpu().numpy())
                site_keys.extend(batch_keys)
                batch_tiles, batch_keys = [], []

    embeddings = np.concatenate(embeddings, axis=0)   # (N, 768)
    site_keys  = np.array(site_keys)
    order      = np.argsort(site_keys)
    embeddings = embeddings[order]
    site_keys  = site_keys[order]

    np.save(out_dir / 'prithvi_embeddings.npy', embeddings)
    np.save(out_dir / 'prithvi_site_keys.npy',  site_keys)
    print(f"  Embeddings : {embeddings.shape}  → {out_dir / 'prithvi_embeddings.npy'}")
    print(f"  Site keys  : {site_keys.shape}   → {out_dir / 'prithvi_site_keys.npy'}")


if __name__ == '__main__':
    main()
