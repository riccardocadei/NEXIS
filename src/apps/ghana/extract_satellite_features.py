"""Extract community-level satellite features for Ghana LEAP 1000.

Uses Prithvi-EO-1.0-100M (IBM/NASA), a ViT-Large MAE pre-trained on HLS
(Harmonized Landsat Sentinel) data with the exact same 6 spectral bands as
our Landsat 8 tiles.

Band order in TIF files (matches download order SR_B4,B3,B2,B5,B6,B7):
  0: Red   (655 nm)    1: Green (562 nm)    2: Blue  (482 nm)
  3: NIR   (865 nm)    4: SWIR-1 (1609 nm)  5: SWIR-2 (2201 nm)

Prithvi expects bands in order B02,B03,B04,B05,B06,B07 (Blue→SWIR2), so we
reindex channels [2,1,0,3,4,5] before feeding the model.

Temporal input: Prithvi was trained with T=3 timesteps. For our single annual
composite we replicate the tile 3× along the time axis — the model still
extracts spatially meaningful features; it just sees the same image for all
three "dates".

Outputs (written to data/ghana/satellite/):
  spectral_indices.csv        — 12 hand-crafted index summaries per community
  prithvi_embeddings.npy      — (162, 768) mean-pooled patch token embeddings
  prithvi_comm_ids.npy        — (162,) community IDs aligned with embeddings
"""

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── Band indices (TIF file order) ─────────────────────────────────────────────
RED, GREEN, BLUE, NIR, SWIR1, SWIR2 = 0, 1, 2, 3, 4, 5

# TIF → Prithvi band reorder: B02(Blue), B03(Green), B04(Red), B05(NIR), B06(SWIR1), B07(SWIR2)
PRITHVI_BAND_ORDER = [BLUE, GREEN, RED, NIR, SWIR1, SWIR2]

# Normalization: Prithvi training statistics (HLS global, ×10000 reflectance scale).
# Our data is in [0,1] reflectance, so we divide by 10000.
PRITHVI_MEAN = (np.array([775.2, 1080.9, 1228.6, 2497.2, 2204.2, 1610.8]) / 10000).astype(np.float32)
PRITHVI_STD  = (np.array([1281.5, 1270.0, 1399.5, 1368.3, 1291.7, 1154.5]) / 10000).astype(np.float32)

IMG_SIZE   = 224
BATCH_SIZE = 16


# ── Prithvi loader ─────────────────────────────────────────────────────────────

def load_prithvi(device: torch.device) -> torch.nn.Module:
    """Download Prithvi model code + weights from HuggingFace and return the encoder."""
    from huggingface_hub import hf_hub_download
    import importlib.util

    repo = 'ibm-nasa-geospatial/Prithvi-EO-1.0-100M'
    print(f"  Loading Prithvi from {repo} …")

    # Download model source and weights
    code_path   = hf_hub_download(repo, 'prithvi_mae.py')
    weight_path = hf_hub_download(repo, 'Prithvi_EO_V1_100M.pt')

    # Dynamically import PrithviMAE
    spec   = importlib.util.spec_from_file_location('prithvi_mae', code_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    PrithviMAE = module.PrithviMAE

    model = PrithviMAE(
        img_size=IMG_SIZE,
        patch_size=(1, 16, 16),
        num_frames=3,        # matches checkpoint; we repeat single tile 3×
        in_chans=6,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        encoder_only=True,   # skip decoder; saves memory
    )

    state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
    # weights are stored under 'encoder.*' and 'decoder.*' at the top level
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys in Prithvi checkpoint: {missing[:5]}")
    print(f"  Loaded weights ({len(state_dict)} params). "
          f"Unexpected keys (decoder, expected): {len(unexpected)}")

    model.eval().to(device)
    return model


# ── Preprocessing ──────────────────────────────────────────────────────────────

def load_tile(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)   # (6, H, W)


def comm_id_from_path(p: Path) -> int:
    import re
    m = re.search(r'(\d+)$', p.stem)
    if not m:
        raise ValueError(f"Cannot parse numeric ID from filename: {p.name}")
    return int(m.group(1))


def preprocess_for_prithvi(bands: np.ndarray) -> torch.Tensor:
    """(6,H,W) → (1, 6, 3, 224, 224): reorder bands, normalise, resize, repeat T=3."""
    # Reorder to Prithvi's B02…B07 convention
    x = bands[PRITHVI_BAND_ORDER]                           # (6, H, W)
    # Normalise with Prithvi's HLS training statistics
    x = (x - PRITHVI_MEAN[:, None, None]) / (PRITHVI_STD[:, None, None] + 1e-8)
    # Resize to 224×224
    t = torch.from_numpy(x).unsqueeze(0)                    # (1, 6, H, W)
    t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    # Add time dim and repeat 3× to match Prithvi's T=3 expectation
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
    p.add_argument('--tif-dir',  default='../../data/ghana/satellite/tif')
    p.add_argument('--out-dir',  default='../../data/ghana/satellite')
    p.add_argument('--no-embed', action='store_true',
                   help='Skip Prithvi embedding extraction (spectral indices only)')
    p.add_argument('--device',   default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args       = parse_args()
    script_dir = Path(__file__).parent
    tif_dir    = (script_dir / args.tif_dir).resolve()
    out_dir    = (script_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device     = torch.device(args.device)

    tif_paths = sorted(tif_dir.glob('*.tif'))
    print(f"Found {len(tif_paths)} tiles in {tif_dir}")

    # ── 1. Spectral indices ────────────────────────────────────────────────────
    print("\n[1/2] Computing spectral indices …")
    rows = []
    for path in tqdm(tif_paths):
        bands = load_tile(path)
        row   = {'comm_id': comm_id_from_path(path)}
        row.update(spectral_indices(bands))
        rows.append(row)

    df_idx = pd.DataFrame(rows).sort_values('comm_id').reset_index(drop=True)
    out_csv = out_dir / 'spectral_indices.csv'
    df_idx.to_csv(out_csv, index=False)
    print(f"  Saved {len(df_idx)} rows → {out_csv}")

    if args.no_embed:
        return

    # ── 2. Prithvi embeddings ─────────────────────────────────────────────────
    print(f"\n[2/2] Extracting Prithvi embeddings (device={device}) …")
    model = load_prithvi(device)

    comm_ids   = []
    embeddings = []

    with torch.no_grad():
        batch_tiles, batch_ids = [], []
        for i, path in enumerate(tqdm(tif_paths)):
            bands = load_tile(path)
            tile  = preprocess_for_prithvi(bands)   # (1, 6, 3, 224, 224)
            batch_tiles.append(tile)
            batch_ids.append(comm_id_from_path(path))

            if len(batch_tiles) == BATCH_SIZE or i == len(tif_paths) - 1:
                x = torch.cat(batch_tiles).to(device)   # (B, 6, 3, 224, 224)

                # forward_features returns list of per-block outputs, last is normed
                # shape of each: (B, N_tokens+1, 768)
                block_outs = model.encoder.forward_features(x)
                tokens = block_outs[-1]               # (B, 197, 768)

                # mean-pool over patch tokens (skip CLS at position 0)
                patch_embs = tokens[:, 1:, :].mean(dim=1)   # (B, 768)
                embeddings.append(patch_embs.cpu().numpy())
                comm_ids.extend(batch_ids)
                batch_tiles, batch_ids = [], []

    embeddings = np.concatenate(embeddings, axis=0)
    comm_ids   = np.array(comm_ids)
    order      = np.argsort(comm_ids)
    embeddings = embeddings[order]
    comm_ids   = comm_ids[order]

    np.save(out_dir / 'prithvi_embeddings.npy', embeddings)
    np.save(out_dir / 'prithvi_comm_ids.npy',   comm_ids)
    print(f"  Embeddings: {embeddings.shape}  → {out_dir / 'prithvi_embeddings.npy'}")
    print(f"  Comm IDs:   {comm_ids.shape}    → {out_dir / 'prithvi_comm_ids.npy'}")


if __name__ == '__main__':
    main()
