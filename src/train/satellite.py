"""
Uganda satellite dataset — CSV-band loading, normalization, and embedding pipeline.

Each site stores three per-band CSV files (Green, NIR, SWIR reflectance).
Pixels exceeding FILL_THRESHOLD are treated as no-data / cloud-masked.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from .backbone import extract_embeddings, load_model, _auto_device, _progress

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UgandaSatelliteDataset(Dataset):
    """Dataset for Uganda Landsat images stored as per-band CSV files.

    Each site has three files:
        GeoKey{key}_BAND1.csv  — Green reflectance
        GeoKey{key}_BAND2.csv  — NIR reflectance
        GeoKey{key}_BAND3.csv  — SWIR reflectance

    Pixels with value ≥ FILL_THRESHOLD are masked (no-data).
    Each band is percentile-stretched to [0, 1] independently.
    Channels are stacked as false-color: NIR→R, Green→G, SWIR→B.
    """

    FILL_THRESHOLD = 5000
    PCT_LO, PCT_HI = 2, 98

    def __init__(
        self,
        img_dir: Union[str, Path],
        keys: list[int],
        transform=None,
        preload_workers: int = 0,
    ):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.keys = [k for k in keys if self._has_all_bands(k)]
        dropped = len(keys) - len(self.keys)
        if dropped:
            print(f"UgandaSatelliteDataset: skipped {dropped}/{len(keys)} keys "
                  f"with missing band files")

        self._cache: dict[int, np.ndarray] | None = None
        if preload_workers > 0:
            print(f"Preloading {len(self.keys)} sites from NFS "
                  f"({preload_workers} threads)...")
            def _load(key):
                return key, self._build_array(key)
            with ThreadPoolExecutor(max_workers=preload_workers) as ex:
                results = list(_progress(
                    ex.map(_load, self.keys),
                    total=len(self.keys), desc="preload", unit="site",
                ))
            self._cache = dict(results)
            print(f"Preload complete — "
                  f"{sum(v.nbytes for v in self._cache.values()) / 1e6:.0f} MB in RAM")

    def _has_all_bands(self, key: int) -> bool:
        return all(
            (self.img_dir / f"GeoKey{key}_BAND{b}.csv").exists() for b in [1, 2, 3]
        )

    def _load_stretched(self, key: int, band: int) -> np.ndarray:
        arr = pd.read_csv(
            self.img_dir / f"GeoKey{key}_BAND{band}.csv", header=None
        ).values.astype(np.float32)
        arr[arr >= self.FILL_THRESHOLD] = np.nan
        valid = arr[~np.isnan(arr)]
        lo = np.percentile(valid, self.PCT_LO)
        hi = np.percentile(valid, self.PCT_HI)
        s = np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        s[np.isnan(arr)] = 0.0
        return s

    def _build_array(self, key: int) -> np.ndarray:
        """Return (H, W, 3) float32 false-color array for one site."""
        channels = [self._load_stretched(key, b) for b in [2, 1, 3]]
        return np.stack(channels, axis=-1)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        key = self.keys[idx]
        img_arr = self._cache[key] if self._cache is not None else self._build_array(key)
        img = Image.fromarray((img_arr * 255).astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, key


# ---------------------------------------------------------------------------
# Data-derived normalization
# ---------------------------------------------------------------------------

def compute_dataset_stats(
    dataset: UgandaSatelliteDataset,
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Compute per-channel mean and std from the dataset (after resize/crop)."""
    prep = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    original_transform = dataset.transform
    dataset.transform = prep

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    channel_sum    = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    n_pixels = 0

    for imgs, _ in loader:
        b, c, h, w = imgs.shape
        channel_sum    += imgs.sum(dim=(0, 2, 3))
        channel_sum_sq += (imgs ** 2).sum(dim=(0, 2, 3))
        n_pixels       += b * h * w

    mean = (channel_sum / n_pixels).tolist()
    std  = ((channel_sum_sq / n_pixels - torch.tensor(mean) ** 2)
            .clamp(min=0).sqrt()).tolist()

    dataset.transform = original_transform
    return tuple(mean), tuple(std)


def make_uganda_transform(
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> transforms.Compose:
    """Resize → crop → normalize with data-derived stats."""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ---------------------------------------------------------------------------
# Full pipeline convenience wrapper
# ---------------------------------------------------------------------------

def embed_uganda_sites(
    img_dir: Union[str, Path],
    keys: list[int],
    model_name: str = "dinov2_vitb14",
    batch_size: int = 8,
    num_workers: int = 0,
    device: str | None = None,
    mode: str = "patch",
    verbose: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """Full pipeline: Uganda CSV bands → vision backbone → embeddings.

    Computes data-derived normalization stats before extraction.
    Works with any model in MODEL_REGISTRY (torchhub or HuggingFace).
    """
    if device is None:
        device = _auto_device()

    dataset = UgandaSatelliteDataset(img_dir, keys, transform=None,
                                     preload_workers=num_workers)

    if verbose:
        print(f"Computing per-channel stats from {len(dataset)} sites...")
    mean, std = compute_dataset_stats(dataset, batch_size=batch_size)
    if verbose:
        print(f"  mean={tuple(f'{v:.3f}' for v in mean)}  "
              f"std={tuple(f'{v:.3f}' for v in std)}")

    dataset.transform = make_uganda_transform(mean, std)

    model = load_model(model_name, device=device)
    if verbose:
        print(f"Extracting {model_name} [{mode}] embeddings "
              f"for {len(dataset)} sites on {device}...")

    embeddings = extract_embeddings(dataset, model, batch_size=batch_size,
                                    num_workers=num_workers, device=device,
                                    mode=mode, verbose=verbose)
    return embeddings, dataset.keys
