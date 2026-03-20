"""
DINOv2 feature extraction for image datasets.

Two extraction modes:
  mode='cls'   → one 768-d vector per image  (N, d)
  mode='patch' → one vector per 14×14 patch  (N, n_patches, d)
                 ViT-B/14 on 224×224 gives 256 patches per image.

For the Uganda satellite data use UgandaSatelliteDataset, which reads the
per-band CSV files, builds the false-color composite, and normalizes using
statistics computed from the actual data (not ImageNet).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Literal, Union


# Embedding dimensionality for each DINOv2 variant
DINO_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

# Standard ImageNet normalization (used by ImageFolderFlat / generic images)
_IMAGENET_NORM = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)

_RESIZE_CROP = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

_DINO_TRANSFORM = transforms.Compose([*_RESIZE_CROP.transforms, _IMAGENET_NORM])


def load_dinov2(model_name: str = "dinov2_vitb14", device: str | None = None) -> torch.nn.Module:
    """Load a pretrained DINOv2 model from torch.hub."""
    if device is None:
        device = _auto_device()
    model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
    model.eval().to(device)
    return model


# ---------------------------------------------------------------------------
# Dataset: generic image folder
# ---------------------------------------------------------------------------

class ImageFolderFlat(Dataset):
    """Loads every image under a directory as RGB PIL images."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(
        self,
        root: Union[str, Path],
        transform=_DINO_TRANSFORM,
        extensions: set[str] | None = None,
        recursive: bool = True,
    ):
        self.root = Path(root)
        self.transform = transform
        exts = extensions or self.IMAGE_EXTENSIONS
        glob = self.root.rglob("*") if recursive else self.root.glob("*")
        self.paths = sorted(p for p in glob if p.suffix.lower() in exts)
        if not self.paths:
            raise ValueError(f"No images found in {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, str(self.paths[idx])


# ---------------------------------------------------------------------------
# Dataset: Uganda satellite bands (CSV format)
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

    Args:
        img_dir: Directory containing the GeoKey CSV files.
        keys: List of integer site keys to include.
        transform: Torchvision transform applied after building the PIL image.
                   Use make_uganda_transform() with data-derived stats.
    """

    FILL_THRESHOLD = 5000
    PCT_LO, PCT_HI = 2, 98

    def __init__(
        self,
        img_dir: Union[str, Path],
        keys: list[int],
        transform=None,
    ):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.keys = [k for k in keys if self._has_all_bands(k)]
        dropped = len(keys) - len(self.keys)
        if dropped:
            print(f"UgandaSatelliteDataset: skipped {dropped}/{len(keys)} keys with missing band files")

    def _has_all_bands(self, key: int) -> bool:
        return all((self.img_dir / f"GeoKey{key}_BAND{b}.csv").exists() for b in [1, 2, 3])

    def _load_stretched(self, key: int, band: int) -> np.ndarray:
        arr = pd.read_csv(
            self.img_dir / f"GeoKey{key}_BAND{band}.csv", header=None
        ).values.astype(np.float32)
        arr[arr >= self.FILL_THRESHOLD] = np.nan
        valid = arr[~np.isnan(arr)]
        lo = np.percentile(valid, self.PCT_LO)
        hi = np.percentile(valid, self.PCT_HI)
        s = np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        s[np.isnan(arr)] = 0.0   # fill pixels → black
        return s

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        key = self.keys[idx]
        # False-color composite: NIR(B2)→R, Green(B1)→G, SWIR(B3)→B
        channels = [self._load_stretched(key, b) for b in [2, 1, 3]]
        img_arr = np.stack(channels, axis=-1)                  # (H, W, 3) float32 in [0, 1]
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
    """Compute per-channel mean and std from the dataset (after resize/crop, before normalize).

    Returns:
        mean: (R_mean, G_mean, B_mean)
        std:  (R_std,  G_std,  B_std)
    """
    prep = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),   # → float32 in [0, 1]
    ])
    original_transform = dataset.transform
    dataset.transform = prep

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    channel_sum   = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    n_pixels = 0

    for imgs, _ in loader:
        # imgs: (B, 3, H, W)
        b, c, h, w = imgs.shape
        channel_sum    += imgs.sum(dim=(0, 2, 3))
        channel_sum_sq += (imgs ** 2).sum(dim=(0, 2, 3))
        n_pixels       += b * h * w

    mean = (channel_sum / n_pixels).tolist()
    std  = ((channel_sum_sq / n_pixels - torch.tensor(mean) ** 2).clamp(min=0).sqrt()).tolist()

    dataset.transform = original_transform
    return tuple(mean), tuple(std)


def make_uganda_transform(
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> transforms.Compose:
    """Build the standard resize→crop→normalize transform with data-derived stats."""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    dataset: Dataset,
    model: torch.nn.Module,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str | None = None,
    mode: Literal["cls", "patch"] = "cls",
    verbose: bool = True,
) -> np.ndarray:
    """Run the dataset through DINOv2 and return embeddings.

    Args:
        dataset: PyTorch Dataset returning (image_tensor, *metadata).
        model: DINOv2 model (from load_dinov2).
        batch_size: Images per forward pass.
        num_workers: DataLoader workers.
        device: Inference device.
        mode: 'cls'   → (N, d) one vector per image.
              'patch' → (N, n_patches, d) one vector per 14×14 patch.
                        For ViT-B/14 on 224×224 input: n_patches = 256.
                        Flatten to (N*256, d) before passing to SAE.
        verbose: Print progress.

    Returns:
        Float32 array of shape (N, d) or (N, n_patches, d).
    """
    if device is None:
        device = _auto_device()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
    )

    all_embeds = []
    for i, batch in enumerate(loader):
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        imgs = imgs.to(device, non_blocking=True)

        feats = model.forward_features(imgs)

        if mode == "cls":
            out = feats["x_norm_clstoken"]          # (B, d)
        else:
            out = feats["x_norm_patchtokens"]        # (B, n_patches, d)

        all_embeds.append(out.cpu().float().numpy())

        if verbose and i % 10 == 0:
            n_done = min((i + 1) * batch_size, len(dataset))
            print(f"  [{n_done}/{len(dataset)}] batch {i+1}/{len(loader)}")

    return np.concatenate(all_embeds, axis=0)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def embed_image_folder(
    root: Union[str, Path],
    model_name: str = "dinov2_vitb14",
    batch_size: int = 32,
    num_workers: int = 0,
    device: str | None = None,
    mode: Literal["cls", "patch"] = "cls",
    verbose: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Embed all images in a folder with DINOv2.

    Returns:
        embeddings: (N, d) or (N, n_patches, d).
        paths: List of N file path strings.
    """
    if device is None:
        device = _auto_device()
    dataset = ImageFolderFlat(root)
    model   = load_dinov2(model_name, device=device)
    if verbose:
        print(f"Extracting {model_name} [{mode}] embeddings for {len(dataset)} images on {device}...")
    embeddings = extract_embeddings(dataset, model, batch_size=batch_size,
                                    num_workers=num_workers, device=device,
                                    mode=mode, verbose=verbose)
    return embeddings, [str(p) for p in dataset.paths]


def embed_uganda_sites(
    img_dir: Union[str, Path],
    keys: list[int],
    model_name: str = "dinov2_vitb14",
    batch_size: int = 8,
    num_workers: int = 0,
    device: str | None = None,
    mode: Literal["cls", "patch"] = "patch",
    verbose: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """Full pipeline: Uganda CSV bands → DINOv2 → embeddings.

    Computes data-derived normalization stats before extraction.

    Args:
        img_dir: Directory containing GeoKey CSV files.
        keys: Site keys to embed.
        model_name: DINOv2 variant.
        batch_size: Images per forward pass (keep small; each image loads 3 CSVs).
        num_workers: DataLoader workers.
        device: Inference device.
        mode: 'cls' → (N, d).  'patch' → (N, 256, d), flatten before SAE.
        verbose: Print progress.

    Returns:
        embeddings: Float32 array of shape (N, d) or (N, n_patches, d).
        valid_keys: Site keys actually embedded (those with all 3 band files present).
    """
    if device is None:
        device = _auto_device()

    # Build dataset without normalization to compute stats
    dataset = UgandaSatelliteDataset(img_dir, keys, transform=None)

    if verbose:
        print(f"Computing per-channel stats from {len(dataset)} sites...")
    mean, std = compute_dataset_stats(dataset, batch_size=batch_size)
    if verbose:
        print(f"  mean={tuple(f'{v:.3f}' for v in mean)}  std={tuple(f'{v:.3f}' for v in std)}")

    dataset.transform = make_uganda_transform(mean, std)

    model = load_dinov2(model_name, device=device)
    if verbose:
        print(f"Extracting {model_name} [{mode}] embeddings for {len(dataset)} sites on {device}...")

    embeddings = extract_embeddings(dataset, model, batch_size=batch_size,
                                    num_workers=num_workers, device=device,
                                    mode=mode, verbose=verbose)
    return embeddings, dataset.keys


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
