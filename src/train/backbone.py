"""
Vision feature extraction — generic backbone loading and embedding extraction.

Supports two backends:
  hf      — HuggingFace AutoModel (DINOv2 / DINOv3, 14×14 or 16×16 patches)
  prithvi — IBM/NASA Prithvi-EO geospatial foundation model (16×16 patches)
            Prithvi is pretrained on Landsat-8 / Sentinel-2 data and expects
            6 spectral bands.  Our 3-band false-colour (NIR, Green, SWIR1) is
            mapped into the corresponding Landsat-8 positions; the remaining 3
            channels are zero-padded.

Two extraction modes:
  mode='cls'   → one d-dim vector per image  (N, d)
  mode='patch' → one vector per patch token  (N, n_patches, d)
                 ViT/14 → 256 patches on 224²
                 ViT/16 → 196 patches on 224²  (DINOv3, Prithvi)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Literal, Union

try:
    from tqdm import tqdm as _tqdm
    def _progress(it, **kw): return _tqdm(it, **kw)
except ImportError:
    def _progress(it, **kw): return it


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Maps short name → (backend, identifier, embed_dim)
# backend  : "hf", "prithvi", or "siglip"
# identifier: HF repo id
MODEL_REGISTRY: dict[str, tuple[str, str, int]] = {
    # ── HuggingFace DINOv2 (14×14 patches → 256 patches on 224²) ─────────
    "dinov2":       ("hf", "facebook/dinov2-base",                          768),
    "dinov2_large": ("hf", "facebook/dinov2-large",                        1024),
    # ── HuggingFace DINOv3 (16×16 patches → 196 patches on 224²) ─────────
    "dinov3":       ("hf", "facebook/dinov3-vitb16-pretrain-lvd1689m",      768),
    "dinov3_large": ("hf", "facebook/dinov3-vitl16-pretrain-lvd1689m",     1024),
    # ── IBM/NASA Prithvi-EO (16×16 patches → 196 patches on 224²) ────────
    # Landsat-8 / Sentinel-2 geospatial foundation model.
    # 3-band input is zero-padded to 6 bands at the correct spectral positions.
    "prithvi":      ("prithvi", "ibm-nasa-geospatial/Prithvi-EO-1.0-100M", 768),
    # ── SigLIP (patch/14 → 729 patches on 384²) ───────────────────────────
    # Google SigLIP SO400M — best natural-image ViT from the ECI benchmark.
    # No CLS token; CLS mode uses mean-pooling of all patch tokens.
    # Preprocessing: 384×384, normalize mean=0.5 std=0.5.
    "siglip":       ("siglip", "google/siglip-so400m-patch14-384",         1152),
}

# Backwards-compat alias used by older code
DINO_DIMS = {k: v[2] for k, v in MODEL_REGISTRY.items()}


def model_embed_dim(model_name: str) -> int:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. "
                         f"Known: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name][2]


# ---------------------------------------------------------------------------
# Standard transforms
# ---------------------------------------------------------------------------

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

# SigLIP SO400M preprocessing: 384×384, normalise to [-1, 1].
_SIGLIP_TRANSFORM = transforms.Compose([
    transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


# ---------------------------------------------------------------------------
# Prithvi wrapper
# ---------------------------------------------------------------------------

class _PrithviWrapper(torch.nn.Module):
    """Adapt Prithvi-EO to the standard ``pixel_values → last_hidden_state`` API.

    Prithvi is a 6-band temporal ViT (PatchEmbed expects (B, C, T, H, W)).
    Our false-colour stack has 3 bands (NIR, Green, SWIR1 at indices 0, 1, 2).
    We place them into their correct Landsat-8 positions within the 6-channel
    tensor (0-indexed order: B2, B3, B4, B5, B6, B7):

        our idx 0 (NIR)   → Landsat B5 → channel position 3
        our idx 1 (Green) → Landsat B3 → channel position 1
        our idx 2 (SWIR1) → Landsat B6 → channel position 4

    Remaining channels (B2, B4, B7) are zero-padded.

    PrithviViT has a CLS token; ``_n_prefix_tokens`` is 1 so ``extract_embeddings``
    skips it and returns the 196 patch tokens in patch mode.
    """

    # 0-indexed channel positions in 6-band Landsat-8 order (B2,B3,B4,B5,B6,B7)
    _CHANNEL_POSITIONS: list[int] = [3, 1, 4]  # NIR→B5, Green→B3, SWIR1→B6
    _n_prefix_tokens: int = 1  # CLS token

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor):
        # pixel_values: (B, 3, H, W) — our 3-band false-colour
        B, _, H, W = pixel_values.shape
        x6 = torch.zeros(B, 6, H, W,
                         device=pixel_values.device, dtype=pixel_values.dtype)
        for our_ch, landsat_pos in enumerate(self._CHANNEL_POSITIONS):
            x6[:, landsat_pos] = pixel_values[:, our_ch]

        # PatchEmbed expects (B, C, T, H, W); T=1 time step
        x6 = x6.unsqueeze(2)  # (B, 6, 1, H, W)

        # Use forward_features (no masking) to get all patch tokens
        # Returns list of per-block outputs; last entry is normed: (B, 1+n_patches, d)
        block_outs = self.model.encoder.forward_features(x6)
        hs = block_outs[-1]  # (B, 1+196, 768)

        return type("_Out", (), {"last_hidden_state": hs})()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str | None = None) -> torch.nn.Module:
    """Load a pretrained vision backbone by registry name.

    Attaches two attributes to the returned model:
      _n_prefix_tokens: number of non-patch tokens before the first patch token
                        in last_hidden_state (CLS + any register tokens).
                        0 for Prithvi (no CLS token).
    """
    if device is None:
        device = _auto_device()

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. "
                         f"Known: {list(MODEL_REGISTRY)}")

    backend, identifier, _ = MODEL_REGISTRY[model_name]

    if backend == "siglip":
        from transformers import SiglipVisionModel
        model = SiglipVisionModel.from_pretrained(identifier)
        # SigLIP has no CLS token; all tokens are patch tokens.
        # extract_embeddings uses mean-pooling when _backend == "siglip".
        model._n_prefix_tokens = 0
        model._backend = "siglip"
        model.eval().to(device)
        return model

    if backend == "prithvi":
        # Prithvi-EO stores weights as a raw .pt checkpoint (no model.safetensors).
        # Load the architecture from the HuggingFace repo's prithvi_mae.py and
        # instantiate manually.
        import importlib.util
        import yaml
        from huggingface_hub import hf_hub_download

        arch_path = hf_hub_download(identifier, "prithvi_mae.py")
        ckpt_path = hf_hub_download(identifier, "Prithvi_EO_V1_100M.pt")
        cfg_path  = hf_hub_download(identifier, "config.yaml")

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        ma = cfg["model_args"]

        spec = importlib.util.spec_from_file_location("prithvi_mae", arch_path)
        prithvi_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prithvi_mod)

        mae = prithvi_mod.PrithviMAE(
            img_size=ma["img_size"],
            patch_size=ma["patch_size"],
            num_frames=ma["num_frames"],
            in_chans=ma["in_chans"],
            embed_dim=ma["embed_dim"],
            depth=ma["depth"],
            num_heads=ma["num_heads"],
            decoder_embed_dim=ma["decoder_embed_dim"],
            decoder_depth=ma["decoder_depth"],
            decoder_num_heads=ma["decoder_num_heads"],
        )
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        mae.load_state_dict(state_dict, strict=True)

        model = _PrithviWrapper(mae)
        model.eval().to(device)
        return model

    from transformers import AutoModel
    from transformers.configuration_utils import PretrainedConfig
    # Workaround: some TimmWrapper configs store num_labels=null which causes
    # the property setter to call range(None) → TypeError.
    _orig = PretrainedConfig._create_id_label_maps
    PretrainedConfig._create_id_label_maps = (
        lambda self, n: _orig(self, n) if n is not None else None
    )
    try:
        model = AutoModel.from_pretrained(identifier)
    finally:
        PretrainedConfig._create_id_label_maps = _orig

    n_registers = getattr(model.config, "num_register_tokens", 0)
    model._n_prefix_tokens = 1 + n_registers  # CLS + optional register tokens

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
    """Run the dataset through a vision backbone and return embeddings.

    Handles both torchhub DINOv2 (forward_features API) and HuggingFace
    models (last_hidden_state API) transparently via model._backend.

    Args:
        mode: 'cls'   → (N, d)
              'patch' → (N, n_patches, d)
                        14×14 patch model on 224² → n_patches=256
                        16×16 patch model on 224² → n_patches=196
    """
    if device is None:
        device = _auto_device()

    n_prefix = getattr(model, "_n_prefix_tokens", 1)  # CLS + optional registers

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
    )

    all_embeds = []
    bar = _progress(loader, desc=f"embed-{mode}", unit="batch",
                    disable=not verbose, dynamic_ncols=True)
    for batch in bar:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        imgs = imgs.to(device, non_blocking=True)

        # HuggingFace: last_hidden_state shape = (B, n_prefix + n_patches, d)
        hs = model(pixel_values=imgs).last_hidden_state
        if mode == "cls":
            if getattr(model, "_backend", None) == "siglip":
                # SigLIP has no CLS token — mean-pool all patch tokens
                out = hs.mean(dim=1)        # (B, d)
            else:
                out = hs[:, 0, :]           # (B, d)  — CLS token
        else:
            out = hs[:, n_prefix:, :]   # (B, n_patches, d)

        all_embeds.append(out.cpu().float().numpy())

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
    """Embed all images in a folder."""
    if device is None:
        device = _auto_device()
    dataset = ImageFolderFlat(root)
    model   = load_model(model_name, device=device)
    if verbose:
        print(f"Extracting {model_name} [{mode}] embeddings "
              f"for {len(dataset)} images on {device}...")
    embeddings = extract_embeddings(dataset, model, batch_size=batch_size,
                                    num_workers=num_workers, device=device,
                                    mode=mode, verbose=verbose)
    return embeddings, [str(p) for p in dataset.paths]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
