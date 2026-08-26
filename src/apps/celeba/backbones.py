"""
Backbone registry for the CelebA semi-synthetic pipeline.

The pipeline (embed → SAE → NEXIS sweeps → figures) is backbone-agnostic: the
only things that change are the timm model, its input normalisation, and the
file names of the artefacts it produces.

SigLIP keeps its original, untagged file names (``siglip.npy``, ``sae_k20.npy``,
``results/celeba/experiment/k20/sae/``) so all existing results and the paths
referenced in the paper stay valid.  Every other backbone is tagged, e.g.
``dinov2.npy``, ``sae_dinov2_k20.npy``, ``results/celeba/experiment/dinov2/k20/sae/``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class BackboneSpec:
    key:        str                    # short name used on the CLI and in file names
    timm_model: str                    # timm.create_model identifier
    img_size:   int                    # side length fed to the model
    mean:       Tuple[float, float, float]
    std:        Tuple[float, float, float]
    embed_dim:  int                    # token width (= mean-pooled embedding dim)
    n_patches:  int                    # patch tokens per image (excludes CLS/registers)
    label:      str                    # human-readable name for logs


BACKBONES = {
    # Matches the ECI paper exactly: timm vit_base_patch16_siglip_224 + forward_features().
    # No prefix tokens, 14×14 = 196 patches.
    "siglip": BackboneSpec(
        key="siglip", timm_model="vit_base_patch16_siglip_224", img_size=224,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        embed_dim=768, n_patches=196, label="SigLIP-B/16",
    ),
    # DINOv2-B/14 at 224px → 16×16 = 256 patches (+1 CLS token, dropped when pooling).
    # Native pretrained resolution is 518; timm interpolates the position embedding.
    "dinov2": BackboneSpec(
        key="dinov2", timm_model="vit_base_patch14_dinov2.lvd142m", img_size=224,
        mean=IMAGENET_MEAN, std=IMAGENET_STD,
        embed_dim=768, n_patches=256, label="DINOv2-B/14",
    ),
}

DEFAULT_BACKBONE = "siglip"


def get_backbone(key: str) -> BackboneSpec:
    if key not in BACKBONES:
        raise ValueError(f"Unknown backbone '{key}'. Choices: {sorted(BACKBONES)}")
    return BACKBONES[key]


# ── Artefact paths ────────────────────────────────────────────────────────────
# `_tag` is empty for SigLIP so the legacy file names are preserved untouched.

def _tag(backbone: str) -> str:
    return "" if backbone == DEFAULT_BACKBONE else f"_{backbone}"


def embed_path(data_dir: Path, backbone: str) -> Path:
    """Mean-pooled patch embeddings, (N, embed_dim) float32."""
    return data_dir / "embeddings" / f"{backbone}.npy"


def patches_path(data_dir: Path, backbone: str) -> Path:
    """Per-patch tokens, raw memmap (N, n_patches, embed_dim) float16."""
    return data_dir / "embeddings" / f"{backbone}_patches.npy"


def sae_ckpt_path(out_dir: Path, backbone: str, top_k: int) -> Path:
    return out_dir / f"sae_{backbone}_k{top_k}.pt"


def code_path(data_dir: Path, backbone: str, top_k: int) -> Path:
    """Sparse post-TopK codes z, (N, hidden_dim)."""
    return data_dir / "embeddings" / f"sae{_tag(backbone)}_k{top_k}.npy"


def precode_path(data_dir: Path, backbone: str, top_k: int) -> Path:
    """Continuous pre-activations z_pre, (N, hidden_dim)."""
    return data_dir / "embeddings" / f"sae_precode{_tag(backbone)}_k{top_k}.npy"


def experiment_dir(base_out: Path, backbone: str) -> Path:
    """Root for sweep results: unchanged for SigLIP, nested for other backbones."""
    return base_out if backbone == DEFAULT_BACKBONE else base_out / backbone
