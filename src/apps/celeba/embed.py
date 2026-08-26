#!/usr/bin/env python3
"""
Stage 1 — Download CelebA and extract frozen-backbone embeddings.

Uses streaming mode so only the requested split is read (no full Arrow cache
built for all splits — avoids the ~10 GB train-split cost when only valid is needed).

Writes to --data-dir (default: data/celeba/):
  labels.parquet             CelebA attribute labels, binarised to {0, 1}
  embeddings/{backbone}.npy  Mean-pooled patch embeddings  (N, embed_dim)

Backbones (--backbone): siglip (default, SigLIP-B/16), dinov2 (DINOv2-B/14).
Both are 768-d, so downstream SAE settings are directly comparable.

Usage
-----
    python src/apps/celeba/embed.py --split valid
    python src/apps/celeba/embed.py --split valid --backbone dinov2 --save-patches
    python src/apps/celeba/embed.py --list-attrs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent.parent

from apps.celeba.backbones import (
    BACKBONES, DEFAULT_BACKBONE, BackboneSpec, get_backbone,
    embed_path as _embed_path, patches_path as _patches_path,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

# Backbones are defined in backbones.py.  SigLIP matches the ECI paper exactly:
# timm.create_model('vit_base_patch16_siglip_224', ...) + forward_features().
# HuggingFace SiglipVisionModel produces different embeddings from the same images.


def build_transform(spec: BackboneSpec) -> transforms.Compose:
    """Resize → tensor → backbone-specific normalisation."""
    return transforms.Compose([
        transforms.Resize((spec.img_size, spec.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(spec.mean), std=list(spec.std)),
    ])


_SIGLIP_TRANSFORM = build_transform(BACKBONES[DEFAULT_BACKBONE])

# Known split sizes for tqdm totals (approximate)
_SPLIT_SIZES = {"train": 162_770, "valid": 19_867, "test": 19_962}


# ── Split loading (streaming, or a fixed-size random subsample) ───────────────

def load_split(split: str, sample_n: int | None = None, sample_seed: int = 0):
    """Return ``(iterable, n_total)`` for one CelebA split.

    Default: streaming mode (no Arrow cache build).  With ``sample_n`` set, the
    split is loaded non-streaming (memory-mapped from the local Arrow cache) and
    a fixed random subset of ``sample_n`` rows is selected with ``sample_seed``.
    Rows are kept in ascending original order so labels, embeddings and patches
    stay mutually aligned; only the selected rows are ever decoded.

    This is what makes a same-size / different-sample SAE training corpus
    possible: e.g. ``load_split("train", 19_867, seed)`` draws exactly as many
    images as the valid split, from an identity-disjoint pool.
    """
    from datasets import load_dataset

    if sample_n is None:
        return load_dataset("flwrlabs/celeba", split=split, streaming=True), \
               _SPLIT_SIZES.get(split)

    ds = load_dataset("flwrlabs/celeba", split=split)
    if sample_n > len(ds):
        raise ValueError(f"--sample-n {sample_n} exceeds split size {len(ds)}")
    rng = np.random.default_rng(sample_seed)
    idx = np.sort(rng.choice(len(ds), size=sample_n, replace=False))
    print(f"Subsampled {sample_n:,} / {len(ds):,} images from [{split}] "
          f"(seed={sample_seed}); first/last row = {idx[0]}/{idx[-1]}")
    return ds.select(idx), sample_n


# ── Streaming dataset wrapper ─────────────────────────────────────────────────

class StreamingImageDataset(IterableDataset):
    """Wraps a HuggingFace streaming dataset for use with DataLoader.

    num_workers must be 0 — streaming iterators cannot be forked safely.
    """

    def __init__(self, hf_iterable, transform=_SIGLIP_TRANSFORM):
        self.hf_iterable = hf_iterable
        self.transform = transform

    def __iter__(self):
        for item in self.hf_iterable:
            img = item["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.asarray(img))
            yield self.transform(img.convert("RGB"))


# ── Label extraction ──────────────────────────────────────────────────────────

def collect_labels(hf_iterable, split: str, n_total: int | None = None) -> pd.DataFrame:
    """Stream through the dataset once, collecting all non-image fields."""
    total = n_total or _SPLIT_SIZES.get(split)
    records = []
    for item in tqdm(hf_iterable, desc="Collecting labels", total=total, unit="img"):
        records.append({k: v for k, v in item.items() if k != "image"})

    df = pd.DataFrame(records)
    # Binarise: handle -1/1 → 0/1
    for col in df.columns:
        if df[col].dtype.kind in ("i", "f") and df[col].min() < 0:
            df[col] = ((df[col] + 1) // 2).astype(np.int8)
        else:
            df[col] = df[col].astype(np.int8)
    return df


# ── Embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    hf_iterable,
    split: str,
    spec: BackboneSpec = BACKBONES[DEFAULT_BACKBONE],
    batch_size: int = 64,
    device: str = "cuda",
    save_patches: bool = False,
    patches_out_path: Path | None = None,
    n_total: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Stream images through a frozen timm backbone and mean-pool its patch tokens.

    Uses timm's forward_features(); for SigLIP this matches the ECI paper exactly.
    Prefix tokens (CLS / registers, e.g. DINOv2's single CLS) are dropped so that
    every backbone is pooled over patch tokens only.

    Returns:
        mean_pooled: (N, embed_dim) float32 — mean-pooled patch embeddings
        patches:     memmap array (N, n_patches, embed_dim) float16 at
                     patches_out_path, or None if save_patches is False.

    Patches are written directly to a memory-mapped file so RAM usage stays
    bounded (~1 batch at a time on GPU + small CPU buffer), regardless of N.
    """
    import timm

    print(f"Loading {spec.label} model: {spec.timm_model} (timm, img_size={spec.img_size})")
    model = timm.create_model(spec.timm_model, pretrained=True, num_classes=0,
                              img_size=spec.img_size)
    model.eval().to(device)
    n_prefix = getattr(model, "num_prefix_tokens", 0)

    # num_workers=0: IterableDataset + streaming can't be forked
    dataset = StreamingImageDataset(hf_iterable, transform=build_transform(spec))
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    total = n_total or _SPLIT_SIZES.get(split, 0)
    total_batches = (total + batch_size - 1) // batch_size or None

    all_embeds: list[np.ndarray] = []
    patch_mmap: np.memmap | None = None
    write_offset = 0

    for batch in tqdm(loader, desc=f"{spec.label} embeddings",
                      total=total_batches, unit="batch"):
        batch = batch.to(device, non_blocking=True)
        # timm: forward_features → (B, n_prefix + T, d); keep patch tokens only
        patch_tokens = model.forward_features(batch)[:, n_prefix:]   # (B, T, d)
        emb = patch_tokens.mean(dim=1)                               # (B, d)
        all_embeds.append(emb.cpu().float().numpy())

        if save_patches:
            B, T, d = patch_tokens.shape
            # Lazily create the memmap once we know T and d
            if patch_mmap is None:
                assert patches_out_path is not None
                patches_out_path.parent.mkdir(parents=True, exist_ok=True)
                N_total = total if total > 0 else len(all_embeds) * B  # rough
                size_gb = N_total * T * d * 2 / 1e9
                print(f"\nPre-allocating patch memmap: ({N_total}, {T}, {d}) "
                      f"float16 ≈ {size_gb:.1f} GB  →  {patches_out_path}")
                patch_mmap = np.memmap(
                    patches_out_path, dtype=np.float16, mode="w+",
                    shape=(N_total, T, d),
                )
            chunk = patch_tokens.cpu().to(torch.float16).numpy()
            patch_mmap[write_offset : write_offset + B] = chunk
            write_offset += B

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    mean_pooled = np.concatenate(all_embeds, axis=0)

    if save_patches and patch_mmap is not None:
        actual_N = write_offset
        # Flush and close; reload as read-only with the true shape
        patch_mmap.flush()
        del patch_mmap
        # Re-open as read-only so callers get a proper array reference
        _, T_final, d_final = patch_tokens.shape   # (B, 196, 768) from last batch
        patches = np.memmap(patches_out_path, dtype=np.float16, mode="r",
                            shape=(actual_N, T_final, d_final))
        print(f"Patch memmap written: shape=({actual_N}, {T_final}, {d_final})")
    else:
        patches = None

    return mean_pooled, patches


# ── Image saving ─────────────────────────────────────────────────────────────

_THUMB_TRANSFORM = transforms.Compose([
    transforms.Resize(128, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(128),
])

def save_images(hf_iterable, split: str, out_path: Path, n_total: int | None = None) -> None:
    """Stream images, resize to 128×128, save as (N, 128, 128, 3) uint8 array."""
    total = n_total or _SPLIT_SIZES.get(split)
    imgs = []
    for item in tqdm(hf_iterable, desc="Saving images", total=total, unit="img"):
        img = item["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        img = _THUMB_TRANSFORM(img.convert("RGB"))
        imgs.append(np.asarray(img, dtype=np.uint8))
    arr = np.stack(imgs, axis=0)          # (N, 128, 128, 3)
    np.save(out_path, arr)
    print(f"Saved images: shape={arr.shape}  →  {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir",    type=Path, default="data/celeba")
    p.add_argument("--backbone",    default=DEFAULT_BACKBONE, choices=sorted(BACKBONES),
                   help=f"Frozen vision backbone (default: {DEFAULT_BACKBONE}). "
                        "Embeddings are written to embeddings/{backbone}.npy.")
    p.add_argument("--split",       default="valid",
                   help="HF dataset split (default: valid, ~19 867 images)")
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--sample-n",    type=int, default=None,
                   help="Embed a fixed random subset of --sample-n images from the split "
                        "instead of all of it (non-streaming, from the local Arrow cache). "
                        "Used to build a same-size / different-sample SAE training corpus, "
                        "e.g. --split train --sample-n 19867 --sample-seed 1.")
    p.add_argument("--sample-seed", type=int, default=0,
                   help="RNG seed for --sample-n (default: 0)")
    p.add_argument("--device",      default=None)
    p.add_argument("--force",        action="store_true")
    p.add_argument("--save-images",  action="store_true",
                   help="Also save 128×128 thumbnails to images.npy (~1 GB)")
    p.add_argument("--save-patches", action="store_true",
                   help="Also save per-patch features to embeddings/{backbone}_patches.npy "
                        "(N, n_patches, embed_dim) float16 — needed for SAE patch-training "
                        "(~6 GB for SigLIP, ~7.8 GB for DINOv2 on the valid split)")
    p.add_argument("--list-attrs",   action="store_true",
                   help="Print attribute names and prevalences then exit")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = (ROOT / args.data_dir
                if not args.data_dir.is_absolute() else args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    spec = get_backbone(args.backbone)
    print(f"Device: {device}   Backbone: {spec.label} ({spec.key})")

    labels_path = data_dir / "labels.parquet"
    embed_path  = _embed_path(data_dir, spec.key)

    labels_done   = labels_path.exists() and not args.force
    embed_done    = embed_path.exists()  and not args.force
    patches_path  = _patches_path(data_dir, spec.key)
    patches_done  = patches_path.exists() and not args.force
    images_path   = data_dir / "images.npy"
    images_done   = images_path.exists() and not args.force

    need_stream = (not labels_done) or (not embed_done) or \
                  (args.save_patches and not patches_done) or \
                  (args.save_images and not images_done)

    if labels_done and embed_done and not args.list_attrs:
        emb = np.load(embed_path, mmap_mode="r")
        print(f"Mean-pooled embeddings already exist. Shape: {emb.shape}")
        if args.save_patches and patches_done:
            pat = np.load(patches_path, mmap_mode="r")
            print(f"Per-patch embeddings already exist. Shape: {pat.shape}")
        if not need_stream:
            print("Use --force to recompute.")
            return

    from datasets import load_dataset

    if args.list_attrs:
        print(f"Loading one example to list attributes…")
        sample = next(iter(load_dataset("flwrlabs/celeba", split=args.split,
                                        streaming=True)))
        for k, v in sorted(sample.items()):
            if k != "image":
                print(f"  {k}")
        return

    # ── Step 1: labels (stream once) ─────────────────────────────────────────
    if not labels_done:
        print(f"Reading flwrlabs/celeba [{args.split}] for labels…")
        hf_stream, n_total = load_split(args.split, args.sample_n, args.sample_seed)
        df = collect_labels(hf_stream, args.split, n_total=n_total)
        df.to_parquet(labels_path, index=False)
        n = len(df)
        print(f"Saved labels: {df.shape}  →  {labels_path}")
        print("\nAttribute prevalences:")
        for attr, prev in df.mean().sort_values(ascending=False).items():
            print(f"  {attr:30s}  {prev:.3f}")
    else:
        n = len(pd.read_parquet(labels_path))
        print(f"Labels already exist ({n} rows, skip)")

    # ── Step 2: embeddings (stream once more) ────────────────────────────────
    need_embed_stream = (not embed_done) or (args.save_patches and not patches_done)
    if need_embed_stream:
        do_patches = args.save_patches and not patches_done
        print(f"\nReading flwrlabs/celeba [{args.split}] for embeddings "
              f"{'+ per-patch features ' if do_patches else ''}…")
        hf_stream, n_total = load_split(args.split, args.sample_n, args.sample_seed)
        embeddings, patches = extract_embeddings(
            hf_stream, args.split,
            spec=spec,
            batch_size=args.batch_size,
            device=device,
            save_patches=do_patches,
            patches_out_path=patches_path if do_patches else None,
            n_total=n_total,
        )
        if not embed_done:
            np.save(embed_path, embeddings)
            print(f"Saved mean-pooled embeddings: shape={embeddings.shape}  →  {embed_path}")
        # patches were written to disk incrementally via memmap — no extra save needed
    else:
        print(f"Embeddings already exist (skip)")

    # ── Step 3: images (optional, CPU-only) ──────────────────────────────────
    if args.save_images:
        images_path = data_dir / "images.npy"
        if images_path.exists() and not args.force:
            print(f"Images already exist (skip): {images_path}")
        else:
            print(f"\nReading flwrlabs/celeba [{args.split}] for image thumbnails…")
            hf_stream, n_total = load_split(args.split, args.sample_n, args.sample_seed)
            save_images(hf_stream, args.split, images_path, n_total=n_total)

    print("\nDone.")


if __name__ == "__main__":
    main()
