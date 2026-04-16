#!/usr/bin/env python3
"""
Stage 1 — Download CelebA and extract SigLIP-base embeddings.

Uses streaming mode so only the requested split is read (no full Arrow cache
built for all splits — avoids the ~10 GB train-split cost when only valid is needed).

Writes to --data-dir (default: data/celeba/):
  labels.parquet          CelebA attribute labels, binarised to {0, 1}
  embeddings/siglip.npy   Mean-pooled SigLIP patch embeddings  (N, 768)

Usage
-----
    python src/apps/celeba/embed.py --split valid
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

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

# ── SigLIP model (timm) — matches ECI paper exactly ──────────────────────────
# Reference uses timm.create_model('vit_base_patch16_siglip_224', ...) + forward_features().
# HuggingFace SiglipVisionModel produces different embeddings from the same images.
SIGLIP_TIMM_MODEL = "vit_base_patch16_siglip_224"
SIGLIP_EMBED_DIM = 768

_SIGLIP_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Known split sizes for tqdm totals (approximate)
_SPLIT_SIZES = {"train": 162_770, "valid": 19_867, "test": 19_962}


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

def collect_labels(hf_iterable, split: str) -> pd.DataFrame:
    """Stream through the dataset once, collecting all non-image fields."""
    total = _SPLIT_SIZES.get(split)
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
def extract_siglip_embeddings(
    hf_iterable,
    split: str,
    batch_size: int = 64,
    device: str = "cuda",
    save_patches: bool = False,
    patches_out_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Stream images through SigLIP (timm, vit_base_patch16_siglip_224).

    Uses timm's forward_features() to match the ECI paper exactly.

    Returns:
        mean_pooled: (N, 768) float32 — mean-pooled patch embeddings
        patches:     memmap array (N, 196, 768) float16 at patches_out_path,
                     or None if save_patches is False.

    Patches are written directly to a memory-mapped file so RAM usage stays
    bounded (~1 batch at a time on GPU + small CPU buffer), regardless of N.
    """
    import timm

    print(f"Loading SigLIP model: {SIGLIP_TIMM_MODEL} (timm)")
    model = timm.create_model(SIGLIP_TIMM_MODEL, pretrained=True, num_classes=0)
    model.eval().to(device)

    # num_workers=0: IterableDataset + streaming can't be forked
    dataset = StreamingImageDataset(hf_iterable, transform=_SIGLIP_TRANSFORM)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    total = _SPLIT_SIZES.get(split, 0)
    total_batches = (total + batch_size - 1) // batch_size or None

    all_embeds: list[np.ndarray] = []
    patch_mmap: np.memmap | None = None
    write_offset = 0

    for batch in tqdm(loader, desc="SigLIP embeddings", total=total_batches, unit="batch"):
        batch = batch.to(device, non_blocking=True)
        # timm: forward_features → (B, T, d) patch tokens (no CLS for SigLIP)
        patch_tokens = model.forward_features(batch)   # (B, 196, 768)
        emb = patch_tokens.mean(dim=1)                 # (B, 768) mean-pooled
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

def save_images(hf_iterable, split: str, out_path: Path) -> None:
    """Stream images, resize to 128×128, save as (N, 128, 128, 3) uint8 array."""
    total = _SPLIT_SIZES.get(split)
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
    p.add_argument("--split",       default="valid",
                   help="HF dataset split (default: valid, ~19 867 images)")
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--device",      default=None)
    p.add_argument("--force",        action="store_true")
    p.add_argument("--save-images",  action="store_true",
                   help="Also save 128×128 thumbnails to images.npy (~1 GB)")
    p.add_argument("--save-patches", action="store_true",
                   help="Also save per-patch SigLIP features to siglip_patches.npy "
                        "(N, 729, 1152) float16, ~33 GB — needed for SAE patch-training")
    p.add_argument("--list-attrs",   action="store_true",
                   help="Print attribute names and prevalences then exit")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = (ROOT / args.data_dir
                if not args.data_dir.is_absolute() else args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    labels_path = data_dir / "labels.parquet"
    embed_path  = data_dir / "embeddings" / "siglip.npy"

    labels_done   = labels_path.exists() and not args.force
    embed_done    = embed_path.exists()  and not args.force
    patches_path  = data_dir / "embeddings" / "siglip_patches.npy"
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
        print(f"Streaming flwrlabs/celeba [{args.split}] for labels…")
        hf_stream = load_dataset("flwrlabs/celeba", split=args.split, streaming=True)
        df = collect_labels(hf_stream, args.split)
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
        print(f"\nStreaming flwrlabs/celeba [{args.split}] for embeddings "
              f"{'+ per-patch features ' if do_patches else ''}…")
        hf_stream = load_dataset("flwrlabs/celeba", split=args.split, streaming=True)
        embeddings, patches = extract_siglip_embeddings(
            hf_stream, args.split,
            batch_size=args.batch_size,
            device=device,
            save_patches=do_patches,
            patches_out_path=patches_path if do_patches else None,
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
            print(f"\nStreaming flwrlabs/celeba [{args.split}] for image thumbnails…")
            hf_stream = load_dataset("flwrlabs/celeba", split=args.split, streaming=True)
            save_images(hf_stream, args.split, images_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
