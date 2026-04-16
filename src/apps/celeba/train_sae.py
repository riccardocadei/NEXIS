#!/usr/bin/env python3
"""
Stage 2 — Train a Sparse Autoencoder on CelebA SigLIP embeddings.

Reads:   data/celeba/embeddings/siglip.npy         (N, 1152) mean-pooled — always needed
         data/celeba/embeddings/siglip_patches.npy  (N, 729, 1152) float16 — per-patch training
                                                    (if present, used for SAE training)

Writes:  results/celeba/sae_siglip.pt        SAE weights + normalisation stats
         data/celeba/embeddings/sae.npy       (N, hidden_dim) SAE feature activations (z, post-topk)
         data/celeba/embeddings/sae_precode.npy (N, hidden_dim) SAE pre-activations (z_pre)

The SAE is trained on per-patch features if siglip_patches.npy exists (recommended —
this replicates the ECI paper which trains on all 729 SigLIP patch tokens per image,
giving 729× more training vectors and much stronger principal alignment).

Inference always runs on mean-pooled embeddings (siglip.npy), following ECI's
s4-sae_encoding.py which also encodes mean/CLS-pooled features through the SAE.

Usage
-----
    python src/apps/celeba/train_sae.py
    python src/apps/celeba/train_sae.py --hidden-dim 4608 --epochs 100
    python src/apps/celeba/train_sae.py --out-dir results/celeba
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from train.sae import TopKSAE, TopKSAETrainConfig, SAETrainResult, train_topk_sae, get_features, get_pre_features


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_sae(result: SAETrainResult, sae_path: Path) -> None:
    sae = result.sae
    torch.save({
        "state_dict":  sae.state_dict(),
        "sae_type":    "topk",
        "input_dim":   sae.dictionary.in_dimensions,   # overcomplete attr
        "hidden_dim":  sae.nb_concepts,                # overcomplete attr
        "top_k":       sae.top_k,
        "input_mean":  result.input_mean,
        "input_std":   result.input_std,
    }, sae_path)
    print(f"SAE saved  →  {sae_path}")


def load_sae(sae_path: Path) -> SAETrainResult:
    ckpt = torch.load(sae_path, map_location="cpu", weights_only=False)
    sae = TopKSAE(
        input_shape = ckpt["input_dim"],
        nb_concepts = ckpt["hidden_dim"],
        top_k       = ckpt.get("top_k", 5),
    )
    sae.load_state_dict(ckpt["state_dict"])
    return SAETrainResult(
        sae        = sae,
        input_mean = ckpt.get("input_mean"),
        input_std  = ckpt.get("input_std"),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir",       type=Path, default="data/celeba",
                   help="Directory with embeddings/siglip.npy; SAE features written here too")
    p.add_argument("--train-data-dir", type=Path, default=None,
                   help="Directory with siglip_patches.npy for SAE TRAINING (default: same as "
                        "--data-dir). Set to e.g. data/celeba_train to train on the larger "
                        "CelebA training split while encoding the validation split.")
    p.add_argument("--out-dir",        type=Path, default="results/celeba",
                   help="Directory for sae_siglip.pt model checkpoint (default: results/celeba)")
    p.add_argument("--hidden-dim",     type=int,  default=13824,
                   help="SAE hidden dim (default: 12× SigLIP dim = 13824)")
    p.add_argument("--top-k",          type=int,  default=5,
                   help="Top-k sparsity (default: 5)")
    p.add_argument("--epochs",         type=int,   default=20,
                   help="Training epochs (default: 20)")
    p.add_argument("--batch-size",     type=int,   default=4096,
                   help="SAE training batch size")
    p.add_argument("--lr",             type=float, default=5e-4,
                   help="Adam learning rate (default: 5e-4)")
    p.add_argument("--force",          action="store_true",
                   help="Retrain SAE even if checkpoint already exists")
    return p.parse_args()


def main():
    args = parse_args()
    ROOT = Path(__file__).resolve().parents[3]

    data_dir = (ROOT / args.data_dir
                if not args.data_dir.is_absolute() else args.data_dir)
    # train_data_dir: where to find patches for SAE training (may differ from data_dir)
    train_data_dir = data_dir if args.train_data_dir is None else (
        ROOT / args.train_data_dir
        if not args.train_data_dir.is_absolute() else args.train_data_dir
    )
    out_dir  = (ROOT / args.out_dir
                if not args.out_dir.is_absolute() else args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embed_path      = data_dir / "embeddings" / "siglip.npy"           # for encoding (eval split)
    patches_path    = train_data_dir / "embeddings" / "siglip_patches.npy"  # for SAE training
    sae_path        = out_dir  / f"sae_siglip_k{args.top_k}.pt"
    features_path   = data_dir / "embeddings" / f"sae_k{args.top_k}.npy"
    precode_path    = data_dir / "embeddings" / f"sae_precode_k{args.top_k}.npy"

    if train_data_dir != data_dir:
        print(f"SAE training data : {train_data_dir}")
        print(f"SAE encoding data : {data_dir}")

    if sae_path.exists() and features_path.exists() and precode_path.exists() and not args.force:
        print("SAE and features already exist. Use --force to retrain.")
        feat = np.load(features_path, mmap_mode="r")
        print(f"Features shape: {feat.shape}   Pre-code: {np.load(precode_path, mmap_mode='r').shape}")
        return

    print(f"Loading mean-pooled embeddings from {embed_path} …")
    embeddings = np.load(embed_path)
    N, d = embeddings.shape
    print(f"Mean-pooled embeddings: {embeddings.shape}")

    # ── Train SAE ────────────────────────────────────────────────────────────
    if not sae_path.exists() or args.force:
        # Prefer per-patch training if siglip_patches.npy is available.
        use_patches = patches_path.exists()
        if use_patches:
            print(f"\nFound per-patch features at {patches_path}")
            print("Opening with memmap (file may be ~33 GB, only touched pages are read)…")
            # Written by np.memmap (raw binary, no .npy header) — must use np.memmap to read.
            # Shape: (N_images, N_patches, embed_dim). We peek at siglip.npy to get N and d.
            N_imgs = embeddings.shape[0]
            d      = embeddings.shape[1]
            T      = int(patches_path.stat().st_size // (N_imgs * d * 2))  # 2 bytes per float16
            train_data = np.memmap(patches_path, dtype=np.float16, mode="r",
                                   shape=(N_imgs, T, d))
            _, T, _ = train_data.shape
            print(f"Patch array shape: {train_data.shape}  "
                  f"→ {N * T:,} total vectors/epoch")
        else:
            print(f"\nNo per-patch features found — using mean-pooled embeddings.")
            print("For better principal alignment, re-run embed with --save-patches first.")
            train_data = embeddings

        per_patch = use_patches
        print(f"\nTraining TopKSAE: input={d}  hidden={args.hidden_dim}  "
              f"top_k={args.top_k}  lr={args.lr}  epochs={args.epochs}  "
              f"per_patch={per_patch}")
        cfg = TopKSAETrainConfig(
            hidden_dim       = args.hidden_dim,
            top_k            = args.top_k,
            lr               = args.lr,
            batch_size       = args.batch_size,
            num_epochs       = args.epochs,
            normalize_inputs = not per_patch,   # ECI uses no normalization for patches
            per_patch        = per_patch,
            log_every        = 1,
        )
        result = train_topk_sae(train_data, cfg)
        save_sae(result, sae_path)
    else:
        print(f"Loading existing SAE from {sae_path} …")
        result = load_sae(sae_path)

    # ── Encode all images ─────────────────────────────────────────────────────
    print(f"\nEncoding {N} images → SAE features (z + z_pre) …")
    features = get_features(embeddings, result, batch_size=4096)
    np.save(features_path, features)
    l0 = float((features > 0).mean(axis=1).mean() * features.shape[1])
    print(f"z      saved: shape={features.shape}  avg-L0={l0:.1f}  →  {features_path}")

    pre_features = get_pre_features(embeddings, result, batch_size=4096)
    np.save(precode_path, pre_features)
    print(f"z_pre  saved: shape={pre_features.shape}  →  {precode_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
