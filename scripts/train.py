"""
Extract DINOv2 patch embeddings, train an SAE, and produce per-site feature vectors.

Pipeline
--------
1. Extract DINOv2 patch tokens for all 1,318 Uganda sites
   → cached to results/uganda/patch_embeddings.npz (reused on re-runs)
2. Train SAE on all patches from step 1  (N_sites × 256 tokens)
3. Encode experimental-site patches → mean-pool over patches → (332, n_features)
4. Join to individuals: each of the 3,142 obs gets its site's feature vector
5. Save everything to results/uganda/

Usage
-----
    python train.py [--model dinov2_vitb14] [--hidden-dim 3072]
                    [--l1-coeff 1e-3] [--epochs 100] [--force-reextract]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "uganda"
IMG_DIR  = DATA_DIR / "Uganda2000_processed"
OUT_DIR  = ROOT / "results" / "uganda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from embeddings import UgandaSatelliteDataset, embed_uganda_sites
from sae import SAE, SAETrainConfig, train_sae, get_features


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          default="dinov2_vitb14",
                   choices=["dinov2_vits14","dinov2_vitb14","dinov2_vitl14"])
    p.add_argument("--hidden-dim",     type=int,   default=3072,
                   help="SAE hidden dimension (default 4× DINOv2-B = 3072)")
    p.add_argument("--l1-coeff",       type=float, default=1e-3)
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--batch-size",     type=int,   default=2048,
                   help="SAE training batch size (patch tokens, not images)")
    p.add_argument("--force-reextract", action="store_true",
                   help="Re-run DINOv2 extraction even if cache exists")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────
def load_keys():
    """Return (all_keys, experimental_keys)."""
    all_keys = pd.read_csv(DATA_DIR / "UgandaGeoKeyMat.csv")
    all_keys = all_keys.iloc[:, 0].dropna().astype(int).tolist()

    exp_df   = pd.read_csv(DATA_DIR / "UgandaDataProcessed.csv", low_memory=False,
                           usecols=["geo_long_lat_key"])
    exp_keys = exp_df["geo_long_lat_key"].dropna().astype(int).unique().tolist()

    return all_keys, exp_keys


def extract_or_load_patches(keys: list[int], cache_path: Path,
                             model_name: str, force: bool) -> tuple[np.ndarray, list[int]]:
    """Return patch embeddings (N_sites, 256, d) and the valid key list."""
    if cache_path.exists() and not force:
        print(f"Loading cached patch embeddings from {cache_path}")
        data = np.load(cache_path)
        return data["embeddings"], data["keys"].tolist()

    print(f"Extracting patch embeddings for {len(keys)} sites with {model_name}...")
    embeddings, valid_keys = embed_uganda_sites(
        IMG_DIR, keys,
        model_name=model_name,
        batch_size=4,       # each image loads 3 large CSVs; keep low
        num_workers=0,
        mode="patch",
    )
    np.savez_compressed(cache_path, embeddings=embeddings, keys=np.array(valid_keys))
    print(f"Saved patch embeddings to {cache_path}  shape={embeddings.shape}")
    return embeddings, valid_keys


def mean_pool_patches(patch_emb: np.ndarray) -> np.ndarray:
    """(N, 256, d) → (N, d)  simple spatial mean-pool."""
    return patch_emb.mean(axis=1)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── 1. Keys ───────────────────────────────────────────────────────────────
    all_keys, exp_keys = load_keys()
    print(f"All Uganda sites: {len(all_keys)}  |  Experimental: {len(exp_keys)}")

    # ── 2. Extract DINOv2 patch embeddings (cached) ───────────────────────────
    cache_path = OUT_DIR / f"patch_embeddings_{args.model}.npz"
    all_patch_emb, all_valid_keys = extract_or_load_patches(
        all_keys, cache_path, args.model, args.force_reextract
    )
    # all_patch_emb: (N_all, 256, d)
    print(f"Patch embeddings shape: {all_patch_emb.shape}")

    n_sites, n_patches, d = all_patch_emb.shape
    patches_flat = all_patch_emb.reshape(n_sites * n_patches, d)   # (N_all*256, d)

    # ── 3. Train SAE on all patches ───────────────────────────────────────────
    print(f"\nTraining SAE: input_dim={d}  hidden_dim={args.hidden_dim}  "
          f"l1={args.l1_coeff}  epochs={args.epochs}")
    cfg = SAETrainConfig(
        hidden_dim=args.hidden_dim,
        l1_coeff=args.l1_coeff,
        lr=1e-4,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        normalize_inputs=True,
        log_every=10,
    )
    result = train_sae(patches_flat, cfg)

    sae_path = OUT_DIR / "sae.pt"
    torch.save({
        "state_dict":   result.sae.state_dict(),
        "input_dim":    d,
        "hidden_dim":   args.hidden_dim,
        "l1_coeff":     args.l1_coeff,
        "input_mean":   result.input_mean,
        "input_std":    result.input_std,
    }, sae_path)
    print(f"SAE saved to {sae_path}")

    # ── 4. Encode experimental sites → mean-pool over patches ─────────────────
    # Filter all_patch_emb to experimental keys only
    key_to_idx = {k: i for i, k in enumerate(all_valid_keys)}
    exp_valid  = [k for k in exp_keys if k in key_to_idx]
    exp_idxs   = [key_to_idx[k] for k in exp_valid]
    exp_patch_emb = all_patch_emb[exp_idxs]              # (N_exp, 256, d)

    n_exp = len(exp_valid)
    exp_flat = exp_patch_emb.reshape(n_exp * n_patches, d)

    print(f"\nEncoding {n_exp} experimental sites ({n_exp * n_patches} patches) → SAE features...")
    sae_patch_features = get_features(exp_flat, result)   # (N_exp*256, hidden_dim)
    sae_patch_features = sae_patch_features.reshape(n_exp, n_patches, args.hidden_dim)

    # Mean-pool across patches: (N_exp, hidden_dim)
    site_features = sae_patch_features.mean(axis=1)
    print(f"Site features shape: {site_features.shape}")
    print(f"Avg active features per patch: {(sae_patch_features > 0).mean(axis=(0,1)).sum():.1f} / {args.hidden_dim}")

    # ── 5. Join to individuals ─────────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "UgandaDataProcessed.csv", low_memory=False,
                     usecols=["geo_long_lat_key"])
    key_to_feat = {k: site_features[i] for i, k in enumerate(exp_valid)}

    feat_matrix = np.full((len(df), args.hidden_dim), np.nan, dtype=np.float32)
    for row_idx, key in enumerate(df["geo_long_lat_key"].values):
        if key in key_to_feat:
            feat_matrix[row_idx] = key_to_feat[key]

    covered = np.isfinite(feat_matrix[:, 0]).sum()
    print(f"Individuals with image features: {covered}/{len(df)} ({covered/len(df)*100:.1f}%)")

    # ── 6. Save ────────────────────────────────────────────────────────────────
    np.savez_compressed(
        OUT_DIR / "site_features.npz",
        site_features = site_features,
        site_keys     = np.array(exp_valid),
    )
    np.savez_compressed(
        OUT_DIR / "individual_features.npz",
        features = feat_matrix,
    )
    print(f"\nSaved:")
    print(f"  {OUT_DIR / 'site_features.npz'}        ({n_exp}, {args.hidden_dim})")
    print(f"  {OUT_DIR / 'individual_features.npz'}  ({len(df)}, {args.hidden_dim})")
    print(f"  {OUT_DIR / 'sae.pt'}")


if __name__ == "__main__":
    main()
