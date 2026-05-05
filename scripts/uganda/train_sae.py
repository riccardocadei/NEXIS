"""Train a Sparse Autoencoder (SAE) on Prithvi embeddings for Uganda.

Architecture: TopK SAE (Gao et al. 2024).  The encoder maps each 768-dim
Prithvi embedding into a large sparse latent space; only the top-k activations
are kept per sample.

Data roles:
  TRAINING corpus  → national grid embeddings (full Uganda, pre-treatment 2005–2007)
  RCT data         → 331 experimental site embeddings, held out during training

Recommended workflow:
  1. python scripts/uganda/download_tiles.py --mode rct
  2. python scripts/uganda/download_tiles.py --mode national
  3. python scripts/uganda/extract_satellite_features.py \\
       --tif-dir data/uganda/satellite/tif_rct \\
       --out-dir data/uganda/satellite/rct
  4. python scripts/uganda/extract_satellite_features.py \\
       --tif-dir data/uganda/satellite/tif_national \\
       --out-dir data/uganda/satellite/national
  5. python scripts/uganda/train_sae.py   ← this script

Outputs (--out-dir, default results/uganda/prithvi_l5_{d_hidden}/):
  sae_model.pt                — TopK SAE weights
  sae_whiten_mean/std.npy     — whitening stats (fit on national corpus)
  sae_reconstruction_error.csv— per-site reconstruction MSE for RCT sites
  sae_cv_results.csv          — k-fold CV MSE on national corpus

Uganda-pipeline-compatible outputs (same dir):
  site_features.npz           — (N_rct, d_hidden) sparse activations + site_keys
  individual_features.npz     — (N_individuals=3142, d_hidden) one row per person
  patch_embeddings.npz        — (N_rct, 768) raw Prithvi embeddings per RCT site
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset


# ── TopK SAE ──────────────────────────────────────────────────────────────────

class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder with unit-norm decoder columns (Gao et al. 2024)."""

    def __init__(self, d_in: int, d_hidden: int, k: int):
        super().__init__()
        self.d_in     = d_in
        self.d_hidden = d_hidden
        self.k        = k

        self.W_enc = nn.Linear(d_in, d_hidden, bias=True)
        self.W_dec = nn.Parameter(torch.empty(d_hidden, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre  = self.W_enc(x - self.b_dec)
        acts = F.relu(pre)
        if self.k < self.d_hidden:
            topk_vals, _ = torch.topk(acts, self.k, dim=-1)
            threshold     = topk_vals[:, -1:].detach()
            acts          = acts * (acts >= threshold)
        return acts

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ F.normalize(self.W_dec, dim=1) + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)


# ── Training ──────────────────────────────────────────────────────────────────

def train_sae(
    d_in: int, d_hidden: int, k: int,
    embeddings: torch.Tensor,   # (N, d_in), already whitened
    epochs: int, lr: float, noise_std: float, batch_size: int,
    device: torch.device, seed: int = 42, verbose: bool = True,
) -> TopKSAE:
    torch.manual_seed(seed)
    sae       = TopKSAE(d_in, d_hidden, k).to(device)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=lr, weight_decay=0.0)
    loader    = DataLoader(TensorDataset(embeddings.to(device)),
                           batch_size=batch_size, shuffle=True, drop_last=False)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (x,) in loader:
            x_noisy  = x + torch.randn_like(x) * noise_std if noise_std > 0 else x
            x_hat, _ = sae(x_noisy)
            loss      = F.mse_loss(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()
            epoch_loss += loss.item()
        epoch_loss /= len(loader)
        if verbose and epoch % max(1, epochs // 10) == 0:
            print(f"    epoch {epoch:5d}/{epochs}  train_loss={epoch_loss:.6f}")

    return sae


def reconstruction_mse(sae: TopKSAE, embeddings: torch.Tensor,
                        device: torch.device) -> np.ndarray:
    sae.eval()
    with torch.no_grad():
        x     = embeddings.to(device)
        x_hat = sae(x)[0]
        return ((x - x_hat) ** 2).mean(dim=1).cpu().numpy()


# ── Cross-validation ──────────────────────────────────────────────────────────

def run_cv(
    d_in: int, d_hidden: int, k: int,
    embeddings: torch.Tensor, epochs: int, lr: float, noise_std: float,
    batch_size: int, device: torch.device, n_folds: int, seed: int,
) -> pd.DataFrame:
    kf   = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    print(f"\n  CV ({n_folds} folds, training from scratch each fold) …")
    for fold, (tr_idx, val_idx) in enumerate(kf.split(np.arange(len(embeddings))), 1):
        print(f"  Fold {fold}/{n_folds}:  train={len(tr_idx)}  val={len(val_idx)}")
        sae = train_sae(d_in, d_hidden, k, embeddings[tr_idx], epochs, lr,
                        noise_std, batch_size, device, seed=seed + fold, verbose=False)
        mse = reconstruction_mse(sae, embeddings[val_idx], device)
        rows.append({'fold': fold, 'n_val': len(val_idx),
                     'val_mse_mean': mse.mean(), 'val_mse_std': mse.std()})
        print(f"         val MSE = {mse.mean():.4f} ± {mse.std():.4f}")

    df_cv = pd.DataFrame(rows)
    print(f"\n  CV summary: mean={df_cv['val_mse_mean'].mean():.4f}  "
          f"std={df_cv['val_mse_mean'].std():.4f}")
    return df_cv


# ── Uganda-format output helpers ──────────────────────────────────────────────

def make_individual_features(
    site_feats: np.ndarray,
    site_keys: np.ndarray,
    data_csv: Path,
    d_hidden: int,
) -> np.ndarray:
    """Map site-level SAE features to individuals via geo_long_lat_key."""
    df = pd.read_csv(data_csv, low_memory=False, usecols=['geo_long_lat_key'])
    key_to_feat = {int(k): site_feats[i] for i, k in enumerate(site_keys)}

    feat_matrix = np.full((len(df), d_hidden), np.nan, dtype=np.float32)
    for row_idx, key in enumerate(df['geo_long_lat_key'].values):
        if not pd.isna(key) and int(key) in key_to_feat:
            feat_matrix[row_idx] = key_to_feat[int(key)]

    covered = np.isfinite(feat_matrix[:, 0]).sum()
    print(f"  Individuals with image features: {covered}/{len(df)} "
          f"({covered/len(df)*100:.1f}%)")
    return feat_matrix


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # Embedding paths
    p.add_argument('--train-embeddings',
                   default='../../data/uganda/satellite/national/prithvi_embeddings.npy',
                   help='(N_train, 768) .npy — national grid embeddings (training corpus)')
    p.add_argument('--rct-embeddings',
                   default='../../data/uganda/satellite/rct/prithvi_embeddings.npy',
                   help='(N_rct, 768) .npy — RCT site embeddings (held out during training)')
    p.add_argument('--rct-keys',
                   default='../../data/uganda/satellite/rct/prithvi_site_keys.npy',
                   help='(N_rct,) .npy — geo_long_lat_key values for RCT sites')
    p.add_argument('--data-csv',
                   default='../../data/uganda/UgandaDataProcessed.csv',
                   help='UgandaDataProcessed.csv (for individual-level feature join)')
    p.add_argument('--out-dir', default=None,
                   help='Output directory (default: results/uganda/prithvi_l5_{d_hidden})')

    # SAE architecture
    p.add_argument('--d-hidden',   type=int,   default=1024,
                   help='SAE latent dimension (default 1024)')
    p.add_argument('--k',          type=int,   default=25,
                   help='Top-k active latents per sample (default 25)')

    # Training
    p.add_argument('--epochs',     type=int,   default=2000)
    p.add_argument('--lr',         type=float, default=2e-4)
    p.add_argument('--noise-std',  type=float, default=0.05)
    p.add_argument('--batch-size', type=int,   default=64)
    p.add_argument('--cv-folds',   type=int,   default=5,
                   help='CV folds on national corpus (0 = skip)')
    p.add_argument('--device',
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',       type=int,   default=42)
    return p.parse_args()


def main():
    args       = parse_args()
    script_dir = Path(__file__).parent
    repo_root  = (script_dir / '../..').resolve()

    out_dir = (Path(args.out_dir).resolve() if args.out_dir
               else repo_root / 'results' / 'uganda' / f'prithvi_l5_{args.d_hidden}')
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── Load embeddings ───────────────────────────────────────────────────────
    train_path = (script_dir / args.train_embeddings).resolve()
    rct_path   = (script_dir / args.rct_embeddings).resolve()
    keys_path  = (script_dir / args.rct_keys).resolve()
    data_csv   = (script_dir / args.data_csv).resolve()

    train_embs = np.load(train_path).astype(np.float32)
    rct_embs   = np.load(rct_path).astype(np.float32)
    rct_keys   = np.load(keys_path)
    d_in       = train_embs.shape[1]

    print(f"Train (national grid) embeddings : {train_embs.shape}")
    print(f"RCT (experimental sites) emb     : {rct_embs.shape}")
    print(f"SAE: d_in={d_in}  d_hidden={args.d_hidden}  k={args.k}")

    # ── Whitening — fit on national corpus only ───────────────────────────────
    w_mean = train_embs.mean(0, keepdims=True)
    w_std  = train_embs.std(0,  keepdims=True).clip(1e-6)

    train_w = torch.from_numpy((train_embs - w_mean) / w_std)
    rct_w   = torch.from_numpy((rct_embs   - w_mean) / w_std)

    # ── Cross-validation on national corpus ───────────────────────────────────
    df_cv = None
    if args.cv_folds > 1:
        print(f"\n[CV] {args.cv_folds}-fold on national grid (N={len(train_w)}) …")
        df_cv = run_cv(d_in, args.d_hidden, args.k, train_w,
                       args.epochs, args.lr, args.noise_std,
                       args.batch_size, device, args.cv_folds, args.seed)

    # ── Final SAE — trained on full national corpus ───────────────────────────
    n_params = (d_in * args.d_hidden + args.d_hidden
                + args.d_hidden * d_in + d_in)
    print(f"\n[Final] Training SAE on {len(train_w)} national samples …")
    print(f"  params≈{n_params:,}  sparsity={args.k/args.d_hidden:.2%}")

    sae_final = train_sae(d_in, args.d_hidden, args.k, train_w,
                          args.epochs, args.lr, args.noise_std,
                          args.batch_size, device, seed=args.seed)

    # ── Evaluate on RCT sites (never seen during training) ────────────────────
    rct_mse = reconstruction_mse(sae_final, rct_w, device)
    print(f"\n[Apply] Reconstruction on RCT sites (N={len(rct_embs)}, held out):")
    print(f"  MSE mean={rct_mse.mean():.4f}  max={rct_mse.max():.4f}  "
          f"var_explained={1 - rct_mse.mean() / rct_w.var().item():.1%}")

    sae_final.eval()
    with torch.no_grad():
        _, activations = sae_final(rct_w.to(device))
        activations    = activations.cpu().numpy()    # (N_rct, d_hidden)

    n_active = (activations > 0).sum(axis=1)
    n_live   = (activations > 0).any(axis=0).sum()
    cov      = (activations > 0).sum(axis=0)
    print(f"  Active/sample : mean={n_active.mean():.1f} "
          f"min={n_active.min()} max={n_active.max()}")
    print(f"  Live latents  : {n_live}/{args.d_hidden} "
          f"({n_live/args.d_hidden:.1%})")
    print(f"  Coverage/latent: mean={cov[cov>0].mean():.1f} "
          f"median={np.median(cov[cov>0]):.0f}")

    # ── Uganda-pipeline outputs ───────────────────────────────────────────────
    print(f"\n[Uganda] Building individual-level feature matrix …")
    ind_feats = make_individual_features(activations, rct_keys, data_csv, args.d_hidden)

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(sae_final.state_dict(), out_dir / 'sae_model.pt')

    np.save(out_dir / 'sae_whiten_mean.npy', w_mean)
    np.save(out_dir / 'sae_whiten_std.npy',  w_std)

    df_recon = pd.DataFrame({
        'site_key':  rct_keys,
        'recon_mse': rct_mse,
        'n_active':  n_active,
    })
    df_recon.to_csv(out_dir / 'sae_reconstruction_error.csv', index=False)
    if df_cv is not None:
        df_cv.to_csv(out_dir / 'sae_cv_results.csv', index=False)

    np.savez_compressed(
        out_dir / 'site_features.npz',
        site_features = activations,
        site_keys     = rct_keys,
    )
    np.savez_compressed(
        out_dir / 'individual_features.npz',
        features = ind_feats,
    )
    np.savez_compressed(
        out_dir / 'patch_embeddings.npz',
        embeddings = rct_embs,
        keys       = rct_keys,
    )

    print(f"\nSaved to {out_dir}/")
    for fname in ['sae_model.pt', 'site_features.npz',
                  'individual_features.npz', 'sae_reconstruction_error.csv']:
        print(f"  {fname}")

    print(f"\nTo run the NEXIS analysis pipeline:")
    print(f"  bash scripts/uganda/run.sh --models=prithvi_l5 --all-outcomes")


if __name__ == '__main__':
    main()
