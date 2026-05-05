"""Train a Sparse Autoencoder (SAE) on Prithvi satellite embeddings.

Architecture: TopK SAE (Gao et al. 2024, "Scaling and evaluating sparse
autoencoders"). The encoder maps each d_in-dim embedding into a large latent
space; only the top-k activations are retained per sample, forcing the model
towards monosemantic features (one latent ≈ one visual concept: vegetation
density, settlement pattern, water proximity, bare soil, etc.).

Data roles — kept strictly separate:
  TRAINING corpus  → used to fit the SAE weights
  EXPERIMENT data  → the 162 LEAP community embeddings, NEVER seen during
                     training; the SAE is applied to them after training

Recommended workflow (after running download_national_grid.py):
  1. Extract Prithvi embeddings from the national grid tiles:
       python extract_satellite_features.py \\
         --tif-dir ../../data/ghana/satellite/tif_national \\
         --out-dir ../../data/ghana/satellite/national
  2. Train SAE on the national corpus, evaluate on LEAP communities:
       python train_sae.py \\
         --train-embeddings ../../data/ghana/satellite/national/prithvi_embeddings.npy \\
         --eval-embeddings  ../../data/ghana/satellite/prithvi_embeddings.npy \\
         --eval-ids         ../../data/ghana/satellite/prithvi_comm_ids.npy

Fallback (national data not yet downloaded):
  Run without --train-embeddings; the script trains on the 162 LEAP embeddings
  with 5-fold cross-validation to estimate held-out reconstruction quality.

Cross-validation:
  When train == eval (fallback mode), k-fold CV is run: for each fold the SAE
  is trained from scratch on the remaining folds and evaluated on the held-out
  fold. This gives an honest reconstruction metric uncontaminated by overfitting.
  When train != eval (recommended mode), CV is run on a 80/20 split of the
  training corpus as a training sanity check; the LEAP evaluation is the primary
  held-out metric.

Outputs (--out-dir):
  sae_model.pt                — final SAE weights (trained on full training corpus)
  sae_activations.npy         — (N_eval, d_hidden) sparse activations for LEAP communities
  sae_comm_ids.npy            — (N_eval,) community IDs aligned with activations
  sae_reconstruction_error.csv— per-community reconstruction MSE + n_active latents
  sae_cv_results.csv          — per-fold CV reconstruction MSE
  sae_whiten_mean/std.npy     — whitening stats (needed to apply SAE to new embeddings)
"""

import argparse
from copy import deepcopy
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
    """Sparse Autoencoder with top-k activation, unit-norm decoder, and aux loss.

    Main loss: MSE reconstruction (Gao et al. 2024, TopK variant).
    Auxiliary loss: dead latents are nudged to explain the reconstruction
      residual — the part of the input that live latents failed to capture.
      This gives gradient signal to latents that top-k would otherwise
      permanently starve of updates.

    Dead-latent tracking: EMA of per-latent activation frequency. A latent
    is "dead" when its EMA frequency drops below dead_threshold (default 1e-4,
    i.e. active in fewer than 0.01% of samples seen recently).
    """

    def __init__(self, d_in: int, d_hidden: int, k: int,
                 dead_threshold: float = 1e-4, ema_decay: float = 0.999):
        super().__init__()
        self.d_in           = d_in
        self.d_hidden       = d_hidden
        self.k              = k
        self.dead_threshold = dead_threshold
        self.ema_decay      = ema_decay

        self.W_enc = nn.Linear(d_in, d_hidden, bias=True)
        self.W_dec = nn.Parameter(torch.empty(d_hidden, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # EMA of activation frequency — not a learnable parameter
        self.register_buffer('ema_freq', torch.zeros(d_hidden))

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
    def update_ema(self, z: torch.Tensor):
        """Update activation-frequency EMA from a batch of activations."""
        freq = (z > 0).float().mean(dim=0)
        self.ema_freq.mul_(self.ema_decay).add_(freq * (1 - self.ema_decay))

    def auxiliary_loss(self, x: torch.Tensor, x_hat: torch.Tensor,
                       aux_coeff: float = 1 / 32) -> torch.Tensor:
        """Aux loss for dead latents (Gao et al. 2024 §3.3).

        Dead latents are asked to reconstruct the residual (x - x̂) using
        their pre-topk activations unit-normed per sample — so only direction
        matters, not magnitude.  The coeff keeps aux loss comparable in scale
        to the main loss.
        """
        dead = self.ema_freq < self.dead_threshold
        if not dead.any():
            return x.new_tensor(0.0)

        # Pre-topk ReLU activations (bypasses top-k mask → gradients flow)
        pre_acts  = F.relu(self.W_enc(x - self.b_dec))   # (B, d_hidden)
        dead_pre  = pre_acts[:, dead]                      # (B, n_dead)

        # Unit-normalise per sample so scale doesn't swamp direction signal
        norms      = dead_pre.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        dead_normed = dead_pre / norms                     # (B, n_dead)

        W_dead     = F.normalize(self.W_dec[dead], dim=1) # (n_dead, d_in)
        x_hat_dead = dead_normed @ W_dead                  # (B, d_in)

        residual   = (x - x_hat).detach()                 # target: what live latents missed
        return F.mse_loss(x_hat_dead, residual) * aux_coeff

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)


# ── Training ──────────────────────────────────────────────────────────────────

def train_sae(
    d_in:       int,
    d_hidden:   int,
    k:          int,
    embeddings: torch.Tensor,   # (N, d_in), already whitened
    epochs:     int,
    lr:         float,
    noise_std:  float,
    batch_size: int,
    device:     torch.device,
    aux_coeff:  float = 1 / 32,
    seed:       int = 42,
    verbose:    bool = True,
) -> TopKSAE:
    torch.manual_seed(seed)
    sae       = TopKSAE(d_in, d_hidden, k).to(device)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=lr, weight_decay=0.0)
    loader    = DataLoader(TensorDataset(embeddings.to(device)),
                           batch_size=batch_size, shuffle=True, drop_last=False)

    for epoch in range(1, epochs + 1):
        epoch_main = epoch_aux = 0.0
        for (x,) in loader:
            x_noisy     = x + torch.randn_like(x) * noise_std if noise_std > 0 else x
            x_hat, z    = sae(x_noisy)
            main_loss   = F.mse_loss(x_hat, x)
            aux_loss    = sae.auxiliary_loss(x, x_hat, aux_coeff)
            loss        = main_loss + aux_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()
            sae.update_ema(z.detach())
            epoch_main += main_loss.item()
            epoch_aux  += aux_loss.item()

        epoch_main /= len(loader)
        epoch_aux  /= len(loader)
        n_dead = (sae.ema_freq < sae.dead_threshold).sum().item()
        if verbose and epoch % max(1, epochs // 10) == 0:
            print(f"    epoch {epoch:5d}/{epochs}  "
                  f"main={epoch_main:.5f}  aux={epoch_aux:.5f}  "
                  f"dead={n_dead}/{d_hidden}")

    return sae


def reconstruction_mse(sae: TopKSAE, embeddings: torch.Tensor, device: torch.device) -> np.ndarray:
    """Per-sample reconstruction MSE on whitened embeddings."""
    sae.eval()
    with torch.no_grad():
        x     = embeddings.to(device)
        x_hat = sae(x)[0]
        mse   = ((x - x_hat) ** 2).mean(dim=1).cpu().numpy()
    return mse


# ── Cross-validation ──────────────────────────────────────────────────────────

def run_cv(
    d_in:       int,
    d_hidden:   int,
    k:          int,
    embeddings: torch.Tensor,
    epochs:     int,
    lr:         float,
    noise_std:  float,
    batch_size: int,
    device:     torch.device,
    n_folds:    int,
    seed:       int,
    aux_coeff:  float = 1 / 32,
) -> pd.DataFrame:
    kf   = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    print(f"\n  Cross-validation ({n_folds} folds, training from scratch each fold) …")
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(embeddings))), 1):
        train_emb = embeddings[train_idx]
        val_emb   = embeddings[val_idx]
        print(f"  Fold {fold}/{n_folds}:  train={len(train_idx)}  val={len(val_idx)}")
        sae  = train_sae(d_in, d_hidden, k, train_emb, epochs, lr,
                         noise_std, batch_size, device,
                         aux_coeff=aux_coeff, seed=seed+fold, verbose=False)
        mse  = reconstruction_mse(sae, val_emb, device)
        rows.append({'fold': fold, 'n_val': len(val_idx),
                     'val_mse_mean': mse.mean(), 'val_mse_std': mse.std()})
        print(f"           val MSE = {mse.mean():.4f} ± {mse.std():.4f}")

    df_cv = pd.DataFrame(rows)
    print(f"\n  CV summary:  mean={df_cv['val_mse_mean'].mean():.4f}  "
          f"std={df_cv['val_mse_mean'].std():.4f}")
    return df_cv


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # Data paths
    p.add_argument('--train-embeddings', required=True,
                   help='(N_train, D) .npy — training corpus (national grid). '
                        'LEAP experiment embeddings must NEVER appear here.')
    p.add_argument('--eval-embeddings',
                   default='../../data/ghana/satellite/prithvi_embeddings.npy',
                   help='(N_eval, D) .npy — LEAP experiment embeddings (never used for training)')
    p.add_argument('--eval-ids',
                   default='../../data/ghana/satellite/prithvi_comm_ids.npy',
                   help='(N_eval,) .npy — community IDs for eval embeddings')
    p.add_argument('--out-dir', default='../../data/ghana/satellite')

    # Architecture
    p.add_argument('--d-hidden',  type=int,   default=4096,
                   help='SAE latent dim (default 4096; use 512 if training only on 162 LEAP)')
    p.add_argument('--k',         type=int,   default=25,
                   help='Top-k active latents per sample (default 25)')

    # Training
    p.add_argument('--epochs',    type=int,   default=2000)
    p.add_argument('--lr',        type=float, default=2e-4)
    p.add_argument('--noise-std', type=float, default=0.05)
    p.add_argument('--batch-size', type=int,  default=64)
    p.add_argument('--aux-coeff', type=float, default=1/32,
                   help='Auxiliary loss weight for dead latents (default 1/32)')
    p.add_argument('--cv-folds',  type=int,   default=5,
                   help='CV folds (default 5; set 0 to skip CV)')
    p.add_argument('--device',    default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',      type=int,   default=42)
    return p.parse_args()


def main():
    args       = parse_args()
    script_dir = Path(__file__).parent
    out_dir    = (script_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device     = torch.device(args.device)

    # ── Load embeddings ───────────────────────────────────────────────────────
    eval_embs = np.load((script_dir / args.eval_embeddings).resolve()).astype(np.float32)
    eval_ids  = np.load((script_dir / args.eval_ids).resolve())
    d_in      = eval_embs.shape[1]
    print(f"Eval (LEAP experiment) embeddings: {eval_embs.shape}")

    train_embs = np.load((script_dir / args.train_embeddings).resolve()).astype(np.float32)
    print(f"Train (national grid) embeddings:  {train_embs.shape}")

    # ── Whitening — fit on training corpus only ───────────────────────────────
    w_mean = train_embs.mean(0, keepdims=True)
    w_std  = train_embs.std(0,  keepdims=True).clip(1e-6)

    train_w = torch.from_numpy((train_embs - w_mean) / w_std)
    eval_w  = torch.from_numpy((eval_embs  - w_mean) / w_std)

    # ── Cross-validation on training corpus (national grid only) ─────────────
    # The LEAP experiment data plays no role here.
    df_cv = None
    if args.cv_folds > 1:
        print(f"\n[CV] {args.cv_folds}-fold cross-validation on national grid  "
              f"(N={len(train_w)}) …")
        df_cv = run_cv(d_in, args.d_hidden, args.k, train_w,
                       args.epochs, args.lr, args.noise_std,
                       args.batch_size, device, args.cv_folds, args.seed,
                       aux_coeff=args.aux_coeff)

    # ── Final SAE — trained on full training corpus ───────────────────────────
    n_params = (d_in * args.d_hidden + args.d_hidden   # W_enc, b_enc
                + args.d_hidden * d_in + d_in)          # W_dec, b_dec
    print(f"\n[Final] Training SAE on {len(train_w)} samples …")
    print(f"  d_in={d_in}  d_hidden={args.d_hidden}  k={args.k}  "
          f"params≈{n_params:,}  sparsity={args.k/args.d_hidden:.2%}")

    sae_final = train_sae(d_in, args.d_hidden, args.k, train_w,
                          args.epochs, args.lr, args.noise_std,
                          args.batch_size, device,
                          aux_coeff=args.aux_coeff, seed=args.seed)

    # ── Evaluate on LEAP experiment data ──────────────────────────────────────
    eval_mse = reconstruction_mse(sae_final, eval_w, device)
    print(f"\n[Apply] Reconstruction on LEAP experiment set (N={len(eval_embs)}, never seen during training):")
    print(f"  MSE mean={eval_mse.mean():.4f}  max={eval_mse.max():.4f}  "
          f"var_explained={1 - eval_mse.mean() / eval_w.var().item():.1%}")

    sae_final.eval()
    with torch.no_grad():
        _, activations = sae_final(eval_w.to(device))
        activations    = activations.cpu().numpy()

    n_active   = (activations > 0).sum(axis=1)
    n_live     = (activations > 0).any(axis=0).sum()
    cov        = (activations > 0).sum(axis=0)
    print(f"  Active latents/sample: mean={n_active.mean():.1f}  "
          f"min={n_active.min()}  max={n_active.max()}")
    print(f"  Live latents: {n_live}/{args.d_hidden}  ({n_live/args.d_hidden:.1%} utilisation)")
    print(f"  Coverage/latent: mean={cov[cov>0].mean():.1f}  "
          f"median={np.median(cov[cov>0]):.0f}  max={cov.max()}")

    # ── Save ──────────────────────────────────────────────────────────────────
    df_recon = pd.DataFrame({
        'comm_id':    eval_ids,
        'recon_mse':  eval_mse,
        'n_active':   n_active,
    })

    torch.save(sae_final.state_dict(),         out_dir / 'sae_model.pt')
    np.save(out_dir / 'sae_activations.npy',   activations)
    np.save(out_dir / 'sae_comm_ids.npy',      eval_ids)
    np.save(out_dir / 'sae_whiten_mean.npy',   w_mean)
    np.save(out_dir / 'sae_whiten_std.npy',    w_std)
    df_recon.to_csv(out_dir / 'sae_reconstruction_error.csv', index=False)
    if df_cv is not None:
        df_cv.to_csv(out_dir / 'sae_cv_results.csv', index=False)
        print(f"  {out_dir / 'sae_cv_results.csv'}")

    print(f"\nSaved:")
    for fname in ['sae_model.pt', 'sae_activations.npy', 'sae_comm_ids.npy',
                  'sae_reconstruction_error.csv']:
        print(f"  {out_dir / fname}")


if __name__ == '__main__':
    main()
