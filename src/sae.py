"""
Sparse Autoencoder (SAE) for learning interpretable representations.

Architecture:
    h = ReLU(W_enc @ (x - b_pre) + b_enc)   # sparse feature activations
    x_hat = W_dec @ h + b_dec                # reconstruction

Training loss:
    L = ||x - x_hat||²  +  λ * ||h||₁

The decoder columns are kept unit-norm after each gradient step so that
the L1 penalty on h has a consistent scale across features.

Typical usage:
    sae = SAE(input_dim=768, hidden_dim=3072)
    sae, losses = train_sae(embeddings, sae)
    features = sae.encode(embeddings)   # shape (N, hidden_dim)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SAE(nn.Module):
    """Sparse Autoencoder with unit-norm decoder columns.

    Args:
        input_dim: Dimension of input embeddings (e.g. 768 for DINOv2-B).
        hidden_dim: Number of learnable sparse features. Typically 4–16× input_dim.
        l1_coeff: Sparsity regularization strength λ.
    """

    def __init__(self, input_dim: int, hidden_dim: int, l1_coeff: float = 1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_coeff = l1_coeff

        # Pre-encoder bias (subtracted before encoding, added back after decoding)
        self.b_pre = nn.Parameter(torch.zeros(input_dim))

        self.W_enc = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.b_enc = nn.Parameter(torch.zeros(hidden_dim))

        self.W_dec = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_enc, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.W_dec, nonlinearity="relu")
        # Start with unit-norm decoder columns
        self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        """Project decoder columns onto the unit sphere."""
        norms = self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data /= norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode inputs to sparse feature activations.

        Args:
            x: Input tensor of shape (..., input_dim).

        Returns:
            h: Non-negative sparse features of shape (..., hidden_dim).
        """
        return torch.relu((x - self.b_pre) @ self.W_enc + self.b_enc)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Decode feature activations back to input space.

        Args:
            h: Feature tensor of shape (..., hidden_dim).

        Returns:
            x_hat: Reconstructed input of shape (..., input_dim).
        """
        return h @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full encode → decode pass.

        Returns:
            x_hat: Reconstruction of shape (..., input_dim).
            h: Sparse features of shape (..., hidden_dim).
        """
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute total loss and diagnostic terms.

        Returns:
            total_loss: Scalar tensor.
            metrics: Dict with keys recon_loss, sparsity_loss, l0 (avg nonzero per sample).
        """
        x_hat, h = self(x)
        recon = ((x - x_hat) ** 2).sum(dim=-1).mean()
        sparsity = h.abs().sum(dim=-1).mean()
        total = recon + self.l1_coeff * sparsity
        l0 = (h > 0).float().sum(dim=-1).mean()
        return total, {"recon_loss": recon.item(), "sparsity_loss": sparsity.item(), "l0": l0.item()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class SAETrainConfig:
    """Hyperparameters for SAE training."""
    hidden_dim: int = 3072          # number of SAE features (default: 4× DINOv2-B dim)
    l1_coeff: float = 1e-3          # sparsity penalty weight
    lr: float = 1e-4                # Adam learning rate
    batch_size: int = 512
    num_epochs: int = 50
    normalize_inputs: bool = True   # standardize embeddings before training
    device: Optional[str] = None    # None → auto-select
    log_every: int = 5              # print loss every N epochs
    seed: int = 0


@dataclass
class SAETrainResult:
    """Returned by train_sae."""
    sae: SAE
    losses: list[dict] = field(default_factory=list)   # one entry per epoch
    input_mean: Optional[np.ndarray] = None
    input_std: Optional[np.ndarray] = None


def train_sae(
    embeddings: np.ndarray,
    cfg: SAETrainConfig | None = None,
    sae: SAE | None = None,
) -> SAETrainResult:
    """Train a Sparse Autoencoder on a set of dense embeddings.

    Args:
        embeddings: Float array of shape (N, d), e.g. DINOv2 CLS features.
        cfg: Training configuration. Uses defaults if None.
        sae: Pre-existing SAE to continue training. Built from cfg if None.

    Returns:
        SAETrainResult with the trained model and per-epoch loss logs.
    """
    if cfg is None:
        cfg = SAETrainConfig()

    device = cfg.device or _auto_device()
    torch.manual_seed(cfg.seed)

    # --- Normalise inputs ---
    input_mean, input_std = None, None
    X = embeddings.astype(np.float32)
    if cfg.normalize_inputs:
        input_mean = X.mean(axis=0)
        input_std = X.std(axis=0).clip(min=1e-8)
        X = (X - input_mean) / input_std

    X_tensor = torch.from_numpy(X).to(device)
    loader = DataLoader(
        TensorDataset(X_tensor),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    # --- Build or validate model ---
    input_dim = X.shape[1]
    if sae is None:
        sae = SAE(input_dim=input_dim, hidden_dim=cfg.hidden_dim, l1_coeff=cfg.l1_coeff)
    else:
        assert sae.input_dim == input_dim, (
            f"SAE input_dim={sae.input_dim} does not match embeddings dim={input_dim}"
        )
    sae = sae.to(device)

    optimiser = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    # --- Training loop ---
    epoch_losses: list[dict] = []
    for epoch in range(cfg.num_epochs):
        batch_metrics: list[dict] = []
        for (batch,) in loader:
            optimiser.zero_grad()
            total, metrics = sae.loss(batch)
            total.backward()
            optimiser.step()
            # Keep decoder unit-norm after each step
            sae._normalize_decoder()
            batch_metrics.append(metrics)

        # Average metrics over batches
        avg = {k: float(np.mean([m[k] for m in batch_metrics])) for k in batch_metrics[0]}
        avg["epoch"] = epoch
        epoch_losses.append(avg)

        if cfg.log_every and (epoch % cfg.log_every == 0 or epoch == cfg.num_epochs - 1):
            print(
                f"Epoch {epoch:4d}/{cfg.num_epochs} | "
                f"recon={avg['recon_loss']:.4f}  "
                f"sparsity={avg['sparsity_loss']:.4f}  "
                f"L0={avg['l0']:.1f}"
            )

    return SAETrainResult(
        sae=sae.cpu(),
        losses=epoch_losses,
        input_mean=input_mean,
        input_std=input_std,
    )


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_features(
    embeddings: np.ndarray,
    result: SAETrainResult,
    batch_size: int = 1024,
) -> np.ndarray:
    """Encode a set of embeddings to SAE feature activations.

    Applies the same normalisation used during training.

    Args:
        embeddings: Float array of shape (N, d).
        result: SAETrainResult from train_sae.
        batch_size: Chunk size for inference.

    Returns:
        Float32 array of shape (N, hidden_dim).
    """
    X = embeddings.astype(np.float32)
    if result.input_mean is not None:
        X = (X - result.input_mean) / result.input_std.clip(min=1e-8)

    sae = result.sae.eval()
    out = []
    for start in range(0, len(X), batch_size):
        chunk = torch.from_numpy(X[start : start + batch_size])
        h = sae.encode(chunk)
        out.append(h.numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
