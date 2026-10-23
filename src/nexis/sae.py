"""TopK sparse autoencoder (SAE) training and encoding.

TopKSAE comes from the `overcomplete` library: a Linear -> BatchNorm1d -> ReLU -> TopK
encoder and an L2-normalised dictionary.  Needs the `gpu` extra (torch, overcomplete);
`import nexis` does not import this module, import it as `nexis.sae`.

    result = train_topk_sae(embeddings, TopKSAETrainConfig())
    features = get_features(embeddings, result)          # (N, hidden_dim), post-top-k
    pre_features = get_pre_features(embeddings, result)  # (N, hidden_dim), pre-activation
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from overcomplete.sae import TopKSAE  # Linear → BatchNorm1d → ReLU → TopK

try:
    from tqdm import tqdm as _tqdm
    def _progress(it, **kw): return _tqdm(it, **kw)
except ImportError:
    def _progress(it, **kw): return it


# ---------------------------------------------------------------------------
# Training config / result for TopKSAE
# ---------------------------------------------------------------------------

@dataclass
class TopKSAETrainConfig:
    """Hyperparameters for TopKSAE training (defaults: the CelebA runs)."""
    hidden_dim:       int   = 9216    # 12 × SigLIP base dim (768)
    top_k:            int   = 5
    lr:               float = 5e-4
    batch_size:       int   = 20      # images; each collated to batch_size × T patch vectors
    num_epochs:       int   = 20
    normalize_inputs: bool  = True    # forced False in per-patch mode
    device:           str | None = None
    log_every:        int   = 1
    seed:             int   = 0
    per_patch:        bool  = False


@dataclass
class SAETrainResult:
    """Returned by train_topk_sae."""
    sae: TopKSAE
    losses: list[dict] = field(default_factory=list)
    input_mean: np.ndarray | None = None
    input_std:  np.ndarray | None = None


# ---------------------------------------------------------------------------
# Per-patch dataset helpers
# ---------------------------------------------------------------------------

class _PatchDataset(torch.utils.data.Dataset):
    """Wraps a per-patch array of shape (N, T, d); yields (T, d) float32 tensors."""

    def __init__(self, arr: np.ndarray):
        self.arr = arr

    def __len__(self) -> int:
        return len(self.arr)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.arr[idx], dtype=np.float32))


def _patch_collate_fn(batch: list[torch.Tensor]) -> torch.Tensor:
    """Collapse (B, T, d) → (B*T, d) for SAE training."""
    return torch.cat(batch, dim=0)


# ---------------------------------------------------------------------------
# Training — TopKSAE (overcomplete)
# ---------------------------------------------------------------------------

def train_topk_sae(
    embeddings: np.ndarray,
    cfg: TopKSAETrainConfig | None = None,
) -> SAETrainResult:
    """Train an overcomplete TopKSAE on pre-computed embeddings.

    ``embeddings`` can be:
    * shape ``(N, d)``     — standard mean-pooled mode
    * shape ``(N, T, d)``  — per-patch mode (set ``cfg.per_patch=True``)
    """
    if cfg is None:
        cfg = TopKSAETrainConfig()

    device = cfg.device or _auto_device()
    torch.manual_seed(cfg.seed)

    # ── Per-patch mode ─────────────────────────────────────────────────────
    if embeddings.ndim == 3 or cfg.per_patch:
        if embeddings.ndim != 3:
            raise ValueError("per_patch=True requires embeddings of shape (N, T, d)")
        N, T, d = embeddings.shape
        input_mean, input_std = None, None
        loader = DataLoader(
            _PatchDataset(embeddings),
            batch_size  = cfg.batch_size,
            shuffle     = True,
            collate_fn  = _patch_collate_fn,
            drop_last   = False,
            num_workers = 0,
            pin_memory  = False,
        )
        print(f"Per-patch training: {N} images × {T} patches = {N * T:,} vectors/epoch")

    else:
        # ── Standard (mean-pooled) mode ─────────────────────────────────────
        X = embeddings.astype(np.float32)
        input_mean, input_std = None, None
        if cfg.normalize_inputs:
            input_mean = X.mean(axis=0)
            input_std  = X.std(axis=0).clip(min=1e-8)
            X = (X - input_mean) / input_std

        X_tensor = torch.from_numpy(X).to(device)
        loader = DataLoader(
            TensorDataset(X_tensor),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
        )
        d = X.shape[1]

    # Build overcomplete TopKSAE: Linear → BatchNorm1d → ReLU → TopK
    sae = TopKSAE(
        input_shape = d,
        nb_concepts = cfg.hidden_dim,
        top_k       = cfg.top_k,
        device      = device,
    )

    optimiser = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    epoch_losses: list[dict] = []
    pbar = _progress(range(cfg.num_epochs), desc="TopKSAE training", unit="epoch",
                     dynamic_ncols=True)

    _is_patch = embeddings.ndim == 3 or cfg.per_patch

    for epoch in pbar:
        batch_metrics: list[dict] = []
        for batch_item in loader:
            batch = batch_item if _is_patch else batch_item[0]
            batch = batch.to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)

            # forward: (z_pre, z, x_hat)  — overcomplete signature
            z_pre, z, x_hat = sae(batch)
            recon = ((batch - x_hat) ** 2).mean()
            recon.backward()

            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimiser.step()
            # Dictionary columns are L2-normalised inside DictionaryLayer.forward()
            # — no explicit _normalize_decoder() call needed.

            l0 = (z > 0).float().sum(dim=-1).mean().item()
            batch_metrics.append({"recon_loss": recon.item(), "l0": l0})

        avg = {k: float(np.mean([m[k] for m in batch_metrics])) for k in batch_metrics[0]}
        avg["epoch"] = epoch
        epoch_losses.append(avg)

        if cfg.log_every and (epoch % cfg.log_every == 0 or epoch == cfg.num_epochs - 1):
            msg = f"recon={avg['recon_loss']:.4f}  L0={avg['l0']:.1f}"
            if hasattr(pbar, "set_postfix_str"):
                pbar.set_postfix_str(msg)
            else:
                print(f"Epoch {epoch:4d}/{cfg.num_epochs} | {msg}")

    sae = sae.cpu()
    return SAETrainResult(
        sae        = sae,
        losses     = epoch_losses,
        input_mean = input_mean,
        input_std  = input_std,
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
    """Encode embeddings to sparse TopK SAE activations (z, post-top-k).

    Returns float32 array of shape (N, hidden_dim).
    """
    X = embeddings.astype(np.float32)
    if result.input_mean is not None:
        X = (X - result.input_mean) / result.input_std.clip(min=1e-8)

    sae = result.sae.eval()
    out = []
    for start in range(0, len(X), batch_size):
        chunk = torch.from_numpy(X[start : start + batch_size])
        _, z = sae.encode(chunk)          # overcomplete: encode → (pre_codes, z)
        out.append(z.numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def get_pre_features(
    embeddings: np.ndarray,
    result: SAETrainResult,
    batch_size: int = 1024,
) -> np.ndarray:
    """Encode embeddings to continuous pre-activations (z_pre, before top-k masking).

    These dense scores are better suited for threshold-sweep F1 evaluation.
    Returns float32 array of shape (N, hidden_dim).
    """
    X = embeddings.astype(np.float32)
    if result.input_mean is not None:
        X = (X - result.input_mean) / result.input_std.clip(min=1e-8)

    sae = result.sae.eval()
    out = []
    for start in range(0, len(X), batch_size):
        chunk = torch.from_numpy(X[start : start + batch_size])
        z_pre, _ = sae.encode(chunk)      # overcomplete: encode → (pre_codes, z)
        out.append(z_pre.numpy())
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
