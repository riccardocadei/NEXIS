"""TopK sparse autoencoder (overcomplete library): training on patch tokens, encoding.

Architecture: overcomplete.sae.TopKSAE, a linear encoder followed by ReLU and TopK
(k active codes) and a unit-norm dictionary.  Training: every SigLIP 2 patch token is a
training vector (a batch of B images gives B x 196 vectors), mean-squared reconstruction
loss, Adam, gradient clipping at norm 1, no input normalisation.  Encoding: the trained
encoder is applied to the mean of each image's 196 patch tokens, giving
  z_pre = the encoder pre-activations (dense)   and   z = TopK(ReLU(z_pre)) (sparse).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class _PatchDataset(Dataset):
    """(N, T, d) patch array -> one (T, d) float32 tensor per image."""

    def __init__(self, arr: np.ndarray):
        self.arr = arr

    def __len__(self) -> int:
        return len(self.arr)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.arr[idx], dtype=np.float32))


def _collate(batch):
    return torch.cat(batch, dim=0)          # (B, T, d) -> (B*T, d)


def open_patches(path: Path, n_images: int, dim: int) -> np.memmap:
    n_tokens = int(path.stat().st_size // (n_images * dim * 2))
    return np.memmap(path, dtype=np.float16, mode="r", shape=(n_images, n_tokens, dim))


def train_topk_sae(patches: np.ndarray, hidden_dim: int, top_k: int, epochs: int,
                   batch_images: int, lr: float, seed: int = 0, device: str | None = None):
    from overcomplete.sae import TopKSAE

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    n, n_tok, d = patches.shape
    loader = DataLoader(_PatchDataset(patches), batch_size=batch_images, shuffle=True,
                        collate_fn=_collate, drop_last=False, num_workers=0)
    print(f"TopK SAE: {n} images x {n_tok} tokens = {n * n_tok:,} vectors/epoch, "
          f"m={hidden_dim}, k={top_k}, device={device}", flush=True)
    sae = TopKSAE(input_shape=d, nb_concepts=hidden_dim, top_k=top_k, device=device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for epoch in range(epochs):
        losses, l0s = [], []
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            _, z, x_hat = sae(batch)
            loss = ((batch - x_hat) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            l0s.append((z > 0).float().sum(dim=-1).mean().item())
        print(f"  epoch {epoch + 1:2d}/{epochs}  mse={np.mean(losses):.4f}  "
              f"L0={np.mean(l0s):.1f}", flush=True)
    return sae.cpu()


def save_sae(sae, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sae.state_dict(), "sae_type": "topk",
                "input_dim": sae.dictionary.in_dimensions, "hidden_dim": sae.nb_concepts,
                "top_k": sae.top_k, "input_mean": None, "input_std": None}, path)


def load_sae(path: Path):
    from overcomplete.sae import TopKSAE

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sae = TopKSAE(input_shape=ckpt["input_dim"], nb_concepts=ckpt["hidden_dim"],
                  top_k=ckpt["top_k"])
    sae.load_state_dict(ckpt["state_dict"])
    return sae


@torch.no_grad()
def encode(sae, embeddings: np.ndarray, batch_size: int = 4096):
    """(z, z_pre) of mean-pooled embeddings, float32 (N, m) each."""
    X = embeddings.astype(np.float32)
    sae = sae.eval()
    zs, pres = [], []
    for start in range(0, len(X), batch_size):
        pre, z = sae.encode(torch.from_numpy(X[start:start + batch_size]))
        zs.append(z.numpy())
        pres.append(pre.numpy())
    return np.concatenate(zs, axis=0), np.concatenate(pres, axis=0)
