"""
CelebA semi-synthetic SCM for effect-modifier discovery.

DGP
---
  T  ~ Bernoulli(p_treat)
  W1 ~ Bernoulli(p_w1)   — drawn from CelebA empirical prevalence
  W2 ~ Bernoulli(p_w2)   — drawn from CelebA empirical prevalence
  X  ~ CelebA image matching (W1, W2)   [sampled without replacement]
  Z  = SAE(ViT(X))                      [pre-computed SAE features]
  Y  = beta_w1*W1 + beta_w2*W2
       + T * [tau_0 + effect_scale*(gamma_w1*W1 + gamma_w2*W2)]
       + noise

Default attributes
  W1 = "Wearing_Hat"  (prevalence ≈ 5%)   — positive T×W1 modification
  W2 = "Eyeglasses"   (prevalence ≈ 7%)   — negative T×W2 modification

The ground truth for NEIS evaluation is the set of SAE neurons whose
activations are most correlated with W1 / W2 across the full CelebA set.
NEIS should recover these through significant T×Z_j interactions with Y.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CelebAData:
    T: np.ndarray           # (n,) float64, binary treatment
    W1: np.ndarray          # (n,) float64, binary effect modifier 1
    W2: np.ndarray          # (n,) float64, binary effect modifier 2
    Z: np.ndarray           # (n, hidden_dim) float64, SAE features
    Y: np.ndarray           # (n,) float64, continuous outcome
    image_indices: np.ndarray  # (n,) int64, indices into CelebA dataset


# ---------------------------------------------------------------------------
# Bucket construction
# ---------------------------------------------------------------------------

def build_buckets(
    labels_df: pd.DataFrame,
    w1_attr: str,
    w2_attr: str,
) -> Dict[Tuple[int, int], List[int]]:
    """
    Pre-stratify CelebA images by (W1, W2) value.

    Returns a dict mapping (w1, w2) ∈ {0,1}² → sorted list of row indices
    in labels_df that have that attribute combination.
    """
    w1 = labels_df[w1_attr].values.astype(int)
    w2 = labels_df[w2_attr].values.astype(int)
    buckets: Dict[Tuple[int, int], List[int]] = {
        (0, 0): [], (0, 1): [], (1, 0): [], (1, 1): [],
    }
    for i, (v1, v2) in enumerate(zip(w1, w2)):
        buckets[(int(v1), int(v2))].append(i)
    return buckets


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_celeba_rct(
    n: int,
    features: np.ndarray,
    labels_df: pd.DataFrame,
    buckets: Dict[Tuple[int, int], List[int]],
    w1_attr: str = "Wearing_Hat",
    w2_attr: str = "Eyeglasses",
    p_treat: float = 0.5,
    tau_0: float = 0.5,
    beta_w1: float = 0.3,
    beta_w2: float = -0.2,
    gamma_w1: float = 1.0,
    gamma_w2: float = -1.0,
    noise_sd: float = 1.0,
    effect_scale: float = 1.0,
    seed: Optional[int] = None,
) -> CelebAData:
    """
    Draw one semi-synthetic RCT sample of size n.

    Images are sampled without replacement within each (W1, W2) bucket so
    that every unit in the sample corresponds to a unique CelebA image.

    Args:
        n:            Sample size.
        features:     (N_celeba, hidden_dim) pre-computed SAE features.
        labels_df:    CelebA attribute table; index aligns with features rows.
        buckets:      Output of build_buckets().
        w1_attr:      Column name of W1 attribute in labels_df.
        w2_attr:      Column name of W2 attribute in labels_df.
        p_treat:      Treatment probability.
        tau_0:        Main ATE (homogeneous part).
        beta_w1/w2:   Main effects of W1/W2 on Y.
        gamma_w1/w2:  T×W1 / T×W2 interaction coefficients (effect modification).
        noise_sd:     Gaussian noise standard deviation.
        effect_scale: Multiplier for gamma terms (sweep this to vary effect size).
        seed:         RNG seed for reproducibility.

    Returns:
        CelebAData with T, W1, W2, Z, Y, image_indices.

    Raises:
        ValueError if any (W1, W2) bucket runs out of images for the requested n.
    """
    rng = np.random.default_rng(seed)

    p_w1 = float(labels_df[w1_attr].mean())
    p_w2 = float(labels_df[w2_attr].mean())

    W1 = rng.binomial(1, p_w1, size=n).astype(np.int32)
    W2 = rng.binomial(1, p_w2, size=n).astype(np.int32)
    T  = rng.binomial(1, p_treat, size=n).astype(np.float64)

    # Shuffle each bucket once, then draw sequentially (= without replacement)
    bucket_perms: Dict[Tuple[int, int], np.ndarray] = {
        k: rng.permutation(v) for k, v in buckets.items()
    }
    bucket_ptrs: Dict[Tuple[int, int], int] = {k: 0 for k in buckets}

    image_idx = np.empty(n, dtype=np.int64)
    for i, (w1, w2) in enumerate(zip(W1, W2)):
        key = (int(w1), int(w2))
        ptr = bucket_ptrs[key]
        perm = bucket_perms[key]
        if ptr >= len(perm):
            raise ValueError(
                f"Bucket ({w1},{w2}) exhausted: size={len(perm)}, "
                f"needed >{ptr} images. "
                f"Try smaller n or attributes with higher prevalence."
            )
        image_idx[i] = perm[ptr]
        bucket_ptrs[key] += 1

    Z = features[image_idx].astype(np.float64)  # (n, hidden_dim)

    tau = tau_0 + effect_scale * (gamma_w1 * W1 + gamma_w2 * W2)
    Y = (
        beta_w1 * W1
        + beta_w2 * W2
        + tau * T
        + rng.normal(0.0, noise_sd, size=n)
    ).astype(np.float64)

    return CelebAData(
        T=T,
        W1=W1.astype(np.float64),
        W2=W2.astype(np.float64),
        Z=Z,
        Y=Y,
        image_indices=image_idx,
    )
