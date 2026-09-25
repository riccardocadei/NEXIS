"""Semi-synthetic randomized experiment on CelebA (Appendix C.1, Eq. C.1).

  W_k ~ Bernoulli(p_k), k = 1..r, independently, p_k = prevalence of attribute k in the pool
  T   ~ Bernoulli(p_treat)
  X   ~ a CelebA image of the cell (W_1, ..., W_r), drawn without replacement
  Z   = the pre-computed SAE representation of X
  Y   = sum_k beta_k W_k + T tau + N(0, noise_sd^2),
        tau = tau_0 + eta sum_k gamma_k W_k                           (effect_form "attr")
        tau = tau_0 + eta (gamma_1 g_1(Z^{j1}) + gamma_2 g_2(Z^{j2}))  ("ortho_quadratic")

The seed fixes the whole draw (W, T, images, noise), so datasets are reproducible cell by
cell and identical across methods, and across DGPs that share the sampled attributes.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class RCTSample:
    T: np.ndarray               # (n,) binary treatment
    W: np.ndarray               # (n, r) sampled attributes
    Z: np.ndarray               # (n, m) representation of the drawn images
    Y: np.ndarray               # (n,) outcome
    image_indices: np.ndarray   # (n,) rows of the pool


def build_buckets(labels_df: pd.DataFrame, attrs: Sequence[str]) -> Dict[Tuple[int, ...], List[int]]:
    """Pool rows grouped by the joint value of the attributes (all 2^r cells as keys)."""
    cols = np.stack([labels_df[a].values.astype(int) for a in attrs], axis=1)
    buckets: Dict[Tuple[int, ...], List[int]] = {key: [] for key in product((0, 1), repeat=len(attrs))}
    for i, row in enumerate(cols):
        buckets[tuple(int(v) for v in row)].append(i)
    return buckets


def max_supported_n(labels_df: pd.DataFrame, attrs: Sequence[str]) -> float:
    """Largest n for which every joint cell holds its expected share of images."""
    buckets = build_buckets(labels_df, attrs)
    p_w = np.array([float(labels_df[a].mean()) for a in attrs])
    cap = np.inf
    for key, rows in buckets.items():
        q = float(np.prod(np.where(np.array(key) == 1, p_w, 1.0 - p_w)))
        if q > 0:
            cap = min(cap, len(rows) / q)
    return cap


def ortho_quadratic_map(col: np.ndarray):
    """g(z): the residual of z~^2 on [1, z~] (z~ standardised), rescaled to unit variance.

    Fitted on the whole pool, so that E[g(Z^j)] = 0 and Cov(g(Z^j), Z^j) = 0 there: a
    treatment effect driven by g is a U-shape in Z^j with zero linear covariance.
    """
    z = np.asarray(col, dtype=float)
    mu, sd = z.mean(), z.std()
    sd = sd if sd > 1e-12 else 1.0
    zt = (z - mu) / sd
    q = zt ** 2
    b = float(np.cov(q, zt, ddof=0)[0, 1] / max(np.var(zt), 1e-12))
    a = float(q.mean() - b * zt.mean())
    g_pop = q - a - b * zt
    gsd = float(g_pop.std())
    gsd = gsd if gsd > 1e-12 else 1.0

    def g(x: np.ndarray) -> np.ndarray:
        xt = (np.asarray(x, dtype=float) - mu) / sd
        return (xt ** 2 - a - b * xt) / gsd

    return g


def generate_rct(
    n: int,
    features: np.ndarray,
    labels_df: pd.DataFrame,
    buckets: Dict[Tuple[int, ...], List[int]],
    attrs: Sequence[str],
    betas: Sequence[float],
    gammas: Sequence[float],
    effect_scale: float,
    seed: int,
    tau0: float = 0.5,
    noise_sd: float = 1.0,
    p_treat: float = 0.5,
    effect_form: str = "attr",
    modifier_cols: Optional[Sequence[int]] = None,
) -> RCTSample:
    """Draw one randomized experiment of size n.

    Raises ValueError when a joint attribute cell runs out of images.
    """
    rng = np.random.default_rng(seed)
    r = len(attrs)
    if len(betas) != r or len(gammas) != r:
        raise ValueError(f"betas and gammas need {r} values")

    p_w = [float(labels_df[a].mean()) for a in attrs]
    W = np.stack([rng.binomial(1, p, size=n) for p in p_w], axis=1).astype(np.int32)
    T = rng.binomial(1, p_treat, size=n).astype(np.float64)

    # shuffle each cell once, then draw sequentially (without replacement)
    perms = {k: rng.permutation(v) for k, v in buckets.items()}
    ptrs = {k: 0 for k in buckets}
    image_idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        key = tuple(int(v) for v in W[i])
        if ptrs[key] >= len(perms[key]):
            cell = ", ".join(f"{a}={v}" for a, v in zip(attrs, key))
            raise ValueError(f"cell ({cell}) exhausted: {len(perms[key])} images, n={n}")
        image_idx[i] = perms[key][ptrs[key]]
        ptrs[key] += 1

    Z = features[image_idx].astype(np.float64)

    if effect_form == "attr":
        tau = tau0 + effect_scale * (W * np.asarray(gammas, float)).sum(axis=1)
    elif effect_form == "ortho_quadratic":
        j1, j2 = int(modifier_cols[0]), int(modifier_cols[1])
        g1 = ortho_quadratic_map(features[:, j1])
        g2 = ortho_quadratic_map(features[:, j2])
        tau = tau0 + effect_scale * (gammas[0] * g1(Z[:, j1]) + gammas[1] * g2(Z[:, j2]))
    else:
        raise ValueError(f"effect_form must be 'attr' or 'ortho_quadratic'; got {effect_form!r}")

    Y = ((W * np.asarray(betas, float)).sum(axis=1) + tau * T
         + rng.normal(0.0, noise_sd, size=n)).astype(np.float64)
    return RCTSample(T=T, W=W.astype(np.float64), Z=Z, Y=Y, image_indices=image_idx)
