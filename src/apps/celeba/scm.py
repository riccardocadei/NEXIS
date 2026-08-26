"""
CelebA semi-synthetic SCM for effect-modifier discovery.

DGP
---
  T   ~ Bernoulli(p_treat)
  W_k ~ Bernoulli(p_k), k = 1..r  — drawn from CelebA empirical prevalence
  X   ~ CelebA image matching (W_1, …, W_r)   [sampled without replacement]
  Z   = SAE(ViT(X))                           [pre-computed SAE features]
  Y   = sum_k beta_k*W_k
        + T * [tau_0 + effect_scale * sum_k gamma_k*W_k]
        + noise

Default attributes (r = 2, the published benchmark)
  W1 = "Wearing_Hat"  (prevalence ≈ 5%)   — positive T×W1 modification
  W2 = "Eyeglasses"   (prevalence ≈ 7%)   — negative T×W2 modification

Larger truth sets (r > 2) are requested with `w_attrs`.  The attribute must be
carried by ONE dominant SAE coordinate for S* to be well defined, and the joint
cell (W_1=1, …, W_r=1) must contain enough CelebA images for the sampler not to
exhaust it.  Both are properties of the attribute set, not of the code, and only
a few CelebA attributes satisfy both alongside Wearing_Hat and Eyeglasses.
"Sideburns" is the one clean r=3 completion: best-threshold F1 0.697 on a single
coordinate whose runner-up is 0.244 lower, and a binding cell of 698 images that
supports the full grid to n = 17,028.  Blond_Hair and Bangs are equally well
aligned but co-occur with hats and glasses in ONE CelebA image, so they cannot
be used with this sampler.

The ground truth for NEXIS evaluation is the set of SAE neurons whose
activations are most correlated with W1 / W2 across the full CelebA set.
NEXIS should recover these through significant T×Z_j interactions with Y.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

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
    W: Optional[np.ndarray] = None   # (n, r) float64, all r modifiers in order
    w_attrs: Optional[List[str]] = None   # attribute names, aligned with W columns


# ---------------------------------------------------------------------------
# Bucket construction
# ---------------------------------------------------------------------------

def resolve_w_attrs(w_attrs=None, w1_attr=None, w2_attr=None) -> List[str]:
    """Normalise the two ways of naming the modifier attributes into one list."""
    if w_attrs:
        return list(w_attrs)
    return [w1_attr or "Wearing_Hat", w2_attr or "Eyeglasses"]


def build_buckets(
    labels_df: pd.DataFrame,
    w1_attr: Optional[str] = None,
    w2_attr: Optional[str] = None,
    w_attrs: Optional[Sequence[str]] = None,
) -> Dict[Tuple[int, ...], List[int]]:
    """
    Pre-stratify CelebA images by the joint value of the r modifier attributes.

    Returns a dict mapping (w_1, …, w_r) ∈ {0,1}^r → sorted list of row indices
    in labels_df with that attribute combination.  All 2^r cells are present as
    keys; cells with no matching image map to an empty list, which
    generate_celeba_rct reports as an exhausted bucket rather than a KeyError.

    Accepts either the legacy (w1_attr, w2_attr) pair or an explicit w_attrs list.
    """
    attrs = resolve_w_attrs(w_attrs, w1_attr, w2_attr)
    cols = np.stack([labels_df[a].values.astype(int) for a in attrs], axis=1)
    buckets: Dict[Tuple[int, ...], List[int]] = {
        key: [] for key in product((0, 1), repeat=len(attrs))
    }
    for i, row in enumerate(cols):
        buckets[tuple(int(v) for v in row)].append(i)
    return buckets


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _ortho_quadratic_map(col: np.ndarray):
    """Population map g(z) = residual of z̃² on [1, z̃], rescaled to unit variance.

    Built from the full CelebA column so that, in the population,

        E[g(Z^j)] = 0    and    Cov(g(Z^j), Z^j) = 0    exactly.

    A treatment effect driven by g is therefore invisible to any test that looks at
    the covariance between the effect and a *linear* function of Z^j — the GCM as
    instantiated here, and a fortiori the linear interaction t-test — while the
    conditional mean E[τ | Z^j] genuinely varies.  This is the alternative class the
    PCM is built for.
    """
    z = np.asarray(col, dtype=float)
    mu, sd = z.mean(), z.std()
    sd = sd if sd > 1e-12 else 1.0
    zt = (z - mu) / sd
    q = zt ** 2
    # residual of q on [1, zt]
    b = float(np.cov(q, zt, ddof=0)[0, 1] / max(np.var(zt), 1e-12))
    a = float(q.mean() - b * zt.mean())
    g_pop = q - a - b * zt
    gsd = float(g_pop.std())
    gsd = gsd if gsd > 1e-12 else 1.0

    def g(x: np.ndarray) -> np.ndarray:
        xt = (np.asarray(x, dtype=float) - mu) / sd
        return (xt ** 2 - a - b * xt) / gsd

    return g


def generate_celeba_rct(
    n: int,
    features: np.ndarray,
    labels_df: pd.DataFrame,
    buckets: Dict[Tuple[int, ...], List[int]],
    w1_attr: str = "Wearing_Hat",
    w2_attr: str = "Eyeglasses",
    w_attrs: Optional[Sequence[str]] = None,
    betas: Optional[Sequence[float]] = None,
    gammas: Optional[Sequence[float]] = None,
    p_treat: float = 0.5,
    tau_0: float = 0.5,
    beta_w1: float = 0.3,
    beta_w2: float = -0.2,
    gamma_w1: float = 1.0,
    gamma_w2: float = -1.0,
    noise_sd: float = 1.0,
    effect_scale: float = 1.0,
    effect_form: str = "attr",
    modifier_cols: Optional[List[int]] = None,
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
        w1_attr:      Column name of W1 attribute in labels_df (r = 2 form).
        w2_attr:      Column name of W2 attribute in labels_df (r = 2 form).
        w_attrs:      Explicit list of r attribute names; overrides w1_attr/w2_attr.
        betas:        Length-r main effects; defaults to [beta_w1, beta_w2] for
                      r = 2 and to an alternating +0.3 / -0.2 pattern beyond.
        gammas:       Length-r interaction coefficients; defaults to
                      [gamma_w1, gamma_w2] for r = 2 and to alternating +1 / -1
                      beyond, so successive modifiers push the effect in opposite
                      directions and no coordinate is a scalar multiple of another.
        p_treat:      Treatment probability.
        tau_0:        Main ATE (homogeneous part).
        beta_w1/w2:   Main effects of W1/W2 on Y.
        gamma_w1/w2:  T×W1 / T×W2 interaction coefficients (effect modification).
        noise_sd:     Gaussian noise standard deviation.
        effect_scale: Multiplier for gamma terms (sweep this to vary effect size).
        effect_form:  "attr" (default) — τ modified by the binary attributes W1, W2.
                      "ortho_quadratic" — τ modified by g(Z^{j1}), g(Z^{j2}) on the
                      two ground-truth coordinates, with g the population map of
                      _ortho_quadratic_map (zero mean, zero covariance with Z^j).
                      Requires modifier_cols and only makes sense on continuous
                      features (SAE pre-activations or raw embeddings): on sparse
                      post-topk codes any function of Z^j is effectively two-valued,
                      hence monotone, and the U-shape degenerates.
        modifier_cols: [j1, j2] column indices driving τ when effect_form != "attr".
        seed:         RNG seed for reproducibility.

    Returns:
        CelebAData with T, W1, W2, Z, Y, image_indices.

    Raises:
        ValueError if any (W1, W2) bucket runs out of images for the requested n.
    """
    rng = np.random.default_rng(seed)

    attrs = resolve_w_attrs(w_attrs, w1_attr, w2_attr)
    r = len(attrs)
    if betas is None:
        betas = [beta_w1, beta_w2] if r == 2 else [
            (0.3 if k % 2 == 0 else -0.2) for k in range(r)]
    if gammas is None:
        gammas = [gamma_w1, gamma_w2] if r == 2 else [
            (1.0 if k % 2 == 0 else -1.0) for k in range(r)]
    if len(betas) != r or len(gammas) != r:
        raise ValueError(f"betas/gammas must have length r={r}; "
                         f"got {len(betas)}/{len(gammas)}")
    key_len = len(next(iter(buckets)))
    if key_len != r:
        raise ValueError(f"buckets are keyed on {key_len} attributes but {r} were "
                         f"requested — rebuild them with build_buckets(w_attrs=…)")

    # Attributes are drawn INDEPENDENTLY at their marginal CelebA prevalence, so
    # the joint cell probabilities are products.  The image bucket for a rare
    # joint cell can be far smaller than n * prod(p), which is what limits r.
    p_w = [float(labels_df[a].mean()) for a in attrs]
    W_mat = np.stack([rng.binomial(1, p, size=n) for p in p_w],
                     axis=1).astype(np.int32)          # (n, r)
    W1, W2 = W_mat[:, 0], (W_mat[:, 1] if r > 1 else np.zeros(n, np.int32))
    T  = rng.binomial(1, p_treat, size=n).astype(np.float64)

    # Shuffle each bucket once, then draw sequentially (= without replacement)
    bucket_perms: Dict[Tuple[int, int], np.ndarray] = {
        k: rng.permutation(v) for k, v in buckets.items()
    }
    bucket_ptrs: Dict[Tuple[int, int], int] = {k: 0 for k in buckets}

    image_idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        key = tuple(int(v) for v in W_mat[i])
        ptr = bucket_ptrs[key]
        perm = bucket_perms[key]
        if ptr >= len(perm):
            cell = ", ".join(f"{a}={v}" for a, v in zip(attrs, key))
            raise ValueError(
                f"Bucket ({cell}) exhausted: size={len(perm)}, needed >{ptr} images. "
                f"Reduce n, or choose attributes whose joint cells are better "
                f"populated — with r={r} the binding cell is usually all-ones."
            )
        image_idx[i] = perm[ptr]
        bucket_ptrs[key] += 1

    Z = features[image_idx].astype(np.float64)  # (n, hidden_dim)

    if effect_form == "attr":
        tau = tau_0 + effect_scale * (W_mat * np.asarray(gammas, float)).sum(axis=1)
    elif effect_form == "ortho_quadratic":
        if modifier_cols is None or len(modifier_cols) != 2:
            raise ValueError("effect_form='ortho_quadratic' requires modifier_cols=[j1, j2]")
        j1, j2 = int(modifier_cols[0]), int(modifier_cols[1])
        g1 = _ortho_quadratic_map(features[:, j1])
        g2 = _ortho_quadratic_map(features[:, j2])
        tau = tau_0 + effect_scale * (gamma_w1 * g1(Z[:, j1]) + gamma_w2 * g2(Z[:, j2]))
    else:
        raise ValueError(f"effect_form must be 'attr' or 'ortho_quadratic'; got {effect_form!r}")

    Y = (
        (W_mat * np.asarray(betas, float)).sum(axis=1)
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
        W=W_mat.astype(np.float64),
        w_attrs=attrs,
    )
