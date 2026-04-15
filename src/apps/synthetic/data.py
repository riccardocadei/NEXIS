
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np


@dataclass
class SyntheticData:
    U: np.ndarray              # (n, 2) true latent effect modifiers
    T: np.ndarray              # (n,) binary treatment
    Z: np.ndarray              # (n, m) embedding neurons
    Y: np.ndarray              # (n,) factual outcome (continuous or binary)
    Y0_mean: np.ndarray        # (n,) conditional mean under control
    Y1_mean: np.ndarray        # (n,) conditional mean under treatment
    A: np.ndarray              # (m, 2) loadings in Z = A U + eps (row-wise)
    principal_idx: Tuple[int, int]
    meta: Dict[str, float]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_loading_matrix(
    m: int = 500,
    seed: Optional[int] = None,
    principal_loading: float = 3.0,
    leakage_scale: float = 0.22,
    delta: float = 1.00,
    cross_loading_principal: float = 0.05,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Create A \in R^{m x 2} with principal coordinates j1=0, j2=1.
    Enforces an A3-style separation margin on each latent component.
    """
    if m < 2:
        raise ValueError("m must be >= 2")

    rng = np.random.default_rng(seed)
    A = rng.normal(loc=0.0, scale=leakage_scale, size=(m, 2))

    # Principal rows
    A[0, 0] = principal_loading
    A[0, 1] = rng.normal(0.0, cross_loading_principal)
    A[1, 1] = principal_loading
    A[1, 0] = rng.normal(0.0, cross_loading_principal)

    # Enforce separation margins
    lim1 = abs(A[0, 0]) - delta
    lim2 = abs(A[1, 1]) - delta
    if lim1 <= 0 or lim2 <= 0:
        raise ValueError("principal_loading must exceed delta")

    for j in range(1, m):
        if abs(A[j, 0]) > lim1:
            A[j, 0] = np.sign(A[j, 0]) * (0.98 * lim1)
    for j in [0] + list(range(2, m)):
        if abs(A[j, 1]) > lim2:
            A[j, 1] = np.sign(A[j, 1]) * (0.98 * lim2)

    # Final assert (tolerant)
    max_other_0 = np.max(np.abs(A[1:, 0]))
    max_other_1 = np.max(np.abs(np.concatenate([A[:1, 1], A[2:, 1]])))
    if not (abs(A[0, 0]) >= max_other_0 + delta - 1e-8):
        raise RuntimeError("Failed to enforce principal alignment for U1")
    if not (abs(A[1, 1]) >= max_other_1 + delta - 1e-8):
        raise RuntimeError("Failed to enforce principal alignment for U2")

    return A, (0, 1)


def generate_synthetic_rct(
    n: int,
    m: int = 500,
    effect_scale: float = 1.0,
    interaction_form: str = "linear",   # "linear" or "quadratic"
    outcome_family: str = "gaussian",   # "gaussian" or "binary"
    p_treat: float = 0.5,
    seed: Optional[int] = None,
    z_noise_sd: float = 0.7,
    y_noise_sd: float = 1.0,
    baseline_intercept: float = 0.0,
    baseline_u_coef: Tuple[float, float] = (0.7, -0.5),
    ate_main: float = 0.4,
    hetero_coef_linear: Tuple[float, float] = (1.0, -1.0),
    hetero_coef_quadratic: Tuple[float, float] = (0.8, -0.8),
    standardize_z: bool = True,
    fixed_A: Optional[np.ndarray] = None,
    fixed_principal_idx: Tuple[int, int] = (0, 1),
) -> SyntheticData:
    """
    DGP:
      U ~ N(0, I_2)
      T ~ Bernoulli(p_treat)
      Z = A U + eps
      tau(U) = ate_main + effect_scale * h(U)
      Gaussian outcome:
          Y = mu0(U) + T * tau(U) + noise
      or Binary outcome:
          Y | T,U ~ Bernoulli(sigmoid(mu0(U) + T * tau(U)))

    h(U) depends on both U1 and U2 via linear / quadratic terms.
    """
    rng = np.random.default_rng(seed)

    U = rng.normal(size=(n, 2))
    T = rng.binomial(1, p_treat, size=n).astype(float)

    if fixed_A is None:
        A, principal_idx = make_loading_matrix(m=m, seed=seed)
    else:
        A = np.asarray(fixed_A, dtype=float)
        if A.shape != (m, 2):
            raise ValueError(f"fixed_A must have shape {(m,2)}, got {A.shape}")
        principal_idx = fixed_principal_idx

    eps = rng.normal(loc=0.0, scale=z_noise_sd, size=(n, m))
    Z = U @ A.T + eps

    if standardize_z:
        Z = (Z - Z.mean(axis=0, keepdims=True)) / (Z.std(axis=0, keepdims=True) + 1e-8)

    mu0 = (
        baseline_intercept
        + baseline_u_coef[0] * U[:, 0]
        + baseline_u_coef[1] * U[:, 1]
    )

    h_lin = hetero_coef_linear[0] * U[:, 0] + hetero_coef_linear[1] * U[:, 1]
    if interaction_form == "linear":
        h = h_lin
    elif interaction_form == "quadratic":
        h_quad = (
            hetero_coef_quadratic[0] * (U[:, 0] ** 2 - 1.0)
            + hetero_coef_quadratic[1] * (U[:, 1] ** 2 - 1.0)
        )
        h = h_lin + h_quad
    else:
        raise ValueError("interaction_form must be 'linear' or 'quadratic'")

    tau = ate_main + effect_scale * h
    mu1 = mu0 + tau

    if outcome_family == "gaussian":
        noise = rng.normal(loc=0.0, scale=y_noise_sd, size=n)
        Y = mu0 + T * tau + noise
    elif outcome_family == "binary":
        p = sigmoid(mu0 + T * tau)
        Y = rng.binomial(1, p, size=n).astype(float)
    else:
        raise ValueError("outcome_family must be 'gaussian' or 'binary'")

    meta = {
        "n": float(n),
        "m": float(m),
        "effect_scale": float(effect_scale),
        "p_treat": float(p_treat),
        "z_noise_sd": float(z_noise_sd),
        "y_noise_sd": float(y_noise_sd),
        "outcome_family_gaussian": 1.0 if outcome_family == "gaussian" else 0.0,
        "interaction_form_linear": 1.0 if interaction_form == "linear" else 0.0,
    }

    return SyntheticData(
        U=U,
        T=T,
        Z=Z,
        Y=Y.astype(float),
        Y0_mean=mu0.astype(float),
        Y1_mean=mu1.astype(float),
        A=A,
        principal_idx=principal_idx,
        meta=meta,
    )
