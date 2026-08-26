"""Training package — SAE implementation used by the CelebA pipeline."""
from .sae import (
    SAE, SAETrainConfig, SAETrainResult, train_sae,
    TopKSAE, TopKSAETrainConfig, train_topk_sae,
    get_features, get_pre_features,
)

__all__ = [
    "SAE", "SAETrainConfig", "SAETrainResult", "train_sae",
    "TopKSAE", "TopKSAETrainConfig", "train_topk_sae",
    "get_features", "get_pre_features",
]
