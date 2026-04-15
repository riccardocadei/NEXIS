"""Training package — backbone loading, SAE, and dataset utilities."""
from .backbone import (
    MODEL_REGISTRY, DINO_DIMS, model_embed_dim,
    ImageFolderFlat, load_model, extract_embeddings, embed_image_folder,
    _auto_device, _DINO_TRANSFORM, _SIGLIP_TRANSFORM,
)
from .satellite import (
    UgandaSatelliteDataset, compute_dataset_stats,
    make_uganda_transform, embed_uganda_sites,
)

from .sae import SAE, SAETrainConfig, SAETrainResult, train_sae, get_features

__all__ = [
    "MODEL_REGISTRY", "DINO_DIMS", "model_embed_dim",
    "ImageFolderFlat", "load_model", "extract_embeddings", "embed_image_folder",
    "_auto_device", "_DINO_TRANSFORM", "_SIGLIP_TRANSFORM",
    "UgandaSatelliteDataset", "compute_dataset_stats",
    "make_uganda_transform", "embed_uganda_sites",
    "SAE", "SAETrainConfig", "SAETrainResult", "train_sae", "get_features",
]
