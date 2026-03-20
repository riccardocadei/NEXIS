from .nems import (
    SelectionResult,
    nems_select,
    marginal_select,
    evaluate_methods_on_dataset,
    iou_score,
)
from .synthetic import SyntheticData, generate_synthetic_rct, make_loading_matrix
from .embeddings import (
    load_dinov2, extract_embeddings,
    ImageFolderFlat, embed_image_folder,
    UgandaSatelliteDataset, embed_uganda_sites,
    compute_dataset_stats, make_uganda_transform,
)
from .sae import SAE, SAETrainConfig, SAETrainResult, train_sae, get_features

__all__ = [
    # NEMS
    "SelectionResult",
    "nems_select",
    "marginal_select",
    "evaluate_methods_on_dataset",
    "iou_score",
    # Synthetic
    "SyntheticData",
    "generate_synthetic_rct",
    "make_loading_matrix",
    # Embeddings
    "load_dinov2",
    "extract_embeddings",
    "ImageFolderFlat",
    "embed_image_folder",
    "UgandaSatelliteDataset",
    "embed_uganda_sites",
    "compute_dataset_stats",
    "make_uganda_transform",
    # SAE
    "SAE",
    "SAETrainConfig",
    "SAETrainResult",
    "train_sae",
    "get_features",
]
