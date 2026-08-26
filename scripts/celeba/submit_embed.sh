#!/usr/bin/env bash
#SBATCH --job-name=celeba-embed
#SBATCH --output=logs/celeba-embed-%j.out
#SBATCH --error=logs/celeba-embed-%j.err
#SBATCH --partition=debug_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# Stage 1: Download CelebA (valid split, 19,867 images) and extract frozen-backbone embeddings.
# Saves mean-pooled embeddings ({backbone}.npy, ~58 MB) AND per-patch features
# ({backbone}_patches.npy, ~6 GB for SigLIP / ~7.8 GB for DINOv2) for SAE patch-training.
# Estimated runtime: ~25-50 min on GPU.
#
# Usage:   [sbatch|bash] scripts/celeba/submit_embed.sh [siglip|dinov2]
# Submit:  sbatch scripts/celeba/submit_embed.sh
#          sbatch scripts/celeba/submit_embed.sh dinov2
# Or run:  bash   scripts/celeba/submit_embed.sh

set -euo pipefail

BACKBONE="${1:-siglip}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

PYTHON="${PYTHON:-/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "============================================================"
echo " CelebA Stage 1: embed_celeba"
echo " Project root : $PROJECT_ROOT"
echo " Python       : $PYTHON"
echo " GPU          : $CUDA_VISIBLE_DEVICES"
echo " Backbone     : $BACKBONE"
echo "============================================================"

# No --force: existing artefacts are skipped, and labels.parquet (shared by every
# backbone, and the row-alignment reference for the SAE features) is left untouched.
# Pass --force explicitly to embed.py to recompute from scratch.
$PYTHON src/apps/celeba/embed.py \
    --data-dir    data/celeba \
    --backbone    "$BACKBONE" \
    --split       valid       \
    --batch-size  128         \
    --save-patches

echo "Stage 1 complete."
