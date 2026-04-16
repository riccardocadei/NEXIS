#!/usr/bin/env bash
#SBATCH --job-name=celeba-embed
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/NEMS/logs/celeba-embed-%j.out
#SBATCH --error=/nfs/scistore19/locatgrp/rcadei/NEMS/logs/celeba-embed-%j.err
#SBATCH --partition=debug_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# Stage 1: Download CelebA (valid split, 19,867 images) and extract SigLIP-base embeddings.
# Saves mean-pooled embeddings (siglip.npy, ~58 MB) AND per-patch features
# (siglip_patches.npy, ~6 GB float16) for SAE patch-training.
# Estimated runtime: ~25-50 min on GPU.
#
# Submit:  sbatch scripts/celeba/submit_embed.sh
# Or run:  bash   scripts/celeba/submit_embed.sh

set -euo pipefail

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
echo "============================================================"

$PYTHON src/apps/celeba/embed.py \
    --data-dir    data/celeba \
    --split       valid       \
    --batch-size  128         \
    --save-patches            \
    --force

echo "Stage 1 complete."
