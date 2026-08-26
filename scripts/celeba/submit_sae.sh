#!/usr/bin/env bash
#SBATCH --job-name=celeba-sae
#SBATCH --output=logs/celeba-sae-%j.out
#SBATCH --error=logs/celeba-sae-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#
# Stage 2: Train SAE on frozen CelebA backbone embeddings and encode all images.
# Trains two SAEs sequentially: top-k=5 and top-k=20.
# Checkpoints/features are named sae_{backbone}_k{K}.pt / sae[_{backbone}]_k{K}.npy /
# sae_precode[_{backbone}]_k{K}.npy  (the tag is dropped for siglip: legacy names).
# Estimated runtime: ~2-3 h per SAE on GPU (~4-6 h total).
#
# Usage:   [sbatch|bash] scripts/celeba/submit_sae.sh [siglip|dinov2]
# Submit:  sbatch scripts/celeba/submit_sae.sh
#          sbatch scripts/celeba/submit_sae.sh dinov2
# Or run:  bash   scripts/celeba/submit_sae.sh

set -euo pipefail

BACKBONE="${1:-siglip}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

PYTHON="${PYTHON:-/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SAE_COMMON=(
    --data-dir   data/celeba
    --backbone   "$BACKBONE"
    --out-dir    results/celeba
    --hidden-dim 9216
    --epochs     20
    --batch-size 20
    --lr         5e-4
    --force
)

for K in 5 20; do
    echo "============================================================"
    echo " CelebA Stage 2: train_sae  backbone=${BACKBONE}  top-k=${K}"
    echo "============================================================"

    $PYTHON src/apps/celeba/train_sae.py \
        "${SAE_COMMON[@]}" \
        --top-k "${K}"

    echo "Stage 2 (k=${K}) complete."
done
