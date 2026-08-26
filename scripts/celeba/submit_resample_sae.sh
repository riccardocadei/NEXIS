#!/usr/bin/env bash
#SBATCH --job-name=celeba-rs-sae
#SBATCH --output=logs/celeba-rs-sae-%j.out
#SBATCH --error=logs/celeba-rs-sae-%j.err
#SBATCH --partition=gpu100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# SAE-resampling robustness check, stage 2: train ONE TopK SAE on the resampled corpus
# with *exactly* the hyperparameters of the paper's SigLIP SAEs (hidden 9216, 20 epochs,
# batch 20 images, lr 5e-4), then encode the ORIGINAL valid-split embeddings with it.
#
#   training data  : data/celeba_resample_<TAG>/embeddings/siglip_patches.npy   (different sample)
#   encoding data  : data/celeba_resample_<TAG>/eval/embeddings/siglip.npy      (symlink → the
#                    original data/celeba embeddings, so the experiment data is unchanged)
#   checkpoint     : results/celeba/resample_<TAG>/sae_siglip_k<K>.pt
#   codes          : data/celeba_resample_<TAG>/eval/embeddings/sae_k<K>.npy, sae_precode_k<K>.npy
#
# One job per k so both k values train concurrently on separate H100s.
#
# Usage:  sbatch scripts/celeba/submit_resample_sae.sh <K> [TAG] [BACKBONE]

set -euo pipefail

K="${1:-5}"
TAG="${2:-b1}"
BACKBONE="${3:-siglip}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

PYTHON="${PYTHON:-/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA="data/celeba_resample_${TAG}"

echo "============================================================"
echo " SAE resample ${TAG}: train_sae  backbone=${BACKBONE}  top-k=${K}"
echo " train on : ${DATA}/embeddings/${BACKBONE}_patches.npy"
echo " encode   : ${DATA}/eval/embeddings/${BACKBONE}.npy  (= original valid split)"
echo "============================================================"

$PYTHON src/apps/celeba/train_sae.py \
    --data-dir       "${DATA}/eval" \
    --train-data-dir "${DATA}" \
    --backbone       "$BACKBONE" \
    --out-dir        "results/celeba/resample_${TAG}" \
    --hidden-dim     9216 \
    --epochs         20 \
    --batch-size     20 \
    --lr             5e-4 \
    --top-k          "$K" \
    --force

echo "Stage 2 (resample ${TAG}, k=${K}) complete."
