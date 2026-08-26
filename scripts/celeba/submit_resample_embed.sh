#!/usr/bin/env bash
#SBATCH --job-name=celeba-rs-embed
#SBATCH --output=logs/celeba-rs-embed-%j.out
#SBATCH --error=logs/celeba-rs-embed-%j.err
#SBATCH --partition=gpu100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#
# SAE-resampling robustness check, stage 1: embed a *different* corpus of exactly the
# same size as the one the paper's SAEs were trained on (19,867 images), drawn from the
# CelebA train split — identity-disjoint from the valid split used in every experiment.
#
# Writes:  data/celeba_resample_<TAG>/embeddings/siglip{,_patches}.npy
#          data/celeba_resample_<TAG>/labels.parquet   (of the training corpus; unused downstream)
#
# The experiment data itself is NOT touched: the SAEs trained on this corpus are later
# used to encode the original valid-split embeddings (see submit_resample_sae.sh).
#
# Usage:  sbatch scripts/celeba/submit_resample_embed.sh [TAG] [SAMPLE_SEED] [N] [BACKBONE]

set -euo pipefail

TAG="${1:-b1}"
SAMPLE_SEED="${2:-1}"
SAMPLE_N="${3:-19867}"
BACKBONE="${4:-siglip}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

PYTHON="${PYTHON:-/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Arrow cache for flwrlabs/celeba (all splits) is already built locally.
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "============================================================"
echo " SAE resample ${TAG}: embed ${SAMPLE_N} train-split images"
echo " sample seed  : ${SAMPLE_SEED}   backbone: ${BACKBONE}"
echo " out          : data/celeba_resample_${TAG}"
echo "============================================================"

$PYTHON src/apps/celeba/embed.py \
    --data-dir     "data/celeba_resample_${TAG}" \
    --backbone     "$BACKBONE" \
    --split        train \
    --sample-n     "$SAMPLE_N" \
    --sample-seed  "$SAMPLE_SEED" \
    --batch-size   256 \
    --save-patches

echo "Stage 1 (resample ${TAG}) complete."
