#!/usr/bin/env bash
#SBATCH --job-name=uganda-sae
#SBATCH --output=logs/slurm-sae-%j.out
#SBATCH --error=logs/slurm-sae-%j.err
#SBATCH --partition=debug_gpu
#SBATCH --gres=gpu:2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
mkdir -p logs

PYTHON=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3

echo "=== Uganda SAE training (prithvi_l5_1024) ==="
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo ""

ROOT=/nfs/scistore19/locatgrp/rcadei/NEXIS

$PYTHON scripts/uganda/train_sae.py \
  --train-embeddings $ROOT/data/uganda/satellite/national/prithvi_embeddings.npy \
  --rct-embeddings   $ROOT/data/uganda/satellite/rct/prithvi_embeddings.npy \
  --rct-keys         $ROOT/data/uganda/satellite/rct/prithvi_site_keys.npy \
  --data-csv         $ROOT/data/uganda/UgandaDataProcessed.csv \
  --d-hidden 1024 \
  --k 25 \
  --epochs 2000 \
  --lr 2e-4 \
  --cv-folds 5 \
  --device cuda

echo ""
echo "=== Done. Results in results/uganda/prithvi_l5_1024/ ==="
