#!/bin/bash
#SBATCH --job-name=ghana_sae
#SBATCH --partition=gpu100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ghana_sae_%j.out
#SBATCH --error=logs/ghana_sae_%j.err
#
# Train the Ghana TopK SAE on the national grid, evaluate on the 162 LEAP communities.
# Submit from the repo root. train_sae.py resolves relative paths against its own folder,
# so every path is passed absolute. The default OUT_DIR holds the SAE behind the paper's
# coordinate indices (not bit-reproducible): the job refuses to overwrite it unless
# OVERWRITE=1; set OUT_DIR to train elsewhere.

set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT=/nfs/scistore19/locatgrp/rcadei/NEXIS
PYTHON=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3
SAT=$ROOT/data/ghana/satellite
OUT_DIR=${OUT_DIR:-$SAT}

if [ -e "$OUT_DIR/sae_model.pt" ] && [ "${OVERWRITE:-0}" != 1 ]; then
  echo "$OUT_DIR/sae_model.pt exists; set OVERWRITE=1 or another OUT_DIR" >&2
  exit 1
fi

mkdir -p "$ROOT/logs"
cd "$ROOT"

$PYTHON -u src/apps/ghana/train_sae.py \
  --train-embeddings "$SAT/national/prithvi_embeddings.npy" \
  --eval-embeddings  "$SAT/prithvi_embeddings.npy" \
  --eval-ids         "$SAT/prithvi_comm_ids.npy" \
  --out-dir          "$OUT_DIR" \
  --d-hidden 4096 --k 25 \
  --epochs 2000 --batch-size 256
