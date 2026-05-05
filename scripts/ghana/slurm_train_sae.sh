#!/bin/bash
#SBATCH --job-name=ghana_sae
#SBATCH --partition=gpu100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ghana_sae_%j.out
#SBATCH --error=logs/ghana_sae_%j.err

export PYTHONUNBUFFERED=1

source /nfs/scistore19/locatgrp/rcadei/miniconda3/etc/profile.d/conda.sh
conda activate crl

echo "=== env ready: $(which python3) ==="

cd /nfs/scistore19/locatgrp/rcadei/NEXIS

python3 -u scripts/ghana/train_sae.py \
  --train-embeddings ../../data/ghana/satellite/national/prithvi_embeddings.npy \
  --eval-embeddings  ../../data/ghana/satellite/prithvi_embeddings.npy \
  --eval-ids         ../../data/ghana/satellite/prithvi_comm_ids.npy \
  --d-hidden 4096 --k 25 \
  --epochs 2000 --batch-size 256
