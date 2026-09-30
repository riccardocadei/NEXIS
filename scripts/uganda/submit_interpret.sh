#!/bin/bash
#SBATCH --job-name=uganda_interpret
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/NEXIS/logs/uganda_interpret_%j.out
#SBATCH --error=/nfs/scistore19/locatgrp/rcadei/NEXIS/logs/uganda_interpret_%j.err
#SBATCH --partition=gpu100
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

set -euo pipefail

ROOT=/nfs/scistore19/locatgrp/rcadei/NEXIS
PYTHON=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3

mkdir -p "$ROOT/logs"
cd "$ROOT"

$PYTHON src/apps/uganda/interpret.py \
    --embed-model prithvi_l5 \
    --sae-dim 1024 \
    --outcomes skilled_employed,log_biz_assets \
    --method nexis_fdr \
    --vlm-model Qwen/Qwen2.5-VL-72B-Instruct \
    --quantize \
    --pipeline qwen72b \
    --k 12 \
    --overwrite \
    "$@"
