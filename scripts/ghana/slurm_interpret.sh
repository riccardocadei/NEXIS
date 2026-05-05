#!/bin/bash
#SBATCH --job-name=ghana_interpret
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/NEXIS/logs/ghana_interpret_%j.out
#SBATCH --error=/nfs/scistore19/locatgrp/rcadei/NEXIS/logs/ghana_interpret_%j.err
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

set -euo pipefail

ROOT=/nfs/scistore19/locatgrp/rcadei/NEXIS
PYTHON=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3

mkdir -p "$ROOT/logs"
cd "$ROOT"

$PYTHON src/apps/ghana/interpret.py \
    --mode both \
    --vlm-model Qwen/Qwen2-VL-72B-Instruct \
    --quantize \
    --pipeline qwen72b \
    --k 8 \
    --alpha 0.05 \
    --interpret-only \
    "$@"
