#!/bin/bash
#SBATCH --job-name=ghana_interp7b
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/NEXIS/logs/ghana_interpret_%j.out
#SBATCH --error=/nfs/scistore19/locatgrp/rcadei/NEXIS/logs/ghana_interpret_%j.err
#SBATCH --partition=debug_gpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6

set -euo pipefail

ROOT=/nfs/scistore19/locatgrp/rcadei/NEXIS
PYTHON=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3

mkdir -p "$ROOT/logs"
cd "$ROOT"

$PYTHON src/apps/ghana/interpret.py \
    --mode both \
    --vlm-model Qwen/Qwen2-VL-7B-Instruct \
    --quantize \
    --pipeline qwen7b \
    --k 8 \
    --alpha 0.05 \
    --interpret-only \
    "$@"
