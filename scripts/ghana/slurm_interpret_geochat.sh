#!/bin/bash
#SBATCH --job-name=ghana_geochat
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/NEXIS/logs/ghana_geochat_%j.out
#SBATCH --error=/nfs/scistore19/locatgrp/rcadei/NEXIS/logs/ghana_geochat_%j.err
#SBATCH --partition=debug_gpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# GeoChat-7B (per-image RS descriptions) + Qwen2.5-7B-Instruct (contrast synthesis).
# Both models run in 4-bit; total VRAM ~8 GB.  10 neurons × 24 images ≈ 60–90 min.

set -euo pipefail

ROOT=/nfs/scistore19/locatgrp/rcadei/NEXIS
PYTHON=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3

mkdir -p "$ROOT/logs"
cd "$ROOT"

$PYTHON src/apps/ghana/interpret.py \
    --mode both \
    --pipeline geochat_llm \
    --geochat-model MBZUAI/geochat-7B \
    --synthesis-model Qwen/Qwen2.5-7B-Instruct \
    --quantize \
    --k 12 \
    --alpha 0.05 \
    --interpret-only \
    --overwrite \
    --neurons 1777,3821 \
    "$@"
