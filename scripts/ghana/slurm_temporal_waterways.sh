#!/bin/bash
#SBATCH --job-name=ghana_temporal
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/NEXIS/logs/ghana_temporal_%j.out
#SBATCH --error=/nfs/scistore19/locatgrp/rcadei/NEXIS/logs/ghana_temporal_%j.err
#SBATCH --partition=gpu100
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

set -euo pipefail

ROOT=/nfs/scistore19/locatgrp/rcadei/NEXIS
PYTHON=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3

mkdir -p "$ROOT/logs"
cd "$ROOT"

$PYTHON scripts/ghana/interpret_temporal_waterways.py \
    --vlm-model Qwen/Qwen2.5-VL-72B-Instruct \
    --quantize \
    "$@"
