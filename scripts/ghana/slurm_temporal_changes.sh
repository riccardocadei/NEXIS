#!/bin/bash
#SBATCH --job-name=ghana_temporal_neutral
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/NEXIS/logs/ghana_temporal_neutral_%j.out
#SBATCH --error=/nfs/scistore19/locatgrp/rcadei/NEXIS/logs/ghana_temporal_neutral_%j.err
#SBATCH --partition=gpu100
#SBATCH --gres=gpu:H100:1
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# Neutral-prompt 2015 -> 2017 change interpretation for the six communities where
# neuron 3821 is active -> results/ghana/temporal/neuron_3821_temporal_neutral.json

set -euo pipefail

ROOT=/nfs/scistore19/locatgrp/rcadei/NEXIS
PYTHON=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3

mkdir -p "$ROOT/logs"
cd "$ROOT"

$PYTHON src/apps/ghana/interpret_temporal_changes.py \
    --vlm-model Qwen/Qwen2.5-VL-72B-Instruct \
    --quantize \
    "$@"
