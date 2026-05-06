#!/bin/bash
#SBATCH --job-name=ghana_interpret
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/NEXIS/logs/ghana_interpret_%j.out
#SBATCH --error=/nfs/scistore19/locatgrp/rcadei/NEXIS/logs/ghana_interpret_%j.err
#SBATCH --partition=gpu100
#SBATCH --time=00:45:00
#SBATCH --nodelist=gpu269
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

set -euo pipefail

ROOT=/nfs/scistore19/locatgrp/rcadei/NEXIS
PYTHON=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3

mkdir -p "$ROOT/logs"
cd "$ROOT"

$PYTHON src/apps/ghana/interpret.py \
    --mode codes \
    --method nexis_no_adj_hc1 \
    --vlm-model Qwen/Qwen2.5-VL-72B-Instruct \
    --quantize \
    --pipeline qwen25_72b \
    --k 12 \
    --interpret-only \
    --overwrite \
    --neurons 1715,1661,1424,3976 \
    "$@"
