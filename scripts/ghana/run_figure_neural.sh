#!/bin/bash
set -euo pipefail

ROOT=/nfs/scistore19/locatgrp/rcadei/NEXIS
PYTHON=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3

mkdir -p "$ROOT/logs"
cd "$ROOT"

LOG="$ROOT/logs/figure_neural_$(date +%Y%m%d_%H%M%S).log"

echo "=== Step 1: Interpret 2015→2017 temporal changes ===" | tee -a "$LOG"
$PYTHON src/apps/ghana/interpret_temporal_changes.py \
    --vlm-model Qwen/Qwen2.5-VL-72B-Instruct \
    --quantize \
    --overwrite \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Step 2: Generate figures ===" | tee -a "$LOG"
$PYTHON src/apps/ghana/figure_neural.py      2>&1 | tee -a "$LOG"
$PYTHON src/apps/ghana/figure_neural_combined.py 2>&1 | tee -a "$LOG"
