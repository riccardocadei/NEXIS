#!/bin/bash
set -euo pipefail

ROOT=/nfs/scistore19/locatgrp/rcadei/NEXIS
PYTHON=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3

mkdir -p "$ROOT/logs"
cd "$ROOT"

$PYTHON src/apps/ghana/interpret.py \
    --mode codes \
    --min-activations 5 \
    --method nexis_fdr \
    --interpret-only \
    --overwrite \
    --neurons 3821,2095,3318 \
    --vlm-model Qwen/Qwen2.5-VL-72B-Instruct \
    --pipeline qwen25_72b \
    --quantize \
    --k 12 \
    "$@"
