#!/bin/bash
set -euo pipefail

ROOT=/nfs/scistore19/locatgrp/rcadei/NEXIS
PYTHON=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3

mkdir -p "$ROOT/logs"
cd "$ROOT"

$PYTHON scripts/ghana/interpret_temporal_waterways.py \
    --vlm-model Qwen/Qwen2.5-VL-72B-Instruct \
    --quantize \
    "$@"
