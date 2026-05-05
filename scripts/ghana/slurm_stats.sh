#!/bin/bash
#SBATCH --job-name=ghana_stats
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/NEXIS/logs/ghana_stats_%j.out
#SBATCH --error=/nfs/scistore19/locatgrp/rcadei/NEXIS/logs/ghana_stats_%j.err
#SBATCH --partition=debug_cpu
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -euo pipefail

ROOT=/nfs/scistore19/locatgrp/rcadei/NEXIS
PYTHON=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3

mkdir -p "$ROOT/logs"
cd "$ROOT"

$PYTHON src/apps/ghana/interpret.py \
    --mode both \
    --alpha 0.05 \
    --no-interpret \
    "$@"
