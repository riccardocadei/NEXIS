#!/usr/bin/env bash
#SBATCH --job-name=celeba-gcm
#SBATCH --output=logs/celeba-gcm-%j.out
#SBATCH --error=logs/celeba-gcm-%j.err
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# GCM-only worker (lgbm ~3× slower than linear).
# Usage: sbatch run_experiment_gcm.sh [raw|sae|sae_precode] [k] [effect|n|both]

set -euo pipefail

FEATURE_TYPE="${1:-sae}"
SAE_K="${2:-5}"
SWEEP="${3:-both}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3}"

COMMON_ARGS=(
    --data-dir     data/celeba
    --out-dir      results/celeba/experiment
    --w1-attr      Wearing_Hat
    --w2-attr      Eyeglasses
    --top-k        1
    --n-seeds      50
    --alpha        0.05
    --max-steps    10
    --fixed-n      500 2000
    --fixed-effect 2.0 5.0
    --gcm-splits   3
    --sweep        "$SWEEP"
    --methods      "NEXIS (test=GCM: lgbm)"
    --merge
)

case "$FEATURE_TYPE" in
    raw)
        $PYTHON src/apps/celeba/run_experiment.py --raw "${COMMON_ARGS[@]}"
        ;;
    sae)
        $PYTHON src/apps/celeba/run_experiment.py \
            --sae-top-k "${SAE_K}" "${COMMON_ARGS[@]}"
        ;;
    sae_precode)
        $PYTHON src/apps/celeba/run_experiment.py \
            --precode --sae-top-k "${SAE_K}" "${COMMON_ARGS[@]}"
        ;;
    *) echo "Unknown feature type: $FEATURE_TYPE" >&2; exit 1 ;;
esac
