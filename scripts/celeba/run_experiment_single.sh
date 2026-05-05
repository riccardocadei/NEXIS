#!/usr/bin/env bash
#SBATCH --job-name=celeba-exp
#SBATCH --output=logs/celeba-exp-%j.out
#SBATCH --error=logs/celeba-exp-%j.err
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# Single-feature-type experiment worker. Called by submit_experiment.sh.
# Usage: sbatch run_experiment_single.sh [raw|sae|sae_precode] [k] [effect|n|both]

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
    --force
)

FAST_METHODS=(
    "Marginal Testing"
    "Marginal Testing (FWER)"
    "Marginal Testing (FDR)"
    "NEXIS"
    "NEXIS (test=GCM: quadratic)"
    "NEXIS (adjust=None)"
    "NEXIS (adjust=FDR)"
    "NEXIS (rho=0)"
    "NEXIS (rho=0.2)"
    "NEXIS (rho=0.8)"
    "NEXIS (backward=False)"
)

case "$FEATURE_TYPE" in
    raw)
        $PYTHON src/apps/celeba/run_experiment.py --raw \
            --methods "${FAST_METHODS[@]}" "${COMMON_ARGS[@]}"
        ;;
    sae)
        $PYTHON src/apps/celeba/run_experiment.py \
            --sae-top-k "${SAE_K}" --methods "${FAST_METHODS[@]}" "${COMMON_ARGS[@]}"
        ;;
    sae_precode)
        $PYTHON src/apps/celeba/run_experiment.py \
            --precode --sae-top-k "${SAE_K}" --methods "${FAST_METHODS[@]}" "${COMMON_ARGS[@]}"
        ;;
    *) echo "Unknown feature type: $FEATURE_TYPE" >&2; exit 1 ;;
esac
