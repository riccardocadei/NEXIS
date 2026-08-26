#!/usr/bin/env bash
#SBATCH --job-name=celeba-pcm
#SBATCH --output=logs/celeba-pcm-%j.out
#SBATCH --error=logs/celeba-pcm-%j.err
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=40
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#
# PCM-only worker (modified Projected Covariance Measure, Lundborg et al. 2024).
# ~3-4x the cost of the GCM: the projection is fitted on one half-sample and
# evaluated on the other, in both directions.
# Usage: sbatch run_experiment_pcm.sh [raw|sae|sae_precode] [k] [effect|n|both] [backbone]

set -euo pipefail

FEATURE_TYPE="${1:-sae}"
SAE_K="${2:-20}"
SWEEP="${3:-both}"
BACKBONE="${4:-siglip}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3}"
export OMP_NUM_THREADS=1

COMMON_ARGS=(
    --data-dir     data/celeba
    --out-dir      results/celeba/experiment
    --backbone     "$BACKBONE"
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
    --methods      "NEXIS (test=PCM: quadratic)" "NEXIS (test=PCM: lgbm)"
    --merge
)

case "$FEATURE_TYPE" in
    raw)
        $PYTHON src/apps/celeba/run_experiment.py --raw "${COMMON_ARGS[@]}"
        ;;
    sae)
        $PYTHON src/apps/celeba/run_experiment.py \
            --sae-top-k "${SAE_K}" \
            --gt-json "results/celeba/experiment/k${SAE_K}/sae/ground_truth.json" \
            "${COMMON_ARGS[@]}"
        ;;
    sae_precode)
        $PYTHON src/apps/celeba/run_experiment.py \
            --precode --sae-top-k "${SAE_K}" \
            --gt-json "results/celeba/experiment/k${SAE_K}/sae_precode/ground_truth.json" \
            "${COMMON_ARGS[@]}"
        ;;
    *) echo "Unknown feature type: $FEATURE_TYPE" >&2; exit 1 ;;
esac
