#!/usr/bin/env bash
#SBATCH --job-name=celeba-concept
#SBATCH --output=logs/celeba-concept-%j.out
#SBATCH --error=logs/celeba-concept-%j.err
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=40
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#
# Concept-level agreement between the paper's SAEs and the resampled-corpus SAEs:
# re-runs the main design point on both dictionaries recording the selected sets, then
# matches recovered features across dictionaries by top-activating examples and compares
# their CATE profiles.  See src/apps/celeba/concept_agreement.py.
#
# Usage: sbatch scripts/celeba/submit_concept_agreement.sh [K] [TAG] [N] [EFFECT]

set -euo pipefail

K="${1:-20}"; TAG="${2:-b1}"; N="${3:-2000}"; EFFECT="${4:-5.0}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-8}"

$PYTHON src/apps/celeba/concept_agreement.py \
    --tag "$TAG" --sae-top-k "$K" --n "$N" --effect "$EFFECT" \
    --n-seeds 50 --top-m 100 --contact-sheets

echo "concept agreement (k=${K}, tag=${TAG}) complete."
