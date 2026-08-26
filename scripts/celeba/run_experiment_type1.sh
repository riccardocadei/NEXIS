#!/usr/bin/env bash
#SBATCH --job-name=celeba-type1
#SBATCH --output=logs/celeba-type1-%j.out
#SBATCH --error=logs/celeba-type1-%j.err
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=40
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#
# Type-I error control check for the CATE-equivalence tests: effect_scale = 0, so
# S* carries no heterogeneity and ANY selection is a false discovery.  The empirical
# FWER is the fraction of runs returning a non-empty set; NEXIS gates at alpha=0.05.
#
# Usage: sbatch run_experiment_type1.sh [sae|sae_precode]

set -euo pipefail
FEAT="${1:-sae}"
SAE_K=20
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3}"
export OMP_NUM_THREADS=1

OUT_DIR="results/celeba/experiment_type1"
PRECODE=""
[[ "$FEAT" == "sae_precode" ]] && PRECODE="--precode"

$PYTHON src/apps/celeba/run_experiment.py \
    $PRECODE --sae-top-k "$SAE_K" \
    --data-dir data/celeba --out-dir "$OUT_DIR" --backbone siglip \
    --gt-json "${OUT_DIR}/k${SAE_K}/${FEAT}/ground_truth.json" \
    --w1-attr Wearing_Hat --w2-attr Eyeglasses --top-k 1 \
    --n-seeds 50 --alpha 0.05 --max-steps 10 --gcm-splits 3 \
    --sweep effect --effect-grid 0 --fixed-n 500 2000 5000 \
    --methods "Marginal Testing" "Marginal Testing (FWER)" "NEXIS" \
              "NEXIS (test=GCM: quadratic)" "NEXIS (test=GCM: lgbm)" \
              "NEXIS (test=PCM: quadratic)" "NEXIS (test=PCM: lgbm)" \
    --merge
