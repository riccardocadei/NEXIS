#!/usr/bin/env bash
#
# Stage 3: Launch SLURM jobs for all feature-type × SAE-k combinations.
# Results are organised as:
#   results/celeba/experiment/k5/sae/
#   results/celeba/experiment/k5/sae_precode/
#   results/celeba/experiment/k20/sae/
#   results/celeba/experiment/k20/sae_precode/
#
# Parallelisation strategy:
#   - 4 "fast" jobs  (linear tests, ~8 h each)  — all methods except NEXIS (test=GCM: lgbm)
#   - 4 "GCM"  jobs  (lgbm test, ~16 h each)    — NEXIS (test=GCM: lgbm) only
#   = 8 jobs total, GCM finishes independently of fast methods.
#
# Submit:  bash scripts/celeba/submit_experiment.sh
# Or run sequentially: bash scripts/celeba/submit_experiment.sh --local

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

LOCAL="${1:-}"

FAST_METHODS=(
    "Marginal Testing"
    "Marginal Testing (FWER)"
    "Marginal Testing (FDR)"
    "NEXIS"
    "NEXIS (test=GCM: quadratic)"
    "NEXIS (adjust=None)"
    "NEXIS (adjust=FDR)"
    "NEXIS (rho=0)"
    "NEXIS (rho=0.1)"
    "NEXIS (rho=0.2)"
    "NEXIS (backward=False)"
)

COMMON_ARGS=(
    --data-dir     data/celeba
    --out-dir      results/celeba/experiment
    --w1-attr      Wearing_Hat
    --w2-attr      Eyeglasses
    --top-k        1
    --n-seeds      20
    --alpha        0.05
    --max-steps    10
    --fixed-n      500 2000
    --fixed-effect 2.0 5.0
    --gcm-splits   3
    --force
)

PYTHON="${PYTHON:-/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3}"

if [[ "$LOCAL" == "--local" ]]; then
    for K in 5 20; do
        $PYTHON src/apps/celeba/run_experiment.py --sae-top-k "$K" \
            --methods "${FAST_METHODS[@]}" "${COMMON_ARGS[@]}"
        $PYTHON src/apps/celeba/run_experiment.py --precode --sae-top-k "$K" \
            --methods "${FAST_METHODS[@]}" "${COMMON_ARGS[@]}"
        $PYTHON src/apps/celeba/run_experiment.py --sae-top-k "$K" \
            --methods "NEXIS (test=GCM: lgbm)" "${COMMON_ARGS[@]}"
        $PYTHON src/apps/celeba/run_experiment.py --precode --sae-top-k "$K" \
            --methods "NEXIS (test=GCM: lgbm)" "${COMMON_ARGS[@]}"
    done
else
    # Fast jobs (8 h)
    sbatch scripts/celeba/run_experiment_single.sh sae         5
    sbatch scripts/celeba/run_experiment_single.sh sae_precode 5
    sbatch scripts/celeba/run_experiment_single.sh sae         20
    sbatch scripts/celeba/run_experiment_single.sh sae_precode 20
    # GCM jobs (16 h)
    sbatch scripts/celeba/run_experiment_gcm.sh    sae         5
    sbatch scripts/celeba/run_experiment_gcm.sh    sae_precode 5
    sbatch scripts/celeba/run_experiment_gcm.sh    sae         20
    sbatch scripts/celeba/run_experiment_gcm.sh    sae_precode 20
    echo "Submitted 8 jobs:"
    echo "  fast (8 h):  sae_k5 / sae_precode_k5 / sae_k20 / sae_precode_k20"
    echo "  GCM  (16 h): sae_k5 / sae_precode_k5 / sae_k20 / sae_precode_k20"
fi
