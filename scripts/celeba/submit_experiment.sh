#!/usr/bin/env bash
#
# Stage 3: Launch SLURM jobs for all feature-type × SAE-k × sweep combinations.
# Results are organised as:
#   results/celeba/experiment/k5/sae/
#   results/celeba/experiment/k5/sae_precode/
#   results/celeba/experiment/k20/sae/
#   results/celeba/experiment/k20/sae_precode/
#
# Parallelisation strategy:
#   - 8 "fast" jobs  (effect + n split, 56 CPUs each) — all methods except GCM: lgbm
#   - 8 "GCM"  jobs  (lgbm only, afterok on paired fast job)
#   = 16 jobs total; all 8 fast jobs run simultaneously.
#
# Submit:  bash scripts/celeba/submit_experiment.sh
# Or run sequentially: bash scripts/celeba/submit_experiment.sh --local

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

LOCAL="${1:-}"

PYTHON="${PYTHON:-/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3}"

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
    --force
)

if [[ "$LOCAL" == "--local" ]]; then
    for K in 5 20; do
        for SWEEP in effect n; do
            $PYTHON src/apps/celeba/run_experiment.py --sae-top-k "$K" \
                --methods "${FAST_METHODS[@]}" --sweep "$SWEEP" "${COMMON_ARGS[@]}"
            $PYTHON src/apps/celeba/run_experiment.py --precode --sae-top-k "$K" \
                --methods "${FAST_METHODS[@]}" --sweep "$SWEEP" "${COMMON_ARGS[@]}"
            $PYTHON src/apps/celeba/run_experiment.py --sae-top-k "$K" \
                --methods "NEXIS (test=GCM: lgbm)" --sweep "$SWEEP" "${COMMON_ARGS[@]}"
            $PYTHON src/apps/celeba/run_experiment.py --precode --sae-top-k "$K" \
                --methods "NEXIS (test=GCM: lgbm)" --sweep "$SWEEP" "${COMMON_ARGS[@]}"
        done
    done
else
    # 8 fast jobs — all start immediately, one per (config × sweep)
    F_EFF_SAE5=$( sbatch --parsable scripts/celeba/run_experiment_single.sh sae         5 effect)
    F_EFF_PRE5=$( sbatch --parsable scripts/celeba/run_experiment_single.sh sae_precode 5 effect)
    F_EFF_SAE20=$(sbatch --parsable scripts/celeba/run_experiment_single.sh sae         20 effect)
    F_EFF_PRE20=$(sbatch --parsable scripts/celeba/run_experiment_single.sh sae_precode 20 effect)
    F_N_SAE5=$(   sbatch --parsable scripts/celeba/run_experiment_single.sh sae         5 n)
    F_N_PRE5=$(   sbatch --parsable scripts/celeba/run_experiment_single.sh sae_precode 5 n)
    F_N_SAE20=$(  sbatch --parsable scripts/celeba/run_experiment_single.sh sae         20 n)
    F_N_PRE20=$(  sbatch --parsable scripts/celeba/run_experiment_single.sh sae_precode 20 n)

    # 8 GCM jobs — each waits for its paired fast job
    sbatch --dependency=afterok:$F_EFF_SAE5  scripts/celeba/run_experiment_gcm.sh sae         5 effect
    sbatch --dependency=afterok:$F_EFF_PRE5  scripts/celeba/run_experiment_gcm.sh sae_precode 5 effect
    sbatch --dependency=afterok:$F_EFF_SAE20 scripts/celeba/run_experiment_gcm.sh sae         20 effect
    sbatch --dependency=afterok:$F_EFF_PRE20 scripts/celeba/run_experiment_gcm.sh sae_precode 20 effect
    sbatch --dependency=afterok:$F_N_SAE5    scripts/celeba/run_experiment_gcm.sh sae         5 n
    sbatch --dependency=afterok:$F_N_PRE5    scripts/celeba/run_experiment_gcm.sh sae_precode 5 n
    sbatch --dependency=afterok:$F_N_SAE20   scripts/celeba/run_experiment_gcm.sh sae         20 n
    sbatch --dependency=afterok:$F_N_PRE20   scripts/celeba/run_experiment_gcm.sh sae_precode 20 n

    echo "Submitted 16 jobs:"
    echo "  fast effect: $F_EFF_SAE5 $F_EFF_PRE5 $F_EFF_SAE20 $F_EFF_PRE20"
    echo "  fast n:      $F_N_SAE5 $F_N_PRE5 $F_N_SAE20 $F_N_PRE20"
    echo "  GCM (afterok on paired fast jobs)"
fi
