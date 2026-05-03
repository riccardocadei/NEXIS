#!/usr/bin/env bash
#
# Stage 3: Launch SLURM jobs for all feature-type × SAE-k combinations.
# Results are organised as:
#   results/celeba/experiment/raw/            (K-independent)
#   results/celeba/experiment/k5/sae/
#   results/celeba/experiment/k5/sae_precode/
#   results/celeba/experiment/k20/sae/
#   results/celeba/experiment/k20/sae_precode/
#
# Submit:  bash scripts/celeba/submit_experiment.sh
# Or run sequentially: bash scripts/celeba/submit_experiment.sh --local

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

LOCAL="${1:-}"

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
    PYTHON="${PYTHON:-/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3}"
    $PYTHON src/apps/celeba/run_experiment.py --sae-top-k 5       "${COMMON_ARGS[@]}"
    $PYTHON src/apps/celeba/run_experiment.py --precode --sae-top-k 5  "${COMMON_ARGS[@]}"
    $PYTHON src/apps/celeba/run_experiment.py --sae-top-k 20      "${COMMON_ARGS[@]}"
    $PYTHON src/apps/celeba/run_experiment.py --precode --sae-top-k 20 "${COMMON_ARGS[@]}"
else
    sbatch scripts/celeba/run_experiment_single.sh sae          5
    sbatch scripts/celeba/run_experiment_single.sh sae_precode  5
    sbatch scripts/celeba/run_experiment_single.sh sae          20
    sbatch scripts/celeba/run_experiment_single.sh sae_precode  20
    echo "Submitted 4 parallel jobs (sae_k5 / sae_precode_k5 / sae_k20 / sae_precode_k20)."
fi
