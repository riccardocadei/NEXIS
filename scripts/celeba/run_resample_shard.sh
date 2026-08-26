#!/usr/bin/env bash
#SBATCH --job-name=celeba-rs-exp
#SBATCH --output=logs/celeba-rs-exp-%j.out
#SBATCH --error=logs/celeba-rs-exp-%j.err
#SBATCH --partition=defaultp
#SBATCH --time=04:00:00
#
# One shard of the resampled-SAE experiment rerun.  A shard is
#   (feature view) × (SAE k) × (sweep) × (one fixed value) × (method group) × (seed block),
# written to its own out-dir so shards never race; merge_shards.py concatenates them
# into the canonical results/celeba/experiment_resample_<TAG>/k<K>/<feature>/ layout.
#
# cpus-per-task / mem are set by the submitter (n-sweeps need more memory: the n=10,000
# design matrix is 10,000 × 9,216 per worker thread).
#
# Usage: sbatch run_resample_shard.sh TAG FEATURE K SWEEP FIXED GROUP SEED_OFFSET N_SEEDS

set -euo pipefail

TAG="${1}"; FEATURE="${2}"; K="${3}"; SWEEP="${4}"; FIXED="${5}"
GROUP="${6}"; SEED_OFFSET="${7}"; N_SEEDS="${8}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3}"
# One BLAS thread per worker: run_sweep already parallelises over (grid point × seed).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-8}"

case "$GROUP" in
    g1) METHODS=("Marginal Testing" "Marginal Testing (FWER)" "Marginal Testing (FDR)"
                 "NEXIS" "NEXIS (adjust=None)" "NEXIS (adjust=FDR)") ;;
    g2) METHODS=("NEXIS (rho=0)" "NEXIS (rho=0.2)" "NEXIS (rho=0.8)"
                 "NEXIS (backward=False)" "NEXIS (test=GCM: quadratic)") ;;
    g3) METHODS=("NEXIS (test=GCM: lgbm)") ;;
    *)  echo "Unknown method group: $GROUP" >&2; exit 1 ;;
esac

SHARD="${FEATURE}_k${K}_${SWEEP}${FIXED}_${GROUP}_s${SEED_OFFSET}"
OUT="results/celeba/experiment_resample_${TAG}/shards/${SHARD}"

ARGS=(
    --data-dir     "data/celeba_resample_${TAG}/eval"
    --out-dir      "$OUT"
    --backbone     siglip
    --sae-top-k    "$K"
    --w1-attr      Wearing_Hat
    --w2-attr      Eyeglasses
    --top-k        1
    --n-seeds      "$N_SEEDS"
    --seed-offset  "$SEED_OFFSET"
    --alpha        0.05
    --max-steps    10
    --gcm-splits   3
    --sweep        "$SWEEP"
    --methods      "${METHODS[@]}"
    --force
)

# The sweep that is not being run still needs its --fixed-* argument to be a no-op.
if [[ "$SWEEP" == "effect" ]]; then
    ARGS+=(--fixed-n "$FIXED")
else
    ARGS+=(--fixed-effect "$FIXED")
fi

if [[ "$FEATURE" == "sae_precode" ]]; then
    ARGS+=(--precode)
fi

echo "shard=${SHARD}  cpus=${SLURM_CPUS_PER_TASK:-?}  methods=${#METHODS[@]}"
$PYTHON src/apps/celeba/run_experiment.py "${ARGS[@]}"
echo "shard ${SHARD} complete."
