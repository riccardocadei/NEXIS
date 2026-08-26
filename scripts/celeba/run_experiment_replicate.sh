#!/usr/bin/env bash
#SBATCH --job-name=celeba-rep
#SBATCH --output=logs/celeba-rep-%j.out
#SBATCH --error=logs/celeba-rep-%j.err
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=40
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#
# Independent replication of the CATE-equivalence test ablation on a DISJOINT seed
# block (default 50..99 vs 0..49 for the primary run), written to a separate results
# tree so the primary numbers are never touched.  The seed *is* the draw, so this is a
# genuine second Monte-Carlo sample, not a re-run of the same one.
#
# Usage: sbatch run_experiment_replicate.sh [attr|ushape] [effect|n|both] [seed_offset]

set -euo pipefail

DGP="${1:-attr}"
SWEEP="${2:-both}"
SEED_OFFSET="${3:-50}"
SAE_K=20

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3}"
export OMP_NUM_THREADS=1

TESTS=( "Marginal Testing (FWER)" "NEXIS"
        "NEXIS (test=GCM: quadratic)" "NEXIS (test=GCM: lgbm)"
        "NEXIS (test=PCM: quadratic)" "NEXIS (test=PCM: lgbm)" )

case "$DGP" in
    attr)
        OUT_DIR="results/celeba/experiment_rep"; FEAT="sae"; PRECODE=""
        EXTRA=( --fixed-n 500 2000 --fixed-effect 2.0 5.0 )
        ;;
    ushape)
        OUT_DIR="results/celeba/experiment_ushape_rep"; FEAT="sae_precode"; PRECODE="--precode"
        EXTRA=( --effect-form ortho_quadratic --fixed-n 2000 --fixed-effect 5.0 )
        ;;
    *) echo "Unknown DGP: $DGP" >&2; exit 1 ;;
esac

$PYTHON src/apps/celeba/run_experiment.py \
    $PRECODE --sae-top-k "$SAE_K" \
    --data-dir data/celeba --out-dir "$OUT_DIR" --backbone siglip \
    --gt-json "${OUT_DIR}/k${SAE_K}/${FEAT}/ground_truth.json" \
    --w1-attr Wearing_Hat --w2-attr Eyeglasses --top-k 1 \
    --n-seeds 50 --seed-offset "$SEED_OFFSET" \
    --alpha 0.05 --max-steps 10 --gcm-splits 3 \
    --sweep "$SWEEP" "${EXTRA[@]}" \
    --methods "${TESTS[@]}" \
    --merge
