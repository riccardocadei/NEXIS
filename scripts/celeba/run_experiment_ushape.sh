#!/usr/bin/env bash
#SBATCH --job-name=celeba-ushape
#SBATCH --output=logs/celeba-ushape-%j.out
#SBATCH --error=logs/celeba-ushape-%j.err
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=40
#SBATCH --mem=120G
#SBATCH --time=20:00:00
#
# CATE-equivalence test ablation on the GCM-blind DGP.
#
# τ is driven by an orthogonalised quadratic of the two ground-truth coordinates,
# so E[τ] and Cov(τ, Z^j) both vanish in the population while E[τ | Z^j] genuinely
# varies.  Every test that looks at a linear functional of Z^j — the default linear
# interaction t-test and both GCM variants — is blind by construction; the PCM,
# which learns its projection on a held-out half, is not.  Continuous SAE
# pre-activations are required: on sparse post-topk codes any function of Z^j is
# effectively two-valued, hence monotone, and the U-shape degenerates.
#
# Usage: sbatch run_experiment_ushape.sh [effect|n|both]

set -euo pipefail

SWEEP="${1:-both}"
SAE_K=20
BACKBONE=siglip

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3}"
export OMP_NUM_THREADS=1

OUT_DIR="results/celeba/experiment_ushape"

$PYTHON src/apps/celeba/run_experiment.py \
    --precode --sae-top-k "${SAE_K}" \
    --data-dir     data/celeba \
    --out-dir      "$OUT_DIR" \
    --backbone     "$BACKBONE" \
    --gt-json      "${OUT_DIR}/k${SAE_K}/sae_precode/ground_truth.json" \
    --effect-form  ortho_quadratic \
    --w1-attr      Wearing_Hat \
    --w2-attr      Eyeglasses \
    --top-k        1 \
    --n-seeds      50 \
    --alpha        0.05 \
    --max-steps    10 \
    --fixed-n      2000 \
    --fixed-effect 5.0 \
    --gcm-splits   3 \
    --sweep        "$SWEEP" \
    --methods      "Marginal Testing (FWER)" \
                   "NEXIS" \
                   "NEXIS (test=GCM: quadratic)" \
                   "NEXIS (test=GCM: lgbm)" \
                   "NEXIS (test=PCM: quadratic)" \
                   "NEXIS (test=PCM: lgbm)" \
    --merge
