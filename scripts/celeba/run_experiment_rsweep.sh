#!/usr/bin/env bash
#SBATCH --job-name=celeba-rsweep
#SBATCH --output=logs/celeba-rsweep-%j.out
#SBATCH --error=logs/celeba-rsweep-%j.err
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#
# Memory: the n sweep reaches n = 10,000 on a 9,216-column feature matrix, i.e.
# ~737 MB per worker for Z alone, and joblib gives each worker its own copy.  At
# 40 workers / 64 GB (the settings inherited from run_experiment_single.sh) both
# n-sweep jobs were OOM-killed at exactly 64 GB; the effect sweep survives only
# because it never exceeds n = 2000.  32 workers with 400 GB clears it.
#
# Truth-set-size sweep worker: the published benchmark fixes |S*| = 2, which is
# the regime in which the backward step provably has nothing to prune (the forward
# pass returns a conditionally independent pair and the backward gate alpha/|S| is
# ~4600x looser than the forward gate alpha/m).  This worker reruns the whole
# ablation at a larger truth set so the claim that the backward step's value grows
# with |S*| can be measured instead of asserted.
#
# Attribute choice is not free.  Each added modifier must (i) be carried by ONE
# dominant SAE coordinate, else S* is not well defined and precision penalises
# target misspecification rather than the algorithm, and (ii) leave every joint
# cell populated, since the sampler draws attributes independently and then needs
# a real CelebA image in the matching cell.  Measured on the k=20 SigLIP
# dictionary (best-threshold F1, gap to the runner-up coordinate):
#
#   Wearing_Hat  F1 0.871  gap 0.328      Sideburns   F1 0.697  gap 0.244
#   Eyeglasses   F1 0.943  gap 0.223      Blond_Hair  F1 0.802  gap 0.207
#
# Sideburns is the only well-aligned attribute that also co-occurs with hats and
# glasses: its binding cell holds 698 images and supports n <= 17,028.  Blond_Hair
# and Bangs are equally well aligned but the (hat, glasses, blond) cell contains
# ONE CelebA image, so they cannot be used with this sampler.
#
# Usage: sbatch run_experiment_rsweep.sh [r] [effect|n|both] [feature] [k] [backbone]

set -euo pipefail

R="${1:-3}"
SWEEP="${2:-both}"
FEATURE="${3:-sae}"
SAE_K="${4:-20}"
BACKBONE="${5:-siglip}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3}"

case "$R" in
    2) W_ATTRS=(Wearing_Hat Eyeglasses) ;;
    3) W_ATTRS=(Wearing_Hat Eyeglasses Sideburns) ;;
    *) echo "No vetted attribute set for r=$R. Only r=2 and r=3 satisfy both the" \
            "single-coordinate and the populated-joint-cell requirements; see the" \
            "header and src/apps/celeba/scm.py." >&2; exit 1 ;;
esac

COMMON_ARGS=(
    --data-dir     data/celeba
    --out-dir      "results/celeba/experiment_r${R}"
    --backbone     "$BACKBONE"
    --w-attrs      "${W_ATTRS[@]}"
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

METHODS=(
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

case "$FEATURE" in
    sae)
        $PYTHON src/apps/celeba/run_experiment.py \
            --sae-top-k "$SAE_K" --methods "${METHODS[@]}" "${COMMON_ARGS[@]}"
        ;;
    sae_precode)
        $PYTHON src/apps/celeba/run_experiment.py \
            --precode --sae-top-k "$SAE_K" --methods "${METHODS[@]}" "${COMMON_ARGS[@]}"
        ;;
    *) echo "Unknown feature type: $FEATURE" >&2; exit 1 ;;
esac
