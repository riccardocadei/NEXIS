#!/usr/bin/env bash
#SBATCH --job-name=celeba-images
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/NEMS/logs/celeba-images-%j.out
#SBATCH --error=/nfs/scistore19/locatgrp/rcadei/NEMS/logs/celeba-images-%j.err
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#
# Save 128×128 image thumbnails for visualization.
# Estimated runtime: ~25 min (streaming 20k images, CPU only).
#
# Submit:  sbatch scripts/celeba/submit_save_images.sh
# Or run:  bash   scripts/celeba/submit_save_images.sh

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

PYTHON="${PYTHON:-/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3}"

echo "============================================================"
echo " CelebA: save image thumbnails"
echo "============================================================"

$PYTHON src/apps/celeba/embed.py \
    --data-dir   data/celeba \
    --split      valid       \
    --save-images

echo "Done. Images saved to data/celeba/images.npy"
