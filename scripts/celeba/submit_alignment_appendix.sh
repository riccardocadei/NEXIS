#!/usr/bin/env bash
#SBATCH --job-name=pa-appendix
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#
# Supervised check of Principal Alignment on the main CelebA dictionary (appendix panel,
# src/apps/celeba/alignment_appendix.py): alignment spectrum, top-activating images of
# the principals, screening-off probe.  CPU only.
# Results: results/celeba/figures_v2/principal_alignment/pa_*.
#
#   sbatch scripts/celeba/submit_alignment_appendix.sh      (or: bash ...)
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
mkdir -p logs
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
PYTHON=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}"
exec "$PYTHON" src/apps/celeba/alignment_appendix.py "$@"
