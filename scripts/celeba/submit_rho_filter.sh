#!/usr/bin/env bash
#SBATCH --job-name=celeba-rho-filter
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G  # n=10000 cells OOM at 48G
#SBATCH --time=12:00:00
#
# Default-NEXIS runs behind the Appendix run statistics (|S~| median/max, terminal
# tests median/max over 6,300 runs; aggregated by src/apps/celeba/run_statistics.py),
# plus the rho x terminal-filter ablation, on the CelebA grid
# (src/apps/celeba/ablation_rho_filter.py).  CPU only.
#
#   bash   scripts/celeba/submit_rho_filter.sh                 # submit all 12 (tree x sweep)
#   sbatch scripts/celeba/submit_rho_filter.sh k5/sae n 2       # one job
#
# Jobs whose CSV already exists in results/celeba/ablation_rho_filter/ are skipped.
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
mkdir -p logs
PYTHON=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3
OUT=results/celeba/ablation_rho_filter

if [[ $# -eq 0 && -z "${SLURM_JOB_ID:-}" ]]; then
  SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  for tree in k20/sae k20/sae_precode k5/sae; do
    for sf in "effect 500" "effect 2000" "n 2" "n 5"; do
      set -- $sf
      csv="$OUT/rf_${tree//\//_}_$1_$2.csv"
      if [[ -f "$csv" ]]; then echo "skip (exists): $csv"; continue; fi
      sbatch --job-name="rf-${tree//\//_}-$1-$2" "$SELF" "$tree" "$1" "$2"
    done
  done
  exit 0
fi

export NJOBS="${SLURM_CPUS_PER_TASK:-12}"
exec "$PYTHON" src/apps/celeba/ablation_rho_filter.py "$@"
