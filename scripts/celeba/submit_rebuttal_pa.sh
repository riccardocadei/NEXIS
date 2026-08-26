#!/usr/bin/env bash
#SBATCH --job-name=celeba-pa
#SBATCH --output=logs/celeba-pa-%x-%j.out
#SBATCH --error=logs/celeba-pa-%x-%j.err
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=40
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#
# Rebuttal analyses for Principal Alignment (Assumption 3).
#
#   sbatch scripts/celeba/submit_rebuttal_pa.sh diagnostics
#   sbatch scripts/celeba/submit_rebuttal_pa.sh alignment
#   sbatch scripts/celeba/submit_rebuttal_pa.sh violation
#   bash   scripts/celeba/submit_rebuttal_pa.sh all        # submit all three
set -euo pipefail

# Slurm copies the batch script to its spool dir, so BASH_SOURCE is only usable
# at submit time; inside the job SLURM_SUBMIT_DIR is the repo root.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
mkdir -p logs
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export JOBLIB_TEMP_FOLDER="${SLURM_TMPDIR:-/tmp}/joblib-${SLURM_JOB_ID:-$$}"
mkdir -p "$JOBLIB_TEMP_FOLDER"
trap 'rm -rf "$JOBLIB_TEMP_FOLDER"' EXIT

NJOBS="${SLURM_CPUS_PER_TASK:-8}"
TASK="${1:-all}"

if [[ "$TASK" == "all" && -z "${SLURM_JOB_ID:-}" ]]; then
  SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  sbatch --job-name="celeba-pa-diagnostics"      "$SELF" diagnostics
  sbatch --job-name="celeba-pa-alignment"        "$SELF" alignment
  sbatch --job-name="celeba-pa-violation-small"  "$SELF" violation-small
  # The n=10^4 arm holds several n x m float64 buffers per worker inside the
  # conditional test (~4 GB each), so it needs few workers and a lot of memory.
  sbatch --job-name="celeba-pa-violation-large" --cpus-per-task=12 --mem=180G \
         "$SELF" violation-large
  exit 0
fi

case "$TASK" in
  diagnostics)
    # Alignment diagnostics for every cached dictionary, incl. the held-out
    # conditional-independence probe.  Cheap; single-threaded.
    python -u src/apps/celeba/principal_alignment.py --all --probe
    ;;

  alignment)
    # NEXIS recovery vs measured alignment, across all dictionaries.
    python -u src/apps/celeba/alignment_vs_recovery.py \
      --n-seeds 50 --fixed-n 2000 --fixed-effect 5.0 --n-jobs "$NJOBS"
    ;;

  violation-small)
    # Controlled Principal-Alignment violation via concept splitting, n = 2000.
    python -u src/apps/celeba/violation_sweep.py \
      --eps-grid 0.0 0.05 0.1 0.2 0.3 0.4 0.5 \
      --fixed-n 2000 --fixed-effect 5.0 \
      --n-seeds 50 --n-jobs "$NJOBS" --tag n2000
    ;;

  violation-large)
    # Same sweep at n = 10^4: tests that the rho-gate does not degrade with power.
    python -u src/apps/celeba/violation_sweep.py \
      --eps-grid 0.0 0.05 0.1 0.2 0.3 0.4 0.5 \
      --fixed-n 10000 --fixed-effect 5.0 \
      --n-seeds 50 --n-jobs "$NJOBS" --tag n10000
    ;;

  *)
    echo "unknown task: $TASK" >&2
    echo "  (diagnostics | alignment | violation-small | violation-large | all)" >&2
    exit 1
    ;;
esac
