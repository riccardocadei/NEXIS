#!/usr/bin/env bash
#SBATCH --job-name=nexis-celeba
#SBATCH --cpus-per-task=40
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#
# One unit of work (experiment x sweep x method) per array task, on CPU.  Add your
# cluster's partition / account options above.  From the package root:
#   BLOCK=ablations-method
#   N=$(python run.py $BLOCK --list-units | wc -l)
#   mkdir -p logs && sbatch --array=0-$((N - 1)) slurm/run_units.sbatch $BLOCK
#   python run.py $BLOCK        # afterwards: skips the finished units, draws the figures
# Units whose output exists are skipped, so failed tasks can simply be resubmitted.
# Memory: a run at n = 10,000 holds ~4 GB, so 200 GB covers 40 concurrent runs.
set -euo pipefail
BLOCK="${1:?usage: sbatch --array=0-N slurm/run_units.sbatch BLOCK}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # parallel over runs
python run.py "$BLOCK" --unit "$SLURM_ARRAY_TASK_ID" --jobs "${SLURM_CPUS_PER_TASK:-1}"
