#!/usr/bin/env bash
#SBATCH --job-name=permod
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=40
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#
# Per-modifier recall of NEXIS-v2 (src/apps/celeba/per_modifier_recall.py): replays the
# NEXIS-v2 runs of the r = 1, 2, 3 sweeps on the same draws, records which modifier
# coordinates were selected, and checks the replayed recall against the stored sweep.
# CPU only.  Results: results/celeba/per_modifier/r{1,2,3}.parquet.
#
#   bash   scripts/celeba/submit_per_modifier_recall.sh      # submit r1, r2, r3
#   sbatch scripts/celeba/submit_per_modifier_recall.sh r2   # one job
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
mkdir -p logs
PYTHON=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3
OUT=results/celeba/per_modifier

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  for set in r1 r2 r3; do
    if [[ -f "$OUT/$set.parquet" ]]; then echo "skip (exists): $OUT/$set.parquet"; continue; fi
    echo "permod-$set: $(sbatch --parsable --job-name="permod-$set" "$SELF" "$set")"
  done
  exit 0
fi

SET="$1"
echo "set=$SET host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}"
case "$SET" in
  r2) exec "$PYTHON" src/apps/celeba/per_modifier_recall.py --gammas 1 -1 \
          --out "$OUT/r2.parquet" --check results/celeba/experiment_v2/k20/sae ;;
  r1) exec "$PYTHON" src/apps/celeba/per_modifier_recall.py --gammas 1 0 \
          --out "$OUT/r1.parquet" --check results/celeba/experiment_v2_r1/k20/sae ;;
  r3) exec "$PYTHON" src/apps/celeba/per_modifier_recall.py \
          --w-attrs Wearing_Hat Eyeglasses Sideburns --gammas 1 -1 1 \
          --gt-json results/celeba/experiment_r3/k20/sae/ground_truth.json \
          --out "$OUT/r3.parquet" --check results/celeba/experiment_v2_r3/k20/sae ;;
  *) echo "unknown set: $SET" >&2; exit 1 ;;
esac
