#!/usr/bin/env bash
#SBATCH --job-name=nems-reanalyze
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/NEMS/logs/slurm-%j.out
#SBATCH --error=/nfs/scistore19/locatgrp/rcadei/NEMS/logs/slurm-%j.err
#SBATCH --partition=gpu100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# Re-run analysis steps only (skip embedding/SAE training) for all/any models.
# By default skips the VLM interpret step (which needs exclusive GPU and cannot
# run in parallel).  Add --steps=analyze,interpret,summarize,plot explicitly
# to include interpretation.
#
# Usage:
#   sbatch scripts/uganda/reanalyze.sh [run.sh flags]
#   bash   scripts/uganda/reanalyze.sh --models=dinov2,dinov3,prithvi --all-outcomes
#   bash   scripts/uganda/reanalyze.sh --models=prithvi --outcomes=log_skilled_hours \
#                               --steps=analyze,interpret,summarize,plot

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

exec bash "$PROJECT_ROOT/scripts/uganda/run.sh" \
  --steps=analyze,interpret,summarize,plot \
  "$@"
