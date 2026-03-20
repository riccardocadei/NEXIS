#!/usr/bin/env bash
#SBATCH --job-name=nems-reanalyze
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/NEMS/logs/slurm-%j.out
#SBATCH --error=/nfs/scistore19/locatgrp/rcadei/NEMS/logs/slurm-%j.err
#SBATCH --partition=visualize
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#
# Re-run NEMS + LLM interpretation for all trained models (skip embedding/SAE).
# Usage:
#   sbatch scripts/reanalyze.sh

set -euo pipefail

# Load API key from ~/.anthropic_key if not already set
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -f "$HOME/.anthropic_key" ]; then
  source "$HOME/.anthropic_key"
fi

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPT="$PROJECT_ROOT/scripts/run.sh"

for MODEL_FLAG in --dinov2 --dinov3; do
  bash "$SCRIPT" "$MODEL_FLAG" --skip-train --w-priority --district-dummies --quantize
done
