#!/usr/bin/env bash
#SBATCH --job-name=nexis-prepare
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#
# Data preparation on one GPU: CelebA download, SigLIP 2 embeddings, TopK SAEs (main
# k = 5, 20 and replica k = 20), encoding and principal coordinates.  Add your cluster's
# partition / account options above.  Submit from the package root:
#   mkdir -p logs && sbatch slurm/prepare.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
python run.py prepare
