#!/usr/bin/env bash
#SBATCH --job-name=celeba-ibwd
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#
# Interleaved backward step experiment (CPU only).  Generic wrapper: runs one
# python module of src/apps/celeba/ with the given arguments.
#
#   sbatch scripts/celeba/submit_interleaved_backward.sh interleaved_backward_align.py sae_k20 sae_k5
#   sbatch scripts/celeba/submit_interleaved_backward.sh interleaved_backward.py --tag T ... --sweep effect --fixed 2000 --grid 1,2
#   sbatch scripts/celeba/submit_interleaved_backward.sh pilot --tag T --attrs ...   # standard pilot:
#       eta in {1,2,3,5} at n=2000 and eta in {2,5,10} at n=500, 20 seeds each
#   sbatch scripts/celeba/submit_interleaved_backward.sh sweep --tag T --attrs ...   # full sweep:
#       eta in 1..10 at n in {500,2000}; n in {50..10000} at eta in {2,5}; 50 seeds
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
mkdir -p logs
PYTHON=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3
export NJOBS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SCRIPT="$1"; shift
RUN=src/apps/celeba/interleaved_backward.py
case "$SCRIPT" in
  pilot)
    "$PYTHON" $RUN "$@" --sweep effect --fixed 2000 --grid 1,2,3,5 --seeds 20
    "$PYTHON" $RUN "$@" --sweep effect --fixed 500 --grid 2,5,10 --seeds 20 ;;
  sweep)
    "$PYTHON" $RUN "$@" --sweep effect --fixed 2000 --grid 1,2,3,4,5,6,7,8,9,10 --seeds 50
    "$PYTHON" $RUN "$@" --sweep effect --fixed 500 --grid 1,2,3,4,5,6,7,8,9,10 --seeds 50
    "$PYTHON" $RUN "$@" --sweep n --fixed 5 --grid 50,100,200,350,500,750,1000,2000,3500,5000,10000 --seeds 50
    "$PYTHON" $RUN "$@" --sweep n --fixed 2 --grid 50,100,200,350,500,750,1000,2000,3500,5000,10000 --seeds 50 ;;
  *) exec "$PYTHON" "src/apps/celeba/$SCRIPT" "$@" ;;
esac
