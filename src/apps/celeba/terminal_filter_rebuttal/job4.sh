#!/usr/bin/env bash
#SBATCH --job-name=nexis-all4
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=06:00:00
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/.cache/nexis_cert
export NJOBS=12
exec /nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3 run_all6.py "$1" "$2" "$3"
