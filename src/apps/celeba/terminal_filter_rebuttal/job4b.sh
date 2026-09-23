#!/usr/bin/env bash
#SBATCH --job-name=nexis-a4b
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=08:00:00
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/.cache/nexis_cert
export NJOBS=6
exec /nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3 run_all6.py "$1" "$2" "$3"
