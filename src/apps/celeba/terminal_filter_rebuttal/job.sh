#!/usr/bin/env bash
#SBATCH --job-name=nexis-cert
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/.cache/nexis_cert
export NJOBS=16
exec /nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3 run_certified_all.py "$1" "$2" "$3"
