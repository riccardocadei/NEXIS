#!/usr/bin/env bash
#SBATCH --job-name=nexis-opt2p
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
cd /nfs/scistore19/locatgrp/rcadei/.cache/nexis_cert
sed -i 's/n_jobs=16/n_jobs=32/' "$1"
exec /nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3 "$1"
