#!/bin/bash
#SBATCH --job-name=ghana_interp_rep
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/NEXIS/logs/ghana_interp_rep_%j.out
#SBATCH --error=/nfs/scistore19/locatgrp/rcadei/NEXIS/logs/ghana_interp_rep_%j.err
#SBATCH --partition=gpu100
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#
# Replication of the published Ghana VLM labels.
#
# The stored interpretations for nexis_fwer were produced from (a) the buggy
# 3-feature selection and (b) the `leap` image pool -- its top tiles peak at
# activation 2.85, against 5.77 on the national grid -- and they disagree with the
# paper ("active land clearing" vs "ephemeral waterways").  This re-runs the
# documented protocol: the corrected 2-feature selection, the national grid pool
# that appendix sec:ghana:vlm specifies, k=12 tiles per side, greedy decoding.
#
# Expected if the paper replicates:
#   neuron 3821 -> "ephemeral waterways"    (high confidence)
#   neuron 2095 -> "Closed-canopy forest"   (high confidence)

set -euo pipefail
ROOT=/nfs/scistore19/locatgrp/rcadei/NEXIS
PYTHON=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3
mkdir -p "$ROOT/logs"; cd "$ROOT"

$PYTHON src/apps/ghana/interpret.py \
    --mode codes \
    --min-activations 5 \
    --method nexis_fwer \
    --interpret-only \
    --overwrite \
    --neurons 3821,2095 \
    --pool national \
    --vlm-model Qwen/Qwen2.5-VL-72B-Instruct \
    --pipeline qwen25_72b \
    --quantize \
    --k 12 \
    "$@"
