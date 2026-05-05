#!/bin/bash
#SBATCH --job-name=test_cuda
#SBATCH --partition=gpu100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/test_cuda_%j.out
#SBATCH --error=logs/test_cuda_%j.err

export PYTHONUNBUFFERED=1

source /nfs/scistore19/locatgrp/rcadei/miniconda3/etc/profile.d/conda.sh
conda activate crl

echo "=== conda activated ==="
echo "Python: $(which python3)"

cd /nfs/scistore19/locatgrp/rcadei/NEXIS

python3 -u - <<'EOF'
import sys
print("Python started", flush=True)
print(f"Python: {sys.version}", flush=True)

print("Importing torch...", flush=True)
import torch
print(f"torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}", flush=True)

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x.T
    print(f"GPU matmul OK: {y.shape}", flush=True)
else:
    print("No CUDA — exiting", flush=True)
    sys.exit(1)

import numpy as np
emb = np.load("data/ghana/satellite/national/prithvi_embeddings.npy")
print(f"Embeddings loaded: {emb.shape}", flush=True)
print("ALL OK", flush=True)
EOF
