#!/usr/bin/env bash
#SBATCH --job-name=nems-pipeline
#SBATCH --output=/nfs/scistore19/locatgrp/rcadei/NEMS/logs/slurm-%j.out
#SBATCH --error=/nfs/scistore19/locatgrp/rcadei/NEMS/logs/slurm-%j.err
#SBATCH --partition=visualize
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# Full pipeline: embedding extraction → SAE → NEMS → LLM interpretation.
# Results are written to results/uganda/{MODEL}_{HIDDEN_DIM}/
#
# Usage:
#   sbatch scripts/run.sh [flags]
#   bash   scripts/run.sh [flags]
#
# Model presets (default: dinov2):
#   --dinov2        DINOv2-B/14  HF  768d  3072 features  (default)
#   --dinov3        DINOv3-B/16  HF  768d  3072 features
#   --dinov2-large  DINOv2-L/14  HF 1024d  4096 features
#   --dinov3-large  DINOv3-L/16  HF 1024d  4096 features
#
# Other flags:
#   --overwrite        re-extract embeddings even if cache exists
#   --skip-train       skip step 1 (use existing embeddings + SAE)
#   --no-w-candidates  treat W covariates as nuisance controls (not HTE candidates)
#   --w-priority       if any W candidate clears its gate, select W before SAE
#   --district-dummies one-hot encode district and include dummies as W candidates
#   --group-level      aggregate to group level before NEMS
#   --quantize         4-bit quantization for VLM (~4 GB VRAM vs ~14 GB)

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

SCRIPTS="$PROJECT_ROOT/scripts"
PYTHON="${PYTHON:-$(which python3)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# Load API key from ~/.anthropic_key if not already set
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -f "$HOME/.anthropic_key" ]; then
  source "$HOME/.anthropic_key"
fi

MODEL="dinov2"
HIDDEN_DIM=3072
L1_COEFF=2.0
EPOCHS=100
EXTRACT_BATCH=64
NUM_WORKERS=12
SAE_BATCH=16384
TRAIN_EXTRA_ARGS=""
ANALYZE_EXTRA_ARGS=""
INTERPRET_EXTRA_ARGS=""
SKIP_TRAIN=0

for arg in "$@"; do
  case $arg in
    --dinov2)           MODEL="dinov2";        HIDDEN_DIM=3072; EPOCHS=100;
                        EXTRACT_BATCH=64;  SAE_BATCH=16384 ;;
    --dinov3)           MODEL="dinov3";        HIDDEN_DIM=3072; EPOCHS=100;
                        EXTRACT_BATCH=64;  SAE_BATCH=16384 ;;
    --dinov2-large)     MODEL="dinov2_large";  HIDDEN_DIM=4096; EPOCHS=150;
                        EXTRACT_BATCH=32;  SAE_BATCH=8192 ;;
    --dinov3-large)     MODEL="dinov3_large";  HIDDEN_DIM=4096; EPOCHS=150;
                        EXTRACT_BATCH=32;  SAE_BATCH=8192 ;;
    --overwrite)        TRAIN_EXTRA_ARGS="$TRAIN_EXTRA_ARGS --overwrite" ;;
    --skip-train)       SKIP_TRAIN=1 ;;
    --no-w-candidates)  ANALYZE_EXTRA_ARGS="$ANALYZE_EXTRA_ARGS --no-w-candidates" ;;
    --w-priority)       ANALYZE_EXTRA_ARGS="$ANALYZE_EXTRA_ARGS --w-priority" ;;
    --district-dummies) ANALYZE_EXTRA_ARGS="$ANALYZE_EXTRA_ARGS --district-dummies" ;;
    --group-level)      ANALYZE_EXTRA_ARGS="$ANALYZE_EXTRA_ARGS --group-level" ;;
    --quantize)         INTERPRET_EXTRA_ARGS="$INTERPRET_EXTRA_ARGS --quantize" ;;
    *)                  TRAIN_EXTRA_ARGS="$TRAIN_EXTRA_ARGS $arg" ;;
  esac
done

echo "============================================================"
echo " Backbone + SAE  ->  NEMS  ->  LLM interpretation"
echo "  model      : $MODEL"
echo "  hidden_dim : $HIDDEN_DIM"
echo "  l1_coeff   : $L1_COEFF"
echo "  epochs     : $EPOCHS"
echo "  results in : results/uganda/${MODEL}_${HIDDEN_DIM}/"
echo "============================================================"

echo ""
if [ "$SKIP_TRAIN" -eq 1 ]; then
  echo "-- Step 1: Skipping (--skip-train) ----------------------------------"
else
echo "-- Step 1: Embedding extraction + SAE training ----------------------"
$PYTHON "$SCRIPTS/train.py" \
  --model               "$MODEL"         \
  --hidden-dim          "$HIDDEN_DIM"    \
  --l1-coeff            "$L1_COEFF"      \
  --epochs              "$EPOCHS"        \
  --extract-batch-size  "$EXTRACT_BATCH" \
  --num-workers         "$NUM_WORKERS"   \
  --batch-size          "$SAE_BATCH"     \
  $TRAIN_EXTRA_ARGS
fi

echo ""
echo "-- Step 2: NEMS feature selection -----------------------------------"
$PYTHON "$SCRIPTS/analyze.py" \
  --embed-model "$MODEL"      \
  --sae-dim     "$HIDDEN_DIM" \
  --alpha 0.05                \
  --max-steps 20              \
  $ANALYZE_EXTRA_ARGS

echo ""
echo "-- Step 3: VLM→LLM interpretation of selected features --------------"
$PYTHON -m pip install -q accelerate bitsandbytes
$PYTHON "$SCRIPTS/interpret.py" \
  --embed-model "$MODEL"        \
  --sae-dim     "$HIDDEN_DIM"   \
  --k 6                         \
  --vlm-model  Qwen/Qwen2-VL-7B-Instruct \
  $INTERPRET_EXTRA_ARGS         \
|| echo "  WARNING: interpret.py failed (see above). NEMS results are still saved."

echo ""
echo "-- Step 4: Results summary (ATE + CATE/GATE per modifier) ----------"
$PYTHON "$SCRIPTS/summarize.py" \
  --embed-model "$MODEL"        \
  --sae-dim     "$HIDDEN_DIM"

echo ""
echo "-- Step 5: Feature image plots (top/bottom activation) --------------"
$PYTHON "$SCRIPTS/plot_features.py" \
  --embed-model "$MODEL"            \
  --sae-dim     "$HIDDEN_DIM"       \
  --k 8

echo ""
echo "Done. Results in results/uganda/${MODEL}_${HIDDEN_DIM}/"
