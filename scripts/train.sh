#!/usr/bin/env bash
# Run the full pipeline: DINOv2 extraction -> SAE training -> NEMS -> LLM interpretation.
# Always run from the project root: bash scripts/train.sh [flags]
#
# Flags:
#   --small      ViT-S, 1536 features, 50 epochs  (fast smoke-test)
#   --large      ViT-L, 4096 features, 150 epochs
#   --reextract  force re-running DINOv2 extraction (ignore cache)

set -euo pipefail

PYTHON="/Users/riccardocadei/miniforge3/bin/python3"
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"   # absolute path to scripts/

MODEL="dinov2_vitb14"
HIDDEN_DIM=3072
L1_COEFF=1e-3
EPOCHS=100
EXTRA_ARGS=""

for arg in "$@"; do
  case $arg in
    --small)     MODEL="dinov2_vits14"; HIDDEN_DIM=1536; EPOCHS=50 ;;
    --large)     MODEL="dinov2_vitl14"; HIDDEN_DIM=4096; EPOCHS=150 ;;
    --reextract) EXTRA_ARGS="$EXTRA_ARGS --force-reextract" ;;
    *)           EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
  esac
done

echo "============================================================"
echo " DINOv2 + SAE  ->  NEMS  ->  LLM interpretation"
echo "  model      : $MODEL"
echo "  hidden_dim : $HIDDEN_DIM"
echo "  l1_coeff   : $L1_COEFF"
echo "  epochs     : $EPOCHS"
echo "============================================================"

echo ""
echo "-- Step 1: DINOv2 extraction + SAE training -------------------------"
$PYTHON "$SCRIPTS/train.py" \
  --model      "$MODEL"      \
  --hidden-dim "$HIDDEN_DIM" \
  --l1-coeff   "$L1_COEFF"   \
  --epochs     "$EPOCHS"     \
  $EXTRA_ARGS

echo ""
echo "-- Step 2: NEMS feature selection -----------------------------------"
$PYTHON "$SCRIPTS/analyze.py" --alpha 0.05 --max-steps 20

echo ""
echo "-- Step 3: LLM interpretation of selected features ------------------"
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "  ANTHROPIC_API_KEY not set -- skipping."
  echo "  Run manually: python scripts/interpret.py"
else
  $PYTHON "$SCRIPTS/interpret.py" --k 6 --model claude-opus-4-6
fi

echo ""
echo "Done. Results in results/uganda/"
