#!/usr/bin/env bash
#SBATCH --job-name=nexis-pipeline
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=gpu100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# Full pipeline: embedding extraction → SAE → NEXIS → LLM interpretation.
# Results are written to results/uganda/{MODEL}_{HIDDEN_DIM}/
#
# Usage:
#   sbatch scripts/uganda/run.sh [flags]
#   bash   scripts/uganda/run.sh [flags]
#
# Key flags:
#   --models=MODEL[,MODEL...]   comma-separated backbone presets (default: prithvi)
#   --outcomes=NAME[,NAME...]   comma-separated outcome aliases  (default: log_skilled_hours)
#                               aliases: log_skilled_hours  skilled_employed  skilled_fulltime
#                                        employ_hours   log_training_hours
#                                        log_earnings   log_biz_assets
#                                        wealth_index   wellbeing
#   --steps=STEP[,STEP...]      steps to run (default: train,analyze,interpret,summarize,plot)
#   --overwrite[=STEP[,...]]    force-rerun steps even if output exists;
#                               bare --overwrite re-runs all steps
#
# Model presets (use in --models=):
#   dinov2        DINOv2-B/14   768d  3072 features  (default)
#   dinov3        DINOv3-B/16   768d  3072 features
#   dinov2_large  DINOv2-L/14  1024d  4096 features
#   dinov3_large  DINOv3-L/16  1024d  4096 features
#   prithvi       Prithvi-EO    768d  1024 features  (year-2000 CSV images)
#   prithvi_l5    Prithvi-EO    768d  1024 features  (Landsat 7 2005–2007 pre-treatment GeoTIFF)
#
# Examples:
#   bash scripts/uganda/run.sh
#   bash scripts/uganda/run.sh --models=dinov2,dinov3 --outcomes=log_skilled_hours
#   bash scripts/uganda/run.sh --models=prithvi --all-outcomes
#   bash scripts/uganda/run.sh --models=dinov2 --steps=analyze,interpret,summarize,plot
#   bash scripts/uganda/run.sh --models=dinov2 --overwrite=analyze,interpret
#   bash scripts/uganda/run.sh --models=dinov2 --overwrite          # re-run everything
#
# NEXIS tuning:
#   --alpha=0.05  --max-steps=20  --l1-coeff=2.0
#   --no-w-candidates  --w-priority  --district-dummies  --group-level
#
# VLM / interpretation:
#   --vlm-model=MODEL_ID    (default: Qwen/Qwen2-VL-7B-Instruct)  [qwen pipeline]
#   --geochat-model=ID      (default: MBZUAI/geochat-7b)           [geochat pipeline]
#   --text-model=ID         (default: Qwen/Qwen2.5-72B-Instruct)   [geochat pipeline]
#   --pipelines=qwen,geochat  which interpretation pipelines to run (default: both)
#   --quantize              4-bit quantization (~4 GB VRAM vs ~14 GB)

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

SCRIPTS="$PROJECT_ROOT/src/apps/uganda"
PYTHON="${PYTHON:-/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# Load API key from ~/.anthropic_key if not already set
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -f "$HOME/.anthropic_key" ]; then
  source "$HOME/.anthropic_key"
fi

# ── Defaults ──────────────────────────────────────────────────────────────────
MODELS_ARG="prithvi"
OUTCOMES_ARG="log_skilled_hours"
ALL_OUTCOMES="log_skilled_hours,skilled_employed,skilled_fulltime,log_training_hours,log_earnings,log_biz_assets,wealth_index,wellbeing"
STEPS_ARG="train,analyze,interpret,summarize,plot"
OVERWRITE_ARG=""

L1_COEFF="2.0"
NUM_WORKERS=12
EXTRACT_BATCH_OVERRIDE=""
SAE_BATCH_OVERRIDE=""
EPOCHS_OVERRIDE=""

ALPHA="0.05"
MAX_STEPS="20"
ANALYZE_EXTRA_ARGS=""
INTERPRET_EXTRA_ARGS="--quantize"
VLM_MODEL="Qwen/Qwen2-VL-7B-Instruct"
GEOCHAT_MODEL="MBZUAI/geochat-7b"
TEXT_MODEL="Qwen/Qwen2.5-72B-Instruct"
PIPELINES_ARG="qwen,geochat"

# ── Parse args ────────────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    # Multi-value flags
    --models=*)      MODELS_ARG="${arg#--models=}" ;;
    --outcomes=*)    OUTCOMES_ARG="${arg#--outcomes=}" ;;
    --all-outcomes)  OUTCOMES_ARG="$ALL_OUTCOMES" ;;
    --steps=*)       STEPS_ARG="${arg#--steps=}" ;;
    --overwrite=*)   OVERWRITE_ARG="${arg#--overwrite=}" ;;
    --overwrite)     OVERWRITE_ARG="train,analyze,interpret,summarize,plot" ;;

    # Legacy single-model shortcuts (kept for backward compat)
    --dinov2)        MODELS_ARG="dinov2" ;;
    --dinov3)        MODELS_ARG="dinov3" ;;
    --dinov2-large)  MODELS_ARG="dinov2_large" ;;
    --dinov3-large)  MODELS_ARG="dinov3_large" ;;
    --prithvi)       MODELS_ARG="prithvi" ;;
    --prithvi-l5)    MODELS_ARG="prithvi_l5" ;;

    # Legacy flags
    --skip-train)
      STEPS_ARG="${STEPS_ARG/train,/}"
      STEPS_ARG="${STEPS_ARG/,train/}"
      STEPS_ARG="${STEPS_ARG/train/}"
      ;;
    --outcome=*)     OUTCOMES_ARG="${arg#--outcome=}" ;;

    # Numeric overrides
    --l1-coeff=*)        L1_COEFF="${arg#--l1-coeff=}" ;;
    --epochs=*)          EPOCHS_OVERRIDE="${arg#--epochs=}" ;;
    --extract-batch=*)   EXTRACT_BATCH_OVERRIDE="${arg#--extract-batch=}" ;;
    --sae-batch=*)       SAE_BATCH_OVERRIDE="${arg#--sae-batch=}" ;;
    --num-workers=*)     NUM_WORKERS="${arg#--num-workers=}" ;;
    --alpha=*)           ALPHA="${arg#--alpha=}" ;;
    --max-steps=*)       MAX_STEPS="${arg#--max-steps=}" ;;

    # analyze flags
    --no-w-candidates)   ANALYZE_EXTRA_ARGS="$ANALYZE_EXTRA_ARGS --no-w-candidates" ;;
    --w-priority)        ANALYZE_EXTRA_ARGS="$ANALYZE_EXTRA_ARGS --w-priority" ;;
    --district-dummies)  ANALYZE_EXTRA_ARGS="$ANALYZE_EXTRA_ARGS --district-dummies" ;;
    --group-level)       ANALYZE_EXTRA_ARGS="$ANALYZE_EXTRA_ARGS --group-level" ;;

    # interpret flags
    --quantize)          INTERPRET_EXTRA_ARGS="$INTERPRET_EXTRA_ARGS --quantize" ;;
    --vlm-model=*)       VLM_MODEL="${arg#--vlm-model=}" ;;
    --geochat-model=*)   GEOCHAT_MODEL="${arg#--geochat-model=}" ;;
    --text-model=*)      TEXT_MODEL="${arg#--text-model=}" ;;
    --pipelines=*)       PIPELINES_ARG="${arg#--pipelines=}" ;;

    *) echo "Unknown flag: $arg" >&2; exit 1 ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────

# model_params MODEL_KEY → "MODEL HIDDEN_DIM EPOCHS EXTRACT_BATCH SAE_BATCH"
model_params() {
  case $1 in
    dinov2)       echo "dinov2       3072 100 64 16384" ;;
    dinov3)       echo "dinov3       3072 100 64 16384" ;;
    dinov2_large) echo "dinov2_large 4096 150 32  8192" ;;
    dinov3_large) echo "dinov3_large 4096 150 32  8192" ;;
    prithvi)      echo "prithvi      1024 100 32  8192" ;;
    prithvi_l5)   echo "prithvi_l5   1024 100 32  8192" ;;  # Landsat 5 2008–2010 tiles
    *) echo "Unknown model preset: $1" >&2; exit 1 ;;
  esac
}

has_step()      { [[ ",$STEPS_ARG,"     == *",$1,"* ]]; }
has_overwrite() { [[ ",$OVERWRITE_ARG," == *",$1,"* ]]; }

# should_run STEP [OUTPUT_PATH]  → 0=run, 1=skip
should_run() {
  local step=$1 path=${2:-}
  has_overwrite "$step" && return 0
  [ -z "$path" ] || [ ! -e "$path" ] && return 0
  return 1
}

# ── Expand comma-separated args into arrays ───────────────────────────────────
IFS=',' read -ra MODELS   <<< "$MODELS_ARG"
IFS=',' read -ra OUTCOMES <<< "$OUTCOMES_ARG"

echo "============================================================"
echo " NEXIS pipeline"
echo "  models   : ${MODELS[*]}"
echo "  outcomes : ${OUTCOMES[*]}"
echo "  steps    : $STEPS_ARG"
echo "  overwrite: ${OVERWRITE_ARG:-none}"
echo "============================================================"
echo ""

# ── Step 1: Train (sequential — GPU-bound) ────────────────────────────────────
if has_step train; then
  for MODEL_KEY in "${MODELS[@]}"; do
    read -r MODEL HIDDEN_DIM EPOCHS EXTRACT_BATCH SAE_BATCH <<< "$(model_params "$MODEL_KEY")"
    [ -n "$EPOCHS_OVERRIDE"        ] && EPOCHS="$EPOCHS_OVERRIDE"
    [ -n "$EXTRACT_BATCH_OVERRIDE" ] && EXTRACT_BATCH="$EXTRACT_BATCH_OVERRIDE"
    [ -n "$SAE_BATCH_OVERRIDE"     ] && SAE_BATCH="$SAE_BATCH_OVERRIDE"

    FEAT_FILE="$PROJECT_ROOT/results/uganda/${MODEL}_${HIDDEN_DIM}/individual_features.npz"

    if should_run train "$FEAT_FILE"; then
      echo "-- [$MODEL] Step 1: Embedding extraction + SAE training --------"
      $PYTHON "$SCRIPTS/train.py" \
        --model               "$MODEL"         \
        --hidden-dim          "$HIDDEN_DIM"    \
        --l1-coeff            "$L1_COEFF"      \
        --epochs              "$EPOCHS"        \
        --extract-batch-size  "$EXTRACT_BATCH" \
        --num-workers         "$NUM_WORKERS"   \
        --batch-size          "$SAE_BATCH"
    else
      echo "-- [$MODEL] Step 1: Skipping (individual_features.npz exists) --"
    fi
    echo ""
  done
fi

# ── Steps 2–5: one job per (model, outcome), parallelized when >1 ─────────────
# run_analyze MODEL_KEY OUTCOME  — NEXIS selection only (safe to parallelize)
run_analyze() {
  local MODEL_KEY=$1 OUTCOME=$2
  local MODEL HIDDEN_DIM EPOCHS EXTRACT_BATCH SAE_BATCH
  read -r MODEL HIDDEN_DIM EPOCHS EXTRACT_BATCH SAE_BATCH <<< "$(model_params "$MODEL_KEY")"

  local OUT_DIR="$PROJECT_ROOT/results/uganda/${MODEL}_${HIDDEN_DIM}"
  local TAG="[$MODEL | $OUTCOME]"

  if has_step analyze; then
    if should_run analyze "$OUT_DIR/${OUTCOME}/nexis_result.json"; then
      echo "-- $TAG Step 2: NEXIS feature selection ----------------------"
      $PYTHON "$SCRIPTS/analyze.py" \
        --embed-model "$MODEL"      \
        --sae-dim     "$HIDDEN_DIM" \
        --alpha       "$ALPHA"      \
        --max-steps   "$MAX_STEPS"  \
        --outcome     "$OUTCOME"    \
        $ANALYZE_EXTRA_ARGS
    else
      echo "-- $TAG Step 2: Skipping (${OUTCOME}/nexis_result.json exists)"
    fi
    echo ""
  fi
}

# run_summarize_plot MODEL_KEY OUTCOME  — summarize + plot (safe to parallelize;
# runs after interpret so VLM labels appear in the narrative)
run_summarize_plot() {
  local MODEL_KEY=$1 OUTCOME=$2
  local MODEL HIDDEN_DIM EPOCHS EXTRACT_BATCH SAE_BATCH
  read -r MODEL HIDDEN_DIM EPOCHS EXTRACT_BATCH SAE_BATCH <<< "$(model_params "$MODEL_KEY")"

  local OUT_DIR="$PROJECT_ROOT/results/uganda/${MODEL}_${HIDDEN_DIM}"
  local TAG="[$MODEL | $OUTCOME]"

  IFS=',' read -ra PIPELINES <<< "$PIPELINES_ARG"

  for PIPELINE in "${PIPELINES[@]}"; do
    if has_step summarize; then
      if should_run summarize "$OUT_DIR/${OUTCOME}/${PIPELINE}/summary.json"; then
        echo "-- $TAG Step 4 ($PIPELINE): Results summary ----------------"
        $PYTHON "$SCRIPTS/summarize.py" \
          --embed-model "$MODEL"        \
          --sae-dim     "$HIDDEN_DIM"   \
          --outcome     "$OUTCOME"      \
          --pipeline    "$PIPELINE"
      else
        echo "-- $TAG Step 4 ($PIPELINE): Skipping (${PIPELINE}/summary.json exists)"
      fi
      echo ""
    fi

    if has_step plot; then
      if should_run plot "$OUT_DIR/${OUTCOME}/${PIPELINE}/summary_illustration.png"; then
        echo "-- $TAG Step 5 ($PIPELINE): Feature image plots ------------"
        $PYTHON "$SCRIPTS/plot_features.py" \
          --embed-model "$MODEL"            \
          --sae-dim     "$HIDDEN_DIM"       \
          --k 8                             \
          --outcome     "$OUTCOME"          \
          --pipeline    "$PIPELINE"
      else
        echo "-- $TAG Step 5 ($PIPELINE): Skipping (${PIPELINE}/summary_illustration.png exists)"
      fi
      echo ""
    fi
  done
}

# run_interpret_serial MODEL_KEY  — interpretation step: runs each pipeline in sequence.
# Each pipeline loads its model(s) once and processes all outcomes before unloading.
# Must NOT be parallelized — single GPU.
run_interpret_serial() {
  local MODEL_KEY=$1
  local MODEL HIDDEN_DIM EPOCHS EXTRACT_BATCH SAE_BATCH
  read -r MODEL HIDDEN_DIM EPOCHS EXTRACT_BATCH SAE_BATCH <<< "$(model_params "$MODEL_KEY")"

  local OUT_DIR="$PROJECT_ROOT/results/uganda/${MODEL}_${HIDDEN_DIM}"
  local TAG="[$MODEL]"

  if ! has_step interpret; then return; fi

  IFS=',' read -ra PIPELINES <<< "$PIPELINES_ARG"
  $PYTHON -m pip install -q accelerate bitsandbytes || true

  for PIPELINE in "${PIPELINES[@]}"; do
    # Collect outcomes that need this pipeline's interpretation
    local OUTCOMES_TO_RUN=()
    for OUTCOME in "${OUTCOMES[@]}"; do
      if should_run interpret "$OUT_DIR/${OUTCOME}/${PIPELINE}/interpretations.json"; then
        OUTCOMES_TO_RUN+=("$OUTCOME")
      else
        echo "-- $TAG [$OUTCOME] Step 3 ($PIPELINE): Skipping (${PIPELINE}/interpretations.json exists)"
      fi
    done

    if [ "${#OUTCOMES_TO_RUN[@]}" -eq 0 ]; then
      echo "-- $TAG Step 3 ($PIPELINE): All outcomes already interpreted"
      continue
    fi

    local OUTCOMES_CSV
    OUTCOMES_CSV=$(IFS=','; echo "${OUTCOMES_TO_RUN[*]}")
    echo "-- $TAG Step 3 ($PIPELINE): ${#OUTCOMES_TO_RUN[@]} outcome(s)"

    if [ "$PIPELINE" = "qwen" ]; then
      $PYTHON "$SCRIPTS/interpret.py" \
        --pipeline    qwen            \
        --embed-model "$MODEL"        \
        --sae-dim     "$HIDDEN_DIM"   \
        --k 10                        \
        --vlm-model   "$VLM_MODEL"    \
        --outcomes    "$OUTCOMES_CSV" \
        $(has_overwrite interpret && echo "--overwrite") \
        $INTERPRET_EXTRA_ARGS         \
      || echo "  WARNING: interpret.py (qwen) failed."
    else
      $PYTHON "$SCRIPTS/interpret.py" \
        --pipeline      geochat         \
        --embed-model   "$MODEL"        \
        --sae-dim       "$HIDDEN_DIM"   \
        --k 10                          \
        --geochat-model "$GEOCHAT_MODEL" \
        --text-model    "$TEXT_MODEL"   \
        --outcomes      "$OUTCOMES_CSV" \
        $(has_overwrite interpret && echo "--overwrite") \
        $INTERPRET_EXTRA_ARGS           \
      || echo "  WARNING: interpret.py (geochat) failed."
    fi
    echo ""
  done
}

_run_parallel() {
  local func=$1; shift
  local PIDS=()
  local n_combos=$(( ${#MODELS[@]} * ${#OUTCOMES[@]} ))
  for MODEL_KEY in "${MODELS[@]}"; do
    for OUTCOME in "${OUTCOMES[@]}"; do
      if [ "$n_combos" -gt 1 ]; then
        "$func" "$MODEL_KEY" "$OUTCOME" &
        PIDS+=($!)
      else
        "$func" "$MODEL_KEY" "$OUTCOME"
      fi
    done
  done
  local FAILED=0
  if [ "${#PIDS[@]}" -gt 0 ]; then
    for pid in "${PIDS[@]}"; do
      wait "$pid" || FAILED=1
    done
  fi
  if [ "$FAILED" -eq 1 ]; then
    echo "ERROR: one or more $func jobs failed." >&2; exit 1
  fi
}

# ── Phase A: NEXIS analysis (parallel across model×outcome) ───────────────────
_run_parallel run_analyze

# ── Phase B: VLM interpret (serial per model — VLM loaded once per model) ────
for MODEL_KEY in "${MODELS[@]}"; do
  run_interpret_serial "$MODEL_KEY"
done

# ── Phase C: summarize + plot (parallel; runs after interpret so VLM labels ──
#             are available in the narrative)
_run_parallel run_summarize_plot

echo "Done."
for MODEL_KEY in "${MODELS[@]}"; do
  read -r MODEL HIDDEN_DIM _ <<< "$(model_params "$MODEL_KEY")"
  for OUTCOME in "${OUTCOMES[@]}"; do
    echo "  results/uganda/${MODEL}_${HIDDEN_DIM}/${OUTCOME}/"
  done
done
