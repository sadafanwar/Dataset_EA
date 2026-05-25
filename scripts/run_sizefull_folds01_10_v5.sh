#!/usr/bin/env bash
set -euo pipefail

EXP="qwen35_9b_qlora_train_size_undersampling_10fold_cv_inner_val_v2"
SIZE="size_full"

BASE_RESULTS="results/${EXP}/${SIZE}"
BASE_OUTPUTS="outputs/${EXP}/${SIZE}"

RUN_LOG_DIR="results/${EXP}/logs"
PROGRESS="results/${EXP}/size_full_fold_progress_v5.csv"
STEP_LOG="results/${EXP}/size_full_step_log_v5.csv"

mkdir -p "results/${EXP}"
mkdir -p "${RUN_LOG_DIR}"

if [ ! -f "$PROGRESS" ]; then
  echo "train_size,fold,status,start_time_utc,end_time_utc,duration_seconds,train_file,config_file,model_dir,test_file,metrics_file,predictions_file,accuracy,trace_precision,trace_recall,trace_f1,no_trace_precision,no_trace_recall,no_trace_f1,valid_predictions,invalid_predictions,notes" > "$PROGRESS"
fi

if [ ! -f "$STEP_LOG" ]; then
  echo "train_size,fold,step,status,start_time_utc,end_time_utc,duration_seconds,artifact_path,notes" > "$STEP_LOG"
fi

log_step () {
  echo "$1,$2,$3,$4,$5,$6,$7,$8,$9" >> "$STEP_LOG"
}

run_step () {
  local train_size="$1"
  local fold="$2"
  local step="$3"
  local artifact="$4"
  local notes="$5"
  shift 5

  local start_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  local start_sec=$(date +%s)

  echo "----------------------------------------------------------------------"
  echo "STEP: ${step} | ${train_size} | ${fold}"
  echo "----------------------------------------------------------------------"

  "$@"

  local end_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  local end_sec=$(date +%s)
  local dur=$((end_sec - start_sec))

  log_step "$train_size" "$fold" "$step" "completed" "$start_iso" "$end_iso" "$dur" "$artifact" "$notes"
}

for FOLD_NUM in 01 02 03 04 05 06 07 08 09 10; do
  FOLD="fold_${FOLD_NUM}"
  TRAIN_SIZE="full"

  echo "======================================================================"
  echo "STARTING ${SIZE} ${FOLD}"
  echo "======================================================================"

  FOLD_START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  FOLD_START_SEC=$(date +%s)

  TRAIN_FILE="data/qwen35_oversampling_10fold_cv/inner_splits/${FOLD}/train_size_full_undersampled.jsonl"
  CONFIG="src/integration/training_scripts/ft/config_qwen35_9b_train_size_undersampling_v2_size_full_${FOLD}.yaml"
  MODEL_DIR="${BASE_OUTPUTS}/${FOLD}/final_model"
  TEST_FILE="data/qwen35_oversampling_10fold_cv/folds/${FOLD}/test.jsonl"
  OUT_DIR="${BASE_RESULTS}/${FOLD}"
  METRICS_FILE="${OUT_DIR}/${FOLD}_metrics.json"
  PRED_FILE="${OUT_DIR}/${FOLD}_predictions.csv"
  TRAIN_LOG="${RUN_LOG_DIR}/size_full_${FOLD}_train.log"
  EVAL_LOG="${RUN_LOG_DIR}/size_full_${FOLD}_eval.log"

  run_step "$TRAIN_SIZE" "$FOLD" "subset_created" "$TRAIN_FILE" "Create balanced undersampled train subset with maximum balanced trace and no_trace samples" \
    python scripts/create_train_size_undersampled_subset.py \
      --fold ${FOLD_NUM#0} \
      --train_size full

  run_step "$TRAIN_SIZE" "$FOLD" "config_created" "$CONFIG" "Create v2 QLoRA config pointing to size_full train file and v2 output directory" \
    python scripts/create_qwen35_train_size_undersampling_config_v2.py \
      --fold ${FOLD_NUM#0} \
      --train_size full

  echo "Checking config for ${FOLD}:"
  grep -n "model_path\|dataset_name\|eval_dataset\|output_dir" "$CONFIG"

  if [ -d "$MODEL_DIR" ]; then
    STEP_START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    STEP_END_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    log_step "$TRAIN_SIZE" "$FOLD" "training_skipped_existing_model" "completed" "$STEP_START_ISO" "$STEP_END_ISO" "0" "$MODEL_DIR" "final_model already exists; training skipped"
    TRAIN_STATUS="skipped_existing_model"
  else
    STEP_START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    STEP_START_SEC=$(date +%s)
    log_step "$TRAIN_SIZE" "$FOLD" "training_started" "started" "$STEP_START_ISO" "$STEP_START_ISO" "0" "$CONFIG" "Training started with QLoRA"

    python src/integration/training_scripts/ft/run_ft.py \
      --config "$CONFIG" \
      --method qlora 2>&1 | tee "$TRAIN_LOG"

    STEP_END_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    STEP_END_SEC=$(date +%s)
    STEP_DUR=$((STEP_END_SEC - STEP_START_SEC))

    log_step "$TRAIN_SIZE" "$FOLD" "training_completed" "completed" "$STEP_START_ISO" "$STEP_END_ISO" "$STEP_DUR" "$MODEL_DIR" "Training completed; log=${TRAIN_LOG}"
    TRAIN_STATUS="completed"
  fi

  if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: expected model directory not found: $MODEL_DIR"
    exit 1
  fi

  mkdir -p "$OUT_DIR"

  STEP_START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  STEP_START_SEC=$(date +%s)
  log_step "$TRAIN_SIZE" "$FOLD" "evaluation_started" "started" "$STEP_START_ISO" "$STEP_START_ISO" "0" "$MODEL_DIR" "Evaluation started using forced-choice v5"

  python scripts/evaluate_qwen35_fold_classification_forced_choice_v5.py \
    --model_dir "$MODEL_DIR" \
    --test_file "$TEST_FILE" \
    --output_dir "$OUT_DIR" \
    --max_length 1024 2>&1 | tee "$EVAL_LOG"

  STEP_END_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  STEP_END_SEC=$(date +%s)
  STEP_DUR=$((STEP_END_SEC - STEP_START_SEC))

  log_step "$TRAIN_SIZE" "$FOLD" "evaluation_completed" "completed" "$STEP_START_ISO" "$STEP_END_ISO" "$STEP_DUR" "$METRICS_FILE" "Evaluation completed; log=${EVAL_LOG}"

  python - << PY
import json
from pathlib import Path

p = Path("${METRICS_FILE}")
m = json.loads(p.read_text())

print("Validation check for ${FOLD}")
print("total_records:", m["total_records"])
print("valid_predictions:", m["valid_predictions"])
print("invalid_predictions:", m["invalid_predictions"])
print("accuracy:", m["accuracy"])
print("trace_recall:", m["trace_recall"])

assert m["invalid_predictions"] == 0, "Invalid predictions found"
assert m["valid_predictions"] == m["total_records"], "Not all predictions valid"
PY

  STEP_START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  STEP_END_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  log_step "$TRAIN_SIZE" "$FOLD" "validation_passed" "completed" "$STEP_START_ISO" "$STEP_END_ISO" "0" "$METRICS_FILE" "valid_predictions equals total_records and invalid_predictions equals 0"

  FOLD_END_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  FOLD_END_SEC=$(date +%s)
  FOLD_DUR=$((FOLD_END_SEC - FOLD_START_SEC))

  python - << PY
import json
from pathlib import Path

m = json.loads(Path("${METRICS_FILE}").read_text())

line = (
    "full,"
    "${FOLD},"
    "completed,"
    "${FOLD_START_ISO},"
    "${FOLD_END_ISO},"
    "${FOLD_DUR},"
    "${TRAIN_FILE},"
    "${CONFIG},"
    "${MODEL_DIR},"
    "${TEST_FILE},"
    "${METRICS_FILE},"
    "${PRED_FILE},"
    f"{m['accuracy']},"
    f"{m['trace_precision']},"
    f"{m['trace_recall']},"
    f"{m['trace_f1']},"
    f"{m['no_trace_precision']},"
    f"{m['no_trace_recall']},"
    f"{m['no_trace_f1']},"
    f"{m['valid_predictions']},"
    f"{m['invalid_predictions']},"
    "training_${TRAIN_STATUS}_evaluation_v5"
)

with open("${PROGRESS}", "a", encoding="utf-8") as f:
    f.write(line + "\\n")
PY

  STEP_START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  STEP_END_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  log_step "$TRAIN_SIZE" "$FOLD" "fold_completed" "completed" "$STEP_START_ISO" "$STEP_END_ISO" "0" "$METRICS_FILE" "Fold completed and recorded in progress CSV"

  echo "COMPLETED ${SIZE} ${FOLD}"
done

echo "All requested size_full folds 01-10 completed."
echo "Fold progress: ${PROGRESS}"
echo "Step log: ${STEP_LOG}"
