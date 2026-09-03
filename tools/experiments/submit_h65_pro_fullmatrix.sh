#!/usr/bin/env bash
source /etc/profile
set -euo pipefail

fail() { echo "[H65_PRO_SUBMIT][FAIL] $*" >&2; exit 1; }

ROOT="${DUCA_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
ROOT="$(cd -- "$ROOT" && pwd -P)"
cd "$ROOT"
[[ -f tools/bata/validate_h65_pro_fullmatrix.py ]] || fail "not at H65-Pro repo root: $ROOT"

module load cuda/11.8
module load miniforge3/24.11
ENV_ROOT="/data/run01/sczc063/yuzibo/conda_envs/opentad"
[[ -f "$ENV_ROOT/bin/activate" ]] || fail "canonical environment missing: $ENV_ROOT"
source "$ENV_ROOT/bin/activate"
PYTHON="${PYTHON:-python}"

export DUCA_CELLCF_TRAINING_PROFILE=official60
export YUZIBO_ROOT="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
export THUMOS14_ANNOTATION_PATH="${THUMOS14_ANNOTATION_PATH:-$YUZIBO_ROOT/thumos14/annotations/thumos_14_anno.json}"
export THUMOS14_CLASS_MAP="${THUMOS14_CLASS_MAP:-$YUZIBO_ROOT/thumos14/annotations/category_idx.txt}"
export THUMOS14_TRAIN_DATA_PATH="${THUMOS14_TRAIN_DATA_PATH:-$YUZIBO_ROOT/thumos14/train}"
export THUMOS14_TEST_DATA_PATH="${THUMOS14_TEST_DATA_PATH:-$YUZIBO_ROOT/thumos14/test}"
export ADATAD_PRETRAIN="${ADATAD_PRETRAIN:-$YUZIBO_ROOT/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"

for path in "$THUMOS14_ANNOTATION_PATH" "$THUMOS14_CLASS_MAP" "$THUMOS14_TRAIN_DATA_PATH" "$THUMOS14_TEST_DATA_PATH" "$ADATAD_PRETRAIN"; do
  [[ -r "$path" ]] || fail "canonical path is not readable: $path"
done

[[ -n "${H65_PRO_EXPECTED_COMMIT:-}" ]] || fail "H65_PRO_EXPECTED_COMMIT is required for H65-Pro submission"
COMMIT="$H65_PRO_EXPECTED_COMMIT"
[[ "$(git rev-parse HEAD)" == "$COMMIT" ]] || fail "checkout commit differs from H65_PRO_EXPECTED_COMMIT"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "submission requires a clean exact-commit checkout"
export H65_PRO_EXPECTED_COMMIT="$COMMIT"

MATRIX="${H65_PRO_MATRIX:-docs/experiments/h65_pro_fullmatrix_20260902/03_EXPERIMENT_MATRIX.csv}"
TRAIN_SCRIPT="${H65_PRO_TRAIN_SCRIPT:-tools/experiments/run_h65_pro_train.sbatch}"
EVAL_SCRIPT="${H65_PRO_EVAL_SCRIPT:-tools/experiments/run_h65_pro_eval.sbatch}"
MAX_JOBS_IN_QUEUE="${H65_PRO_MAX_JOBS_IN_QUEUE:-14}"
[[ -r "$MATRIX" ]] || fail "matrix is not readable: $MATRIX"
[[ -r "$TRAIN_SCRIPT" && -r "$EVAL_SCRIPT" ]] || fail "train/eval sbatch scripts are missing"
[[ "$MAX_JOBS_IN_QUEUE" =~ ^[1-9][0-9]*$ ]] || fail "H65_PRO_MAX_JOBS_IN_QUEUE must be a positive integer"
(( MAX_JOBS_IN_QUEUE >= 2 )) || fail "H65_PRO_MAX_JOBS_IN_QUEUE must allow at least two submissions"

"$PYTHON" tools/bata/validate_h65_pro_fullmatrix.py --matrix "$MATRIX"
"$PYTHON" -m py_compile tools/train.py tools/test.py \
  tools/bata/duca_selected_axis_training.py \
  tools/bata/generate_h65_pro_fullmatrix.py \
  tools/bata/validate_h65_pro_fullmatrix.py

SMOKE_IDS="REF-U384 F01 F02 F03 F05 F09 F13 F16"
should_submit() {
  local experiment_id="$1"
  if [[ "${H65_PRO_SMOKE_ONLY:-0}" == 1 ]]; then
    [[ " $SMOKE_IDS " == *" $experiment_id "* ]]
    return
  fi
  local filter="${H65_PRO_EXPERIMENT_FILTER:-}"
  if [[ -n "$filter" ]]; then
    [[ " $filter " == *" $experiment_id "* ]]
    return
  fi
  return 0
}

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  while IFS=, read -r experiment_id category phase ct mod taylor curriculum frames seed config variant train_command eval_command train_job_id eval_job_id status; do
    [[ "$experiment_id" == "experiment_id" || -z "$experiment_id" ]] && continue
    should_submit "$experiment_id" || continue
    PRECHECK_ONLY=1 bash "$TRAIN_SCRIPT" "$config" "$seed" "$experiment_id" "$variant"
    PRECHECK_ONLY=1 bash "$EVAL_SCRIPT" "$config" "$seed" "$experiment_id" "$variant"
  done < "$MATRIX"
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || fail "sbatch is not available in this environment"

wait_for_submission_slots() {
  local current
  while true; do
    current="$(squeue -u "$USER" -h | wc -l)"
    if (( current <= MAX_JOBS_IN_QUEUE - 2 )); then
      return 0
    fi
    echo "H65-Pro queue has ${current}/${MAX_JOBS_IN_QUEUE} jobs; retrying in 60 seconds"
    sleep 60
  done
}

SUBMISSION_DIR="${H65_PRO_SUBMISSION_DIR:-$YUZIBO_ROOT/h65_pro_fullmatrix_20260902_submission/$COMMIT}"
LOG_DIR="$SUBMISSION_DIR/logs"
mkdir -p "$LOG_DIR"
REGISTRY="$SUBMISSION_DIR/submission_registry.csv"
printf 'experiment_id,category,seed,config,variant,train_job_id,eval_job_id,train_dependency,eval_dependency,status\n' > "$REGISTRY"

count=0
while IFS=, read -r experiment_id category phase ct mod taylor curriculum frames seed config variant train_command eval_command train_job_id eval_job_id status; do
  [[ "$experiment_id" == "experiment_id" || -z "$experiment_id" ]] && continue
  should_submit "$experiment_id" || continue
  wait_for_submission_slots
  train_job_name="h65p-tr-${experiment_id}-${seed}"
  eval_job_name="h65p-ev-${experiment_id}-${seed}"
  if ! train_job_id="$(sbatch --parsable \
    --job-name="$train_job_name" \
    --output="$LOG_DIR/%x-%j.out" \
    --error="$LOG_DIR/%x-%j.err" \
    --export=ALL,H65_PRO_EXPECTED_COMMIT="$COMMIT" \
    "$TRAIN_SCRIPT" "$config" "$seed" "$experiment_id" "$variant")"; then
    fail "train submission failed for $experiment_id seed $seed"
  fi
  train_dependency="${train_job_id%%;*}"
  eval_dependency="afterok:${train_dependency}"
  if ! eval_job_id="$(sbatch --parsable \
    --job-name="$eval_job_name" \
    --dependency=afterok:"$train_dependency" \
    --output="$LOG_DIR/%x-%j.out" \
    --error="$LOG_DIR/%x-%j.err" \
    --export=ALL,H65_PRO_EXPECTED_COMMIT="$COMMIT" \
    "$EVAL_SCRIPT" "$config" "$seed" "$experiment_id" "$variant")"; then
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$experiment_id" "$category" "$seed" "$config" "$variant" \
      "$train_job_id" "" "$train_dependency" "$eval_dependency" \
      "EVAL_SUBMIT_FAILED_TRAIN_CANCEL_REQUESTED" >> "$REGISTRY"
    scancel "$train_dependency" >/dev/null 2>&1 || true
    fail "eval submission failed for $experiment_id seed $seed; requested cancellation for train job $train_dependency"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$experiment_id" "$category" "$seed" "$config" "$variant" \
    "$train_job_id" "$eval_job_id" "$train_dependency" "$eval_dependency" "SUBMITTED" >> "$REGISTRY"
  count=$((count + 1))
done < "$MATRIX"

echo "Submitted $count H65-Pro train jobs with afterok eval jobs"
echo "Registry: $REGISTRY"
