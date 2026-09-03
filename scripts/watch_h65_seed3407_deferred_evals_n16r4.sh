#!/usr/bin/env bash
source /etc/profile
set -euo pipefail

fail() { echo "[H65_DEFERRED_EVAL][FAIL] $*" >&2; exit 1; }

PROJECT_DIR="${H65_PRO_PROJECT_DIR:?H65_PRO_PROJECT_DIR is required}"
EXPECTED_COMMIT="${H65_PRO_EXPECTED_COMMIT:?H65_PRO_EXPECTED_COMMIT is required}"
WORK_ROOT="${H65_PRO_WORK_ROOT:?H65_PRO_WORK_ROOT is required}"
SUBMISSION_DIR="${H65_PRO_SUBMISSION_DIR:?H65_PRO_SUBMISSION_DIR is required}"
MAX_JOBS_IN_QUEUE="${H65_PRO_MAX_JOBS_IN_QUEUE:-14}"
EVAL_SCRIPT="$PROJECT_DIR/tools/experiments/run_h65_pro_eval.sbatch"
LOG_DIR="$SUBMISSION_DIR/logs"
REGISTRY="$SUBMISSION_DIR/deferred_eval_registry.csv"

[[ "$(git -C "$PROJECT_DIR" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "checkout commit mismatch"
[[ -z "$(git -C "$PROJECT_DIR" status --porcelain)" ]] || fail "checkout must be clean"
[[ -r "$EVAL_SCRIPT" ]] || fail "evaluation sbatch is missing"
[[ "$MAX_JOBS_IN_QUEUE" =~ ^[1-9][0-9]*$ ]] || fail "queue limit must be positive"
mkdir -p "$LOG_DIR"
[[ ! -e "$REGISTRY" ]] || fail "deferred registry already exists: $REGISTRY"
printf 'experiment_id,train_job_id,eval_job_id,status\n' > "$REGISTRY"

entries=(
  "REF-D768|1267709|configs/adatad/thumos/h65_pro/h65_pro_ref_d768.py|h65_pro_ref_d768"
  "REF-U384|1267711|configs/adatad/thumos/h65_pro/h65_pro_ref_u384.py|h65_pro_ref_u384"
  "REF-MNV3FC384|1267737|configs/adatad/thumos/h65_pro/h65_pro_ref_mnv3fc384.py|h65_pro_ref_mnv3fc384"
)

wait_for_train_success() {
  local job_id="$1"
  local state
  while true; do
    state="$(sacct -j "$job_id" --format=State -n -X | awk 'NF {print $1; exit}')"
    case "$state" in
      COMPLETED) return 0 ;;
      FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED)
        fail "training job $job_id ended in $state"
        ;;
      *) sleep 60 ;;
    esac
  done
}

wait_for_submission_slot() {
  local current
  while true; do
    current="$(squeue -u "$USER" -h | wc -l)"
    if (( current < MAX_JOBS_IN_QUEUE )); then
      return 0
    fi
    sleep 60
  done
}

for entry in "${entries[@]}"; do
  IFS='|' read -r experiment_id train_job_id config variant <<< "$entry"
  wait_for_train_success "$train_job_id"
  wait_for_submission_slot
  eval_job="$(sbatch --parsable \
    --job-name="h65p-ev-${experiment_id}-3407-deferred" \
    --output="$LOG_DIR/%x-%j.out" --error="$LOG_DIR/%x-%j.err" \
    --export=ALL,H65_PRO_EXPECTED_COMMIT="$EXPECTED_COMMIT",H65_PRO_WORK_ROOT="$WORK_ROOT",DUCA_REPO_ROOT="$PROJECT_DIR" \
    "$EVAL_SCRIPT" "$PROJECT_DIR/$config" 3407 "$experiment_id" "$variant")"
  eval_job="${eval_job%%;*}"
  printf '%s,%s,%s,%s\n' "$experiment_id" "$train_job_id" "$eval_job" "SUBMITTED_AFTER_TRAIN_SUCCESS" >> "$REGISTRY"
done
