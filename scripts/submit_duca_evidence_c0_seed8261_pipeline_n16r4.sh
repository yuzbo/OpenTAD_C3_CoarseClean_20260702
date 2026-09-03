#!/usr/bin/env bash
source /etc/profile
set -euo pipefail

fail() { echo "[DUCA_EVIDENCE_C0_PIPELINE][FAIL] $*" >&2; exit 1; }

PROJECT_DIR="${DUCA_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
EXPECTED_COMMIT="${DUCA_EVIDENCE_EXPECTED_COMMIT:?DUCA_EVIDENCE_EXPECTED_COMMIT is required}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${DUCA_RUN_ROOT:-$BASE/experiments/duca_evidence_${EXPECTED_COMMIT:0:8}_seed8261}"
MAX_JOBS_IN_QUEUE="${DUCA_MAX_JOBS_IN_QUEUE:-14}"
LOG_DIR="$RUN_ROOT/slurm_logs"
REGISTRY="$RUN_ROOT/c0_seed8261_submission.tsv"

cd "$PROJECT_DIR"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail "expected commit must be a full lowercase SHA"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "checkout commit mismatch"
[[ -z "$(git status --porcelain)" ]] || fail "checkout must be clean"
[[ "$MAX_JOBS_IN_QUEUE" =~ ^[1-9][0-9]*$ ]] || fail "queue limit must be positive"
mkdir -p "$LOG_DIR"
[[ ! -e "$REGISTRY" ]] || fail "submission registry already exists: $REGISTRY"
printf 'stage\tjob_id\tdependency\tstatus\n' > "$REGISTRY"

wait_for_submission_slot() {
  local current
  while true; do
    current="$(squeue -u "$USER" -h | wc -l)"
    if (( current < MAX_JOBS_IN_QUEUE )); then
      return 0
    fi
    echo "Evidence queue has ${current}/${MAX_JOBS_IN_QUEUE} jobs; retrying in 60 seconds"
    sleep 60
  done
}

wait_for_terminal_success() {
  local job_id="$1"
  local stage="$2"
  local state
  while true; do
    state="$(sacct -j "$job_id" --format=State -n -X | awk 'NF {print $1; exit}')"
    case "$state" in
      COMPLETED)
        printf '%s\t%s\t%s\t%s\n' "$stage" "$job_id" "" "COMPLETED" >> "$REGISTRY"
        return 0
        ;;
      FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED)
        printf '%s\t%s\t%s\t%s\n' "$stage" "$job_id" "" "$state" >> "$REGISTRY"
        fail "$stage job $job_id ended in $state"
        ;;
      *)
        echo "Evidence $stage job $job_id is ${state:-UNKNOWN}; retrying in 60 seconds"
        sleep 60
        ;;
    esac
  done
}

wait_for_submission_slot
admission_job="$(sbatch --parsable \
  --output="$LOG_DIR/%x_%j.out" --error="$LOG_DIR/%x_%j.err" \
  --export=ALL,DUCA_REPO_ROOT="$PROJECT_DIR",DUCA_EVIDENCE_EXPECTED_COMMIT="$EXPECTED_COMMIT" \
  scripts/run_duca_evidence_recovery_admission_n16r4.sbatch)"
admission_job="${admission_job%%;*}"
printf 'admission\t%s\t%s\t%s\n' "$admission_job" "" "SUBMITTED" >> "$REGISTRY"
wait_for_terminal_success "$admission_job" admission

wait_for_submission_slot
train_job="$(sbatch --parsable --array=0 \
  --output="$LOG_DIR/%x_%A_%a.out" --error="$LOG_DIR/%x_%A_%a.err" \
  --export=ALL,DUCA_REPO_ROOT="$PROJECT_DIR",DUCA_EVIDENCE_EXPECTED_COMMIT="$EXPECTED_COMMIT",DUCA_RUN_ROOT="$RUN_ROOT",DUCA_SEEDS=8261 \
  scripts/run_duca_evidence_recovery_train_array_n16r4.sbatch)"
train_job="${train_job%%;*}"
printf 'c0_train\t%s\t%s\t%s\n' "$train_job" "afterok:$admission_job" "SUBMITTED" >> "$REGISTRY"

wait_for_submission_slot
eval_job="$(sbatch --parsable --dependency="afterok:$train_job" --array=0 \
  --output="$LOG_DIR/%x_%A_%a.out" --error="$LOG_DIR/%x_%A_%a.err" \
  --export=ALL,DUCA_REPO_ROOT="$PROJECT_DIR",DUCA_EVIDENCE_EXPECTED_COMMIT="$EXPECTED_COMMIT",DUCA_RUN_ROOT="$RUN_ROOT",DUCA_SEEDS=8261 \
  scripts/run_duca_evidence_recovery_eval_array_n16r4.sbatch)"
eval_job="${eval_job%%;*}"
printf 'c0_eval\t%s\t%s\t%s\n' "$eval_job" "afterok:$train_job" "SUBMITTED" >> "$REGISTRY"
printf 'Evidence C0 seed 8261 submitted: admission=%s train=%s eval=%s\n' "$admission_job" "$train_job" "$eval_job"
