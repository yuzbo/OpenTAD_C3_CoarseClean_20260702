#!/usr/bin/env bash
source /etc/profile
set -euo pipefail

fail() { echo "[BAFDR_SEED4407_PIPELINE][FAIL] $*" >&2; exit 1; }

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
EXPECTED_COMMIT="${BAFDR_EXPECTED_COMMIT:?BAFDR_EXPECTED_COMMIT is required}"
MAX_JOBS_IN_QUEUE="${BAFDR_MAX_JOBS_IN_QUEUE:-14}"
RUN_ROOT="${ZOOMTOKEN_RUN_ROOT:-$BASE/experiments/zoomtoken_bafdr_${EXPECTED_COMMIT:0:8}_seed4407}"
LOG_DIR="$RUN_ROOT/slurm_logs"
REGISTRY="$RUN_ROOT/seed4407_submission.tsv"
SCREEN_RECEIPT="$RUN_ROOT/manifest/screen_receipt.json"
TEACHER_CONFIG="configs/adatad/thumos/bafdr_k16_d160_seed4407.py"
TEACHER_CHECKPOINT="$RUN_ROOT/work_dirs/bafdr_k16_d160_seed4407/checkpoint/epoch_59.pth"

cd "$PROJECT_DIR"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail "expected commit must be a full lowercase SHA"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "checkout commit mismatch"
[[ -z "$(git status --porcelain)" ]] || fail "checkout must be clean"
[[ "$MAX_JOBS_IN_QUEUE" =~ ^[1-9][0-9]*$ ]] || fail "queue limit must be positive"
[[ -r "$TEACHER_CONFIG" ]] || fail "D160 teacher config is missing"
mkdir -p "$LOG_DIR" "$(dirname "$SCREEN_RECEIPT")"
[[ ! -e "$REGISTRY" ]] || fail "submission registry already exists: $REGISTRY"
printf 'stage\tjob_id\tdependency\tstatus\n' > "$REGISTRY"

wait_for_submission_slot() {
  local current
  while true; do
    current="$(squeue -u "$USER" -h | wc -l)"
    if (( current < MAX_JOBS_IN_QUEUE )); then
      return 0
    fi
    echo "BAFDR queue has ${current}/${MAX_JOBS_IN_QUEUE} jobs; retrying in 60 seconds"
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
        echo "BAFDR $stage job $job_id is ${state:-UNKNOWN}; retrying in 60 seconds"
        sleep 60
        ;;
    esac
  done
}

wait_for_submission_slot
gradient_job="$(sbatch --parsable \
  --output="$LOG_DIR/%x_%j.out" --error="$LOG_DIR/%x_%j.err" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",BAFDR_EXPECTED_COMMIT="$EXPECTED_COMMIT" \
  scripts/run_zoomtoken_bafdr_gradient_gate_n16r4.sbatch)"
gradient_job="${gradient_job%%;*}"
printf 'gradient_gate\t%s\t%s\t%s\n' "$gradient_job" "" "SUBMITTED" >> "$REGISTRY"
wait_for_terminal_success "$gradient_job" gradient_gate

wait_for_submission_slot
teacher_job="$(sbatch --parsable --partition=gpu --account=sczc063 --qos=normal \
  --gres=gpu:2 --cpus-per-task=8 --time=72:00:00 \
  --job-name=bafdr-d160-s4407 \
  --output="$LOG_DIR/%x_%j.out" --error="$LOG_DIR/%x_%j.err" \
  --wrap="source /etc/profile; set -euo pipefail; module load cuda/11.8; module load miniforge3/24.11; source ${BASE}/conda_envs/opentad/bin/activate; cd \"${PROJECT_DIR}\"; PROJECT_DIR=\"${PROJECT_DIR}\" ZOOMTOKEN_RUN_ROOT=\"${RUN_ROOT}\" BAFDR_REQUIRE_SCREEN_GATE=0 BAFDR_EXPECTED_COMMIT=${EXPECTED_COMMIT} bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh train \"${TEACHER_CONFIG}\"")"
teacher_job="${teacher_job%%;*}"
printf 'd160_teacher\t%s\t%s\t%s\n' "$teacher_job" "afterok:$gradient_job" "SUBMITTED" >> "$REGISTRY"
wait_for_terminal_success "$teacher_job" d160_teacher
[[ -r "$TEACHER_CHECKPOINT" ]] || fail "D160 terminal checkpoint is missing after job $teacher_job"

BAFDR_SCREEN_RECEIPT="$SCREEN_RECEIPT" \
BAFDR_TEACHER_CHECKPOINT="$TEACHER_CHECKPOINT" \
BAFDR_TEACHER_COMMIT="$EXPECTED_COMMIT" \
BAFDR_EXPECTED_COMMIT="$EXPECTED_COMMIT" \
BAFDR_MAX_JOBS_IN_QUEUE="$MAX_JOBS_IN_QUEUE" \
ZOOMTOKEN_RUN_ROOT="$RUN_ROOT" \
PROJECT_DIR="$PROJECT_DIR" \
bash scripts/submit_zoomtoken_bafdr_k16_screen_n16r4.sh --seed 4407

printf 'screen_submission\t%s\t%s\t%s\n' "" "afterok:$teacher_job" "SUBMITTED" >> "$REGISTRY"
echo "BAFDR seed-4407 screen submitted; receipt: $SCREEN_RECEIPT"
