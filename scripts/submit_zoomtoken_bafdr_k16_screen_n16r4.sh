#!/usr/bin/env bash
source /etc/profile
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
MAX_JOBS_IN_QUEUE="${BAFDR_MAX_JOBS_IN_QUEUE:-14}"
SEED=4407
ARMS="G96,U16-UNIFORM-A0,BAFDR-K16-LATE,BAFDR-K16-NOKD,BAFDR-K16-FULL"
TEACHER_CHECKPOINT="${BAFDR_TEACHER_CHECKPOINT:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2;;
    --arms) ARMS="$2"; shift 2;;
    --teacher-checkpoint) TEACHER_CHECKPOINT="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
	done

cd "$PROJECT_DIR"
EXPECTED_COMMIT="${BAFDR_EXPECTED_COMMIT:?BAFDR_EXPECTED_COMMIT must be the full 40-character target SHA}"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]] || { echo "BAFDR_EXPECTED_COMMIT must be a full SHA" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || { echo "BAFDR checkout HEAD mismatch" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "BAFDR checkout is not clean" >&2; exit 2; }
[[ "$MAX_JOBS_IN_QUEUE" =~ ^[1-9][0-9]*$ ]] || { echo "BAFDR_MAX_JOBS_IN_QUEUE must be positive" >&2; exit 2; }
SCREEN_RECEIPT="${BAFDR_SCREEN_RECEIPT:-${BASE}/projects/bafdr_k16_fullmatrix_compute/manifest/screen_receipt.json}"
TEACHER_CONFIG="${BAFDR_TEACHER_CONFIG:-configs/adatad/thumos/bafdr_k16_d160_seed${SEED}.py}"
TEACHER_CONFIG_SHA256=""
TEACHER_CHECKPOINT_SHA256=""
TEACHER_COMMIT="${BAFDR_TEACHER_COMMIT:-}"
IFS=',' read -r -a arm_list <<< "$ARMS"
declare -a screen_job_ids=()
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
for arm in "${arm_list[@]}"; do
  case "$arm" in
    U16-UNIFORM-A0) slug=u16_uniform_a0 ;;
    BAFDR-K16-LATE) slug=late ;;
    BAFDR-K16-NOKD) slug=nokd ;;
    BAFDR-K16-FULL) slug=full ;;
    G96) slug=g96 ;;
    *) echo "unsupported screen arm: $arm" >&2; exit 2 ;;
  esac
  cfg="configs/adatad/thumos/bafdr_k16_${slug}_seed${SEED}.py"
  [[ -f "$cfg" ]] || { echo "missing config: $cfg" >&2; exit 1; }
  if [[ "$arm" == BAFDR-K16-FULL ]]; then
    if [[ -z "$TEACHER_CHECKPOINT" || ! -f "$TEACHER_CHECKPOINT" ]]; then
      echo "BAFDR-K16-FULL blocked: provide an existing terminal D160 teacher checkpoint via --teacher-checkpoint or BAFDR_TEACHER_CHECKPOINT" >&2
      exit 2
    fi
    [[ -f "$TEACHER_CONFIG" ]] || { echo "missing teacher config: $TEACHER_CONFIG" >&2; exit 2; }
    [[ "$TEACHER_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]] || { echo "BAFDR-K16-FULL requires BAFDR_TEACHER_COMMIT (full SHA)" >&2; exit 2; }
    TEACHER_CHECKPOINT_SHA256="$(sha256sum "$TEACHER_CHECKPOINT" | awk '{print $1}')"
    TEACHER_CONFIG_SHA256="$(sha256sum "$TEACHER_CONFIG" | awk '{print $1}')"
    python - "$TEACHER_CHECKPOINT" <<'PY'
import sys, torch
path = sys.argv[1]
state = torch.load(path, map_location="cpu")
if not isinstance(state, dict) or "state_dict_ema" not in state:
    raise SystemExit("teacher checkpoint must contain state_dict_ema")
epoch = state.get("epoch", state.get("meta", {}).get("epoch", None))
if epoch not in (59, "59"):
    raise SystemExit(f"teacher checkpoint epoch must be 59, got {epoch!r}")
print(f"[PRECHECK] terminal teacher={path} epoch=59 state_dict_ema=present")
PY
  fi
  wait_for_submission_slot
  job_id="$(sbatch --parsable --partition=gpu --account=sczc063 --qos=normal --gres=gpu:2 --cpus-per-task=8 --time=72:00:00 \
    --job-name="bafdr-${slug}-s${SEED}" \
    --output="${BASE}/slurm_logs/%x_%j.out" --error="${BASE}/slurm_logs/%x_%j.err" \
    --wrap="source /etc/profile; set -euo pipefail; module load cuda/11.8; module load miniforge3/24.11; source ${BASE}/conda_envs/opentad/bin/activate; cd \"${PROJECT_DIR}\"; BAFDR_REQUIRE_SCREEN_GATE=0 BAFDR_EXPECTED_COMMIT=${EXPECTED_COMMIT} BAFDR_SCREEN_RECEIPT=\"${SCREEN_RECEIPT}\" BAFDR_TEACHER_CHECKPOINT=\"${TEACHER_CHECKPOINT}\" BAFDR_TEACHER_CONFIG=\"${TEACHER_CONFIG}\" BAFDR_TEACHER_CHECKPOINT_SHA256=${TEACHER_CHECKPOINT_SHA256} BAFDR_TEACHER_CONFIG_SHA256=${TEACHER_CONFIG_SHA256} BAFDR_TEACHER_COMMIT=${TEACHER_COMMIT} bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh train \"${cfg}\"")"
  screen_job_ids+=("${job_id}")
  echo "${arm}=${job_id}"
  done

mkdir -p "$(dirname "${SCREEN_RECEIPT}")"
job_ids_csv="$(IFS=,; printf '%s' "${screen_job_ids[*]}")"
python - "${SCREEN_RECEIPT}" "${EXPECTED_COMMIT}" "${SEED}" "${ARMS}" "${TEACHER_CONFIG}" "${TEACHER_CONFIG_SHA256}" "${TEACHER_CHECKPOINT}" "${TEACHER_CHECKPOINT_SHA256}" "${TEACHER_COMMIT}" "${job_ids_csv}" <<'PY'
import json
import sys
import time

path, commit, seed, arms, teacher_cfg, teacher_cfg_sha, teacher_ckpt, teacher_ckpt_sha, teacher_commit, job_ids = sys.argv[1:]
payload = {
    "schema_version": "ZOOMTOKEN-BAFDR-SCREEN-RECEIPT-v001",
    # Submission is not a scientific gate.  The finalizer promotes this
    # receipt to PASS only after every screen arm has a valid terminal receipt.
    "status": "SUBMITTED",
    "commit_sha": commit,
    "seed": int(seed),
    "arms": [arm for arm in arms.split(",") if arm],
    "job_ids": [job for job in job_ids.split(",") if job],
    "teacher": {
        "config": teacher_cfg or None,
        "config_sha256": teacher_cfg_sha or None,
        "checkpoint": teacher_ckpt or None,
        "checkpoint_sha256": teacher_ckpt_sha or None,
        "commit": teacher_commit or None,
    },
    "timestamp": time.time(),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY
