#!/bin/bash
set -euo pipefail

ALLOC_JOB_ID="${ALLOC_JOB_ID:-${1:-1118197}}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
PROJECT_DIR="${PROJECT_DIR:-${BASE}/OpenTAD_C3_CoarseClean_20260702}"
RUN_TAG="${RUN_TAG:-c3_current_runnable_model_zoo_gpu0_direct_alloc${ALLOC_JOB_ID}_$(date '+%Y%m%d_%H%M%S_%z')}"
RUN_C3_READERS="${RUN_C3_READERS:-0}"
LOG_DIR="${LOG_DIR:-${BASE}/projects/c3_lowres_action_probe/direct_launch_logs}"
LAUNCH_LOG="${LOG_DIR}/${RUN_TAG}.log"
PID_FILE="${LOG_DIR}/${RUN_TAG}.pid"

mkdir -p "${LOG_DIR}"

if ! squeue -j "${ALLOC_JOB_ID}" -h -t RUNNING >/dev/null 2>&1; then
  echo "Allocation job ${ALLOC_JOB_ID} is not RUNNING; refusing direct launch." >&2
  exit 41
fi

if [[ ! -x "${PROJECT_DIR}/scripts/run_c3_current_runnable_model_zoo_gpu0_20260702.sh" ]]; then
  chmod +x "${PROJECT_DIR}/scripts/run_c3_current_runnable_model_zoo_gpu0_20260702.sh"
fi

nohup srun \
  --jobid="${ALLOC_JOB_ID}" \
  --overlap \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  bash -lc "cd '${PROJECT_DIR}' && export CUDA_VISIBLE_DEVICES=0 RUN_TAG='${RUN_TAG}' RUN_C3_READERS='${RUN_C3_READERS}' NUM_WORKERS=2 OMP_NUM_THREADS=4 && bash scripts/run_c3_current_runnable_model_zoo_gpu0_20260702.sh" \
  >"${LAUNCH_LOG}" 2>&1 &

LAUNCH_PID="$!"
printf "%s\n" "${LAUNCH_PID}" > "${PID_FILE}"

echo "DIRECT_LAUNCH_STARTED"
echo "ALLOC_JOB_ID=${ALLOC_JOB_ID}"
echo "RUN_TAG=${RUN_TAG}"
echo "RUN_C3_READERS=${RUN_C3_READERS}"
echo "LAUNCH_PID=${LAUNCH_PID}"
echo "LAUNCH_LOG=${LAUNCH_LOG}"
echo "PID_FILE=${PID_FILE}"
