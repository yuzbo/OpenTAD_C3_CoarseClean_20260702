#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_MUST_DYNAMIC_BACKEND][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
FULLTRAIN_CANDIDATE="${FULLTRAIN_CANDIDATE:-0}"
CONFIG="${CONFIG:-configs/adatad/thumos/duca_must_dynamic_official_adatad_backend_full_train.py}"
VALIDATOR="${VALIDATOR:-tools/bata/validate_duca_must_dynamic_official_adatad_backend.py}"
RUN_TAG="${RUN_TAG:-duca_must_dynamic_official_adatad_backend_$(date +%Y%m%d_%H%M%S_%z)}"
RUN_ID="${RUN_ID:-0}"
SEED="${SEED:-0}"
MASTER_PORT="${MASTER_PORT:-30271}"

export DUCA_MUST_DENSE_WINDOW_SIZE="${DUCA_MUST_DENSE_WINDOW_SIZE:-768}"
export DUCA_MUST_BUDGET_MAX="${DUCA_MUST_BUDGET_MAX:-384}"
export DUCA_MUST_BUDGET_MIN="${DUCA_MUST_BUDGET_MIN:-64}"
export DUCA_MUST_BUDGET_TARGET="${DUCA_MUST_BUDGET_TARGET:-256}"
export DUCA_MUST_BUDGET_MULTIPLE="${DUCA_MUST_BUDGET_MULTIPLE:-16}"

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export HOME="${HOME:-${BASE}/tmp/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${BASE}/tmp/xdg_cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${BASE}/tmp/xdg_config}"
export YUZIBO_ROOT="${YUZIBO_ROOT:-${BASE}}"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]; then
  if [[ -z "${SLURM_STEP_GPUS:-}" ]]; then
    fail "DUCA-MUST dynamic run defaults to physical GPU1; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  fi
  echo "[DUCA_MUST_DYNAMIC_BACKEND] accepting Slurm step GPU mapping: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} SLURM_STEP_GPUS=${SLURM_STEP_GPUS}"
fi

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || fail "required file missing: ${path}"
}

require_file "${CONFIG}"
require_file "${VALIDATOR}"

module load cuda/11.8 >/dev/null 2>&1 || true
module load miniforge3/24.11 >/dev/null 2>&1 || true
PYTHON="${PYTHON:-/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python}"
[[ -x "${PYTHON}" ]] || PYTHON="${PYTHON_FALLBACK:-python}"

echo "[DUCA_MUST_DYNAMIC_BACKEND] repo=${REPO_ROOT}"
echo "[DUCA_MUST_DYNAMIC_BACKEND] head=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
echo "[DUCA_MUST_DYNAMIC_BACKEND] config=${CONFIG}"
echo "[DUCA_MUST_DYNAMIC_BACKEND] gpu=${CUDA_VISIBLE_DEVICES}"
echo "[DUCA_MUST_DYNAMIC_BACKEND] slurm_job=${SLURM_JOB_ID:-none} slurm_step=${SLURM_STEP_ID:-none} slurm_step_gpus=${SLURM_STEP_GPUS:-none}"
echo "[DUCA_MUST_DYNAMIC_BACKEND] precheck_only=${PRECHECK_ONLY} fulltrain_candidate=${FULLTRAIN_CANDIDATE}"
echo "[DUCA_MUST_DYNAMIC_BACKEND] dense=${DUCA_MUST_DENSE_WINDOW_SIZE} budget_min=${DUCA_MUST_BUDGET_MIN} budget_target=${DUCA_MUST_BUDGET_TARGET} budget_max=${DUCA_MUST_BUDGET_MAX} multiple=${DUCA_MUST_BUDGET_MULTIPLE}"

bash -n "${BASH_SOURCE[0]}"
"${PYTHON}" -m py_compile "${CONFIG}" "${VALIDATOR}"
"${PYTHON}" "${VALIDATOR}" --config "${CONFIG}" --max-budget "${DUCA_MUST_BUDGET_MAX}"
"${PYTHON}" -m pytest tests/test_duca_online_precheck_config.py -q -k "duca_must_dynamic"

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  echo "[DUCA_MUST_DYNAMIC_BACKEND] PRECHECK_ONLY complete"
  exit 0
fi

if [[ "${FULLTRAIN_CANDIDATE}" != "1" ]]; then
  fail "FULLTRAIN_CANDIDATE=1 is required beyond precheck"
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  fail "formal full train must run inside a Slurm allocation/step"
fi

RUN_DIR="${RUN_DIR:-logs/${RUN_TAG}}"
WORK_DIR="${WORK_DIR:-exps/thumos/adatad/duca_must_dynamic_official_adatad_backend/${RUN_TAG}}"
mkdir -p "${RUN_DIR}" "${WORK_DIR}"

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  tools/train.py \
  "${CONFIG}" \
  --id "${RUN_ID}" \
  --seed "${SEED}" \
  --cfg-options "work_dir=${WORK_DIR}" \
  2>&1 | tee "${RUN_DIR}/train.out"
