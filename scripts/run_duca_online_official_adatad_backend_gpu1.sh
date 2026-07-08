#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_OFFICIAL_ADATAD_BACKEND][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
FULLTRAIN_CANDIDATE="${FULLTRAIN_CANDIDATE:-0}"
CONFIG="${CONFIG:-configs/adatad/thumos/duca_online_official_adatad_backend_full_train.py}"
VALIDATOR="${VALIDATOR:-tools/bata/validate_duca_official_adatad_backend.py}"
RUN_TAG="${RUN_TAG:-duca_online_official_adatad_backend_$(date +%Y%m%d_%H%M%S_%z)}"
RUN_ID="${RUN_ID:-0}"
SEED="${SEED:-0}"
MASTER_PORT="${MASTER_PORT:-30261}"
ADATAD_PRETRAIN_FILENAME="${ADATAD_PRETRAIN_FILENAME:-vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
ADATAD_PRETRAIN_PATH="${ADATAD_PRETRAIN_PATH:-${DUCA_ADATAD_PRETRAIN_PATH:-${BASE:-/data/run01/sczc063/yuzibo}/pretrained/${ADATAD_PRETRAIN_FILENAME}}}"
export DUCA_ONLINE_BUDGET="${DUCA_ONLINE_BUDGET:-384}"
export DUCA_ONLINE_DENSE_WINDOW_SIZE="${DUCA_ONLINE_DENSE_WINDOW_SIZE:-768}"
DUCA_VALIDATOR_MAX_BUDGET="${DUCA_VALIDATOR_MAX_BUDGET:-384}"

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export HOME="${HOME:-${BASE}/tmp/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${BASE}/tmp/xdg_cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${BASE}/tmp/xdg_config}"
export YUZIBO_ROOT="${YUZIBO_ROOT:-${BASE}}"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" logs

if [[ -n "${SLURM_STEP_GPUS:-}${SLURM_JOB_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  if [[ "${CUDA_VISIBLE_DEVICES}" != "0" && "${CUDA_VISIBLE_DEVICES}" != "1" ]]; then
    fail "official-backend DUCA run must see one Slurm-bound GPU as logical 0/1; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  fi
  echo "[DUCA_OFFICIAL_ADATAD_BACKEND] accepting Slurm GPU mapping: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-none} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-none}"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
  if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]; then
    fail "official-backend DUCA run defaults to physical GPU1 outside Slurm remapping; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  fi
fi

resolve_path() {
  local raw="$1"
  if [[ "${raw}" == /* ]]; then
    echo "${raw}"
  elif [[ -f "${REPO_ROOT}/${raw}" ]]; then
    readlink -f "${REPO_ROOT}/${raw}"
  elif [[ -f "${BASE}/${raw}" ]]; then
    readlink -f "${BASE}/${raw}"
  else
    echo "${REPO_ROOT}/${raw}"
  fi
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || fail "required file missing: ${path}"
}

ADATAD_PRETRAIN_PATH="$(resolve_path "${ADATAD_PRETRAIN_PATH}")"
export DUCA_ADATAD_PRETRAIN_PATH="${ADATAD_PRETRAIN_PATH}"

require_file "${CONFIG}"
require_file "${VALIDATOR}"
[[ -f "${ADATAD_PRETRAIN_PATH}" ]] || fail "required file missing: ${ADATAD_PRETRAIN_PATH}"

module load cuda/11.8 >/dev/null 2>&1 || true
module load miniforge3/24.11 >/dev/null 2>&1 || true
PYTHON="${PYTHON:-/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python}"
[[ -x "${PYTHON}" ]] || PYTHON="${PYTHON_FALLBACK:-python}"

echo "[DUCA_OFFICIAL_ADATAD_BACKEND] repo=${REPO_ROOT}"
echo "[DUCA_OFFICIAL_ADATAD_BACKEND] head=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
echo "[DUCA_OFFICIAL_ADATAD_BACKEND] config=${CONFIG}"
echo "[DUCA_OFFICIAL_ADATAD_BACKEND] gpu=${CUDA_VISIBLE_DEVICES}"
echo "[DUCA_OFFICIAL_ADATAD_BACKEND] slurm_job=${SLURM_JOB_ID:-none} slurm_step=${SLURM_STEP_ID:-none} slurm_step_gpus=${SLURM_STEP_GPUS:-none} slurm_job_gpus=${SLURM_JOB_GPUS:-none}"
echo "[DUCA_OFFICIAL_ADATAD_BACKEND] precheck_only=${PRECHECK_ONLY} fulltrain_candidate=${FULLTRAIN_CANDIDATE}"
echo "[DUCA_OFFICIAL_ADATAD_BACKEND] budget=${DUCA_ONLINE_BUDGET} dense_window=${DUCA_ONLINE_DENSE_WINDOW_SIZE} validator_max_budget=${DUCA_VALIDATOR_MAX_BUDGET}"
echo "[DUCA_OFFICIAL_ADATAD_BACKEND] adatad_pretrain_path=${ADATAD_PRETRAIN_PATH}"

bash -n "${BASH_SOURCE[0]}"
"${PYTHON}" -m py_compile "${CONFIG}" "${VALIDATOR}"
"${PYTHON}" "${VALIDATOR}" --config "${CONFIG}" --max-budget "${DUCA_VALIDATOR_MAX_BUDGET}"
"${PYTHON}" -m pytest tests/test_duca_online_precheck_config.py -q

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  echo "[DUCA_OFFICIAL_ADATAD_BACKEND] PRECHECK_ONLY complete"
  exit 0
fi

if [[ "${FULLTRAIN_CANDIDATE}" != "1" ]]; then
  fail "FULLTRAIN_CANDIDATE=1 is required beyond precheck"
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  fail "formal full train must run inside a Slurm allocation/step"
fi

RUN_DIR="${RUN_DIR:-logs/${RUN_TAG}/budget_${DUCA_ONLINE_BUDGET}}"
WORK_DIR="${WORK_DIR:-exps/thumos/adatad/duca_online_official_adatad_backend/${RUN_TAG}/budget_${DUCA_ONLINE_BUDGET}}"
mkdir -p "${RUN_DIR}" "${WORK_DIR}"

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  tools/train.py \
  "${CONFIG}" \
  --id "${RUN_ID}" \
  --seed "${SEED}" \
  --cfg-options "work_dir=${WORK_DIR}" "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" \
  2>&1 | tee "${RUN_DIR}/train.out"
