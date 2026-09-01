#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_CT_DP_BAMOD_TRAIN][FAIL] $*" >&2
  exit 1
}

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
REPO_ROOT="${REPO_ROOT:-${BASE}/projects/opentad_duca_ct_dp_bamod_d9bdb3f_20260901}"
cd "${REPO_ROOT}"

CONFIG="${1:-configs/adatad/thumos/duca_ct_dual_phase_bamod_thumos.py}"
SEED="${SEED:-3407}"
EXP_ID="${EXP_ID:-0}"

export HOME="${BASE}/tmp/home"
export XDG_CACHE_HOME="${BASE}/tmp/xdg_cache"
export XDG_CONFIG_HOME="${BASE}/tmp/xdg_config"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${REPO_ROOT}/logs" "${BASE}/slurm_logs"



export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8 >/dev/null 2>&1 || true
  module load miniforge3/24.11 >/dev/null 2>&1 || true
fi

PYTHON="${PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
[[ -x "${PYTHON}" ]] || fail "Python executable not found: ${PYTHON}"

echo "[DUCA_CT_DP_BAMOD_TRAIN] Starting training for config=${CONFIG} with seed=${SEED}"
echo "[DUCA_CT_DP_BAMOD_TRAIN] repo=${REPO_ROOT} commit=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
echo "[DUCA_CT_DP_BAMOD_TRAIN] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none} hostname=$(hostname)"

MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 2000))}"

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  tools/train.py "${CONFIG}" --seed "${SEED}" --id "${EXP_ID}"
echo "[DUCA_CT_DP_BAMOD_TRAIN] Training finished successfully!"
