#!/usr/bin/env bash
source /etc/profile
set -euo pipefail

fail() { echo "[DUCA_CT_DP_REVISED][FAIL] $*" >&2; exit 1; }

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
REPO_ROOT="${REPO_ROOT:-${BASE}/projects/opentad_duca_ct_dp_revised_20260902}"
CONFIG="${1:-configs/adatad/thumos/duca_ct_dual_phase_bamod_thumos_revised.py}"
SEED="${SEED:-3407}"
EXP_ID="${EXP_ID:-0}"

[[ -d "${REPO_ROOT}" ]] || fail "repo not found: ${REPO_ROOT}"
cd "${REPO_ROOT}"
[[ -f "${CONFIG}" ]] || fail "config not found: ${CONFIG}"

export HOME="${BASE}/tmp/home"
export XDG_CACHE_HOME="${BASE}/tmp/xdg_cache"
export XDG_CONFIG_HOME="${BASE}/tmp/xdg_config"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${REPO_ROOT}/logs" "${BASE}/slurm_logs"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

module load cuda/11.8
module load miniforge3/24.11
PYTHON="${PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
[[ -x "${PYTHON}" ]] || fail "python executable not found: ${PYTHON}"

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  "${PYTHON}" -m py_compile \
    opentad/models/bricks/scale_adaptive_conv1d.py \
    opentad/models/selectors/dual_phase_frame_selector.py \
    opentad/models/detectors/actionformer.py
  echo "[DUCA_CT_DP_REVISED] precheck passed"
  exit 0
fi

echo "[DUCA_CT_DP_REVISED] repo=${REPO_ROOT} commit=$(git rev-parse --short HEAD) config=${CONFIG} seed=${SEED}"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 2000))}"
"${PYTHON}" -m torch.distributed.run --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" tools/train.py "${CONFIG}" --seed "${SEED}" --id "${EXP_ID}"
