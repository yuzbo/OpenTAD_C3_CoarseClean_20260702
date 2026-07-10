#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PHYSTIME_TRAIN][FAIL] $*" >&2
  exit 1
}

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="${PHYSTIME_REPO_ROOT:-${SCRIPT_ROOT}}"
cd "${REPO_ROOT}"

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
PYTHON="${PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
TORCHRUN="${TORCHRUN:-${BASE}/conda_envs/opentad/bin/torchrun}"
CONFIG="${PHYSTIME_CONFIG:?set PHYSTIME_CONFIG}"
EXPERIMENT_ID="${PHYSTIME_EXPERIMENT_ID:?set PHYSTIME_EXPERIMENT_ID}"
RUN_DIR="${PHYSTIME_RUN_DIR:?set PHYSTIME_RUN_DIR}"
SEED="${PHYSTIME_SEED:-42}"
DATA_ROOT="${PHYSTIME_THUMOS_ROOT:-${BASE}/datasets/phystime_thumos_i3d}"
export PHYSTIME_THUMOS_ROOT="${DATA_ROOT}"
export PHYSTIME_FEATURE_PATH="${PHYSTIME_FEATURE_PATH:-${DATA_ROOT}/features/i3d_actionformer_stride4_thumos}"
export PHYSTIME_ANNOTATION_PATH="${PHYSTIME_ANNOTATION_PATH:-${DATA_ROOT}/annotations/thumos_14_anno.json}"
export PHYSTIME_CLASS_MAP="${PHYSTIME_CLASS_MAP:-${DATA_ROOT}/annotations/category_idx.txt}"
export PHYSTIME_BLOCK_LIST="${PHYSTIME_BLOCK_LIST:-${PHYSTIME_FEATURE_PATH}/missing_files.txt}"
export PHYSTIME_WORK_DIR="${RUN_DIR}/work_dir"

[[ -x "${PYTHON}" ]] || fail "Python missing: ${PYTHON}"
[[ -x "${TORCHRUN}" ]] || fail "torchrun missing: ${TORCHRUN}"
[[ -f "${CONFIG}" ]] || fail "config missing: ${CONFIG}"
[[ -f "${DATA_ROOT}/data_ready.json" ]] || fail "data gate marker missing"
[[ -f "${PHYSTIME_ANNOTATION_PATH}" ]] || fail "annotation missing"
[[ -d "${PHYSTIME_FEATURE_PATH}" ]] || fail "feature directory missing"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
  [[ "${CUDA_VISIBLE_DEVICES}" == "1" ]] || fail "outside Slurm this launcher is restricted to physical GPU1"
fi

mkdir -p "${RUN_DIR}"
env | grep '^PHYSTIME_' | sort > "${RUN_DIR}/environment.txt"
git rev-parse HEAD > "${RUN_DIR}/commit.txt"
"${TORCHRUN}" --standalone --nproc_per_node=1 tools/train.py "${CONFIG}" --seed "${SEED}" --id 0 \
  2>&1 | tee "${RUN_DIR}/train.out"
touch "${RUN_DIR}/TRAINING_COMPLETE"
echo "[PHYSTIME_TRAIN] PASS id=${EXPERIMENT_ID} run_dir=${RUN_DIR}"
