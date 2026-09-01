#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${ZOOMTOKEN_RUN_ROOT:-${BASE}/projects/bafdr_k16_fullmatrix_compute}"
MANIFEST_DIR="${RUN_ROOT}/manifest"
WORK_DIR_ROOT="${RUN_ROOT}/work_dirs"
PRED_DIR="${RUN_ROOT}/predictions"
EVAL_DIR="${RUN_ROOT}/evaluation"
PROFILE_DIR="${RUN_ROOT}/profile"

mkdir -p "${MANIFEST_DIR}" "${WORK_DIR_ROOT}" "${PRED_DIR}" "${EVAL_DIR}" "${PROFILE_DIR}"

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8 || true
  module load miniforge3/24.11 || true
fi

CONDA_ENV="${BASE}/conda_envs/opentad/bin/activate"
if [[ -f "${CONDA_ENV}" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_ENV}"
fi

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4

if [[ $# -ge 1 ]]; then
  CONFIG="$1"
  echo "[BA-FDR K16] Executing cell: ${CONFIG}"
  python tools/bata/bafdr_k16_fullmatrix_train.py "${CONFIG}"
else
  echo "[BA-FDR K16] Validating master protocol and 21 cell configs..."
  python tools/bata/bafdr_k16_fullmatrix.py \
      --repo-root "${ROOT}" \
      --output "${MANIFEST_DIR}/submission_receipt.json"
  echo "[BA-FDR K16] Validation successful. Master receipt at ${MANIFEST_DIR}/submission_receipt.json"
fi
