#!/usr/bin/env bash
source /etc/profile
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ETTRC_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${ROOT}"
module load cuda/11.8
module load miniforge3/24.11

BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
PYTHON="${PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
[[ -x "${PYTHON}" ]] || { echo "missing python: ${PYTHON}" >&2; exit 1; }
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
CONFIG="${1:-configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_seed4407.py}"
[[ -f "${CONFIG}" ]] || { echo "missing config: ${CONFIG}" >&2; exit 1; }

echo "[ET-TRC DIAGNOSTIC] fixed-stride Taylor carryover diagnostic; repo=${ROOT} commit=$(git rev-parse --short HEAD)"
mkdir -p diagnostics
if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  "${PYTHON}" -m py_compile opentad/models/backbones/et_trc_videomae.py opentad/models/backbones/backbone_wrapper.py
  exit 0
fi
"${PYTHON}" tools/bata/diagnose_taylor_residual_manifold.py "${CONFIG}" \
    --output "${ETTRC_DIAG_OUTPUT:-diagnostics/zt_diag_2025_01_receipt.json}" \
    --max-batches 50 \
    --stride-k 4

echo "[ET-TRC DIAGNOSTIC] Complete."
