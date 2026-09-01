#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8 || true
  module load miniforge3/24.11 || true
fi

CONDA_ENV="/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate"
if [[ -f "${CONDA_ENV}" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_ENV}"
fi

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4

mkdir -p diagnostics

echo "[ET-TRC DIAGNOSTIC] Launching ZT-DIAG-2025-01 Taylor Residual Manifold Diagnostic..."
python tools/bata/diagnose_taylor_residual_manifold.py \
    configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_seed4407.py \
    --output diagnostics/zt_diag_2025_01_receipt.json \
    --max-batches 20 \
    --stride-k 4

echo "[ET-TRC DIAGNOSTIC] Complete. Receipt written to diagnostics/zt_diag_2025_01_receipt.json."
