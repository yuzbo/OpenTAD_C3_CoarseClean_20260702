#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:."
export OMP_NUM_THREADS=4

echo "[ET-TRC DIAGNOSTIC] Launching ZT-DIAG-2025-01 Taylor Residual Manifold Diagnostic..."
python tools/bata/diagnose_taylor_residual_manifold.py \
    configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_seed4407.py \
    --output diagnostics/zt_diag_2025_01_receipt.json \
    --max-batches 50 \
    --stride-k 4

echo "[ET-TRC DIAGNOSTIC] Complete. Receipt written to diagnostics/zt_diag_2025_01_receipt.json."
