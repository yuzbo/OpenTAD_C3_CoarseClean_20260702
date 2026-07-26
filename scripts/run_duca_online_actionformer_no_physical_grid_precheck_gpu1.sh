#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG="${CONFIG:-configs/adatad/thumos/duca_online_actionformer_no_physical_grid_precheck.py}"
export RUN_TAG="${RUN_TAG:-duca_online_actionformer_no_physical_grid_gpu1_$(date +%Y%m%d_%H%M%S_%z)}"
exec "${SCRIPT_DIR}/run_duca_online_adatad_precheck_gpu1.sh"
