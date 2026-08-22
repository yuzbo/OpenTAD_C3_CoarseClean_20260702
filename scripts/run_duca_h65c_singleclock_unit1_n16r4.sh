#!/usr/bin/env bash
set -euo pipefail
CONFIG="configs/adatad/thumos/duca_h65c_singleclock_k384_seed3407.py"
WORK_DIR="exps/thumos/adatad/duca_h65c_singleclock_k384_seed3407"
SEED="3407"
python tools/bata/validate_duca_h65c_singleclock_unit1.py
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  exit 0
fi
exec python tools/train.py "$CONFIG" --work-dir "$WORK_DIR" --seed "$SEED"
