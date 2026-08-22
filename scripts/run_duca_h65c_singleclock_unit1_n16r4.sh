#!/usr/bin/env bash
set -euo pipefail
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  python tools/bata/validate_duca_h65c_singleclock_unit1.py
  exit 0
fi
exec python tools/train.py configs/adatad/thumos/duca_h65c_singleclock_k384_seed3407.py \
  --work-dir exps/thumos/adatad/duca_h65c_singleclock_k384_seed3407 --seed 3407
