#!/usr/bin/env bash
set -euo pipefail
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
: "${DUCA_STAGE1_CHECKPOINT:?DUCA_STAGE1_CHECKPOINT is required}"
: "${DUCA_STAGE1_CHECKPOINT_SHA256:?DUCA_STAGE1_CHECKPOINT_SHA256 is required}"
export DUCA_STAGE1_CHECKPOINT DUCA_STAGE1_CHECKPOINT_SHA256
export DUCA_STAGE1_CHECKPOINT_EPOCH="${DUCA_STAGE1_CHECKPOINT_EPOCH:-30}"
export DUCA_PJST_SEED=3407
export DUCA_PJST_SUCCESSFUL_UPDATES=6000
export DUCA_PJST_CHECKPOINT_EVERY_EPOCHS=5
export DUCA_PJST_TERMINAL_RULE="final_and_final_ema"
export DUCA_PJST_OFF_ROOT="${DUCA_PJST_OFF_ROOT:-exps/thumos/adatad/DUCA_PJST_DERIVATIVE_CAUSAL_FREEZE-v002/matched_off}"
export DUCA_PJST_ON_ROOT="${DUCA_PJST_ON_ROOT:-exps/thumos/adatad/DUCA_PJST_DERIVATIVE_CAUSAL_FREEZE-v002/pjst_d1_on}"
export DUCA_PJST_EVALUATOR="official"
if [[ "$PRECHECK_ONLY" == 1 ]]; then
  python - <<'PY'
import os
assert os.environ['DUCA_PJST_SEED'] == '3407'
assert os.environ['DUCA_PJST_SUCCESSFUL_UPDATES'] == '6000'
assert os.environ['DUCA_PJST_CHECKPOINT_EVERY_EPOCHS'] == '5'
assert os.environ['DUCA_PJST_OFF_ROOT'] != os.environ['DUCA_PJST_ON_ROOT']
assert os.environ['DUCA_PJST_EVALUATOR'] == 'official'
print('PJST_D1_H65_30PLUS60_PRECHECK_PASS')
PY
  exit 0
fi
echo 'Training/evaluation intentionally requires the project Slurm launcher.' >&2
exit 2
