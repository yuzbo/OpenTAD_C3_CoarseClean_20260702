#!/usr/bin/env bash
#SBATCH --gpus=1
set -euo pipefail
ROOT="${DUCA_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"; cd "$ROOT"
MODE="${MODE:-STAGE2_OFF}"; [[ "$MODE" == STAGE2_OFF || "$MODE" == STAGE2_ON ]] || { echo "invalid MODE" >&2; exit 1; }
if [[ "$MODE" == STAGE2_OFF ]]; then CFG=configs/adatad/thumos/duca_pjst_d1_stage2_off.py; else CFG=configs/adatad/thumos/duca_pjst_d1_stage2_on.py; fi
[[ "${DUCA_STAGE1_CHECKPOINT_EPOCH:-29}" == 29 && -n "${DUCA_STAGE1_CHECKPOINT:-}" && -n "${DUCA_STAGE1_CHECKPOINT_SHA256:-}" ]] || { echo "stage1 checkpoint/hash/epoch29 required" >&2; exit 1; }
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then python -m py_compile "$CFG" tools/train.py; exit 0; fi
exec python tools/train.py "$CFG" --seed 3407
