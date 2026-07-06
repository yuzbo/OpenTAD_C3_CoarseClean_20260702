#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_STAGE3_PRECHECK][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
PYTHON="${PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
[[ -x "${PYTHON}" ]] || fail "python not executable: ${PYTHON}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
[[ "${CUDA_VISIBLE_DEVICES}" == "1" ]] || fail "DUCA Stage3 TrueTime precheck/full run must use GPU1; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

RUN_TAG="${RUN_TAG:-duca_stage3_truetime_precheck_$(date +%Y%m%d_%H%M%S_%z)}"
ROUTE_ROOT="${ROUTE_ROOT:-${BASE}/projects/c3_lowres_action_probe/duca_stage3_truetime/${RUN_TAG}}"
PROOF_DIR="${PROOF_DIR:-${ROUTE_ROOT}/proof}"
mkdir -p "${PROOF_DIR}"

CONFIG="${CONFIG:-configs/adatad/thumos/c3_truetime_joint_selector_adatad_precheck.py}"
EXEC_CONFIG="${EXEC_CONFIG:-configs/adatad/thumos/c3_truetime_joint_selector_adatad_precheck_exec.py}"
PROOF_JSON="${TRUETIME_SELECTOR_GRAD_PROOF_JSON:-${PROOF_DIR}/selector_grad_geometry_precheck.json}"
SUMMARY_JSON="${DUCA_STAGE3_PRECHECK_SUMMARY_JSON:-${ROUTE_ROOT}/duca_stage3_precheck.summary.json}"
export CONFIG EXEC_CONFIG TRUETIME_SELECTOR_GRAD_PROOF_JSON="${PROOF_JSON}"

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || fail "required file missing: ${path}"
}

require_file "${CONFIG}"
require_file "${EXEC_CONFIG}"
require_file tools/bata/run_truetime_joint_selector_precheck.py
require_file tools/bata/validate_truetime_joint_selector_precheck.py
require_file tools/bata/validate_duca_stage23_precheck.py

if [[ "${DUCA_STAGE3_FULL_RUN:-0}" == "1" ]]; then
  "${PYTHON}" - "${SUMMARY_JSON}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"precheck summary missing: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("decision") != "DUCA_STAGE3_PRECHECK_PASS":
    raise SystemExit(f"precheck summary is not DUCA_STAGE3_PRECHECK_PASS: {payload.get('decision')}")
print(f"[DUCA_STAGE3_PRECHECK] full-run gate accepted {path}")
PY
  PRECHECK_ONLY=0 ALLOW_TRUETIME_JOINT_SELECTOR_FULLTRAIN=1 \
    CONFIG="${CONFIG}" EXEC_CONFIG="${EXEC_CONFIG}" \
    bash scripts/run_c3_truetime_joint_selector_adatad_gpu1.sh
  exit 0
fi

"${PYTHON}" -m py_compile \
  tools/train.py \
  tools/bata/run_truetime_joint_selector_precheck.py \
  tools/bata/validate_truetime_joint_selector_precheck.py \
  tools/bata/validate_duca_stage23_precheck.py

"${PYTHON}" tools/bata/run_truetime_joint_selector_precheck.py \
  --config "${CONFIG}" \
  --output-json "${PROOF_JSON}"

"${PYTHON}" tools/bata/validate_duca_stage23_precheck.py \
  --stage stage3 \
  --summary-json "${SUMMARY_JSON}" \
  --stage3-config "${CONFIG}" \
  --stage3-exec-config "${EXEC_CONFIG}" \
  --require-stage3-grad-proof \
  --stage3-grad-proof-json "${PROOF_JSON}"

echo "[DUCA_STAGE3_PRECHECK] PASS summary=${SUMMARY_JSON}"
