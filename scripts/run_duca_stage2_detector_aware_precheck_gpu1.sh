#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_STAGE2_PRECHECK][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
PYTHON="${PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
[[ -x "${PYTHON}" ]] || fail "python not executable: ${PYTHON}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
[[ "${CUDA_VISIBLE_DEVICES}" == "1" ]] || fail "DUCA Stage2 precheck/full run must use GPU1; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

RUN_TAG="${RUN_TAG:-duca_stage2_detector_aware_precheck_$(date +%Y%m%d_%H%M%S_%z)}"
export RUN_TAG
ROUTE_ROOT="${ROUTE_ROOT:-${BASE}/projects/c3_lowres_action_probe/duca_stage2_detector_aware/${RUN_TAG}}"
POLICY_DIR="${POLICY_DIR:-${ROUTE_ROOT}/policy}"
LEDGER_ROOT="${LEDGER_ROOT:-${ROUTE_ROOT}/ledgers}"
export ROUTE_ROOT POLICY_DIR LEDGER_ROOT

CONFIG="${CONFIG:-configs/adatad/thumos/c3_duca_stage2_detector_aware_precheck.py}"
EXEC_CONFIG="${EXEC_CONFIG:-configs/adatad/thumos/c3_duca_stage2_detector_aware_precheck_exec.py}"
export CONFIG EXEC_CONFIG

C3_DETECTOR_AWARE_TEACHER_CHECKPOINT_PATH="${C3_DETECTOR_AWARE_TEACHER_CHECKPOINT_PATH:-}"
C3_DETECTOR_AWARE_TEACHER_CONFIG_PATH="${C3_DETECTOR_AWARE_TEACHER_CONFIG_PATH:-}"
[[ -n "${C3_DETECTOR_AWARE_TEACHER_CHECKPOINT_PATH}" ]] || fail "set C3_DETECTOR_AWARE_TEACHER_CHECKPOINT_PATH to the dense AdaTAD teacher checkpoint"
[[ -n "${C3_DETECTOR_AWARE_TEACHER_CONFIG_PATH}" ]] || fail "set C3_DETECTOR_AWARE_TEACHER_CONFIG_PATH to the dense AdaTAD teacher config"
[[ -f "${C3_DETECTOR_AWARE_TEACHER_CHECKPOINT_PATH}" ]] || fail "dense AdaTAD teacher checkpoint missing: ${C3_DETECTOR_AWARE_TEACHER_CHECKPOINT_PATH}"
[[ -f "${C3_DETECTOR_AWARE_TEACHER_CONFIG_PATH}" ]] || fail "dense AdaTAD teacher config missing: ${C3_DETECTOR_AWARE_TEACHER_CONFIG_PATH}"
export C3_DETECTOR_AWARE_TEACHER_CHECKPOINT_PATH C3_DETECTOR_AWARE_TEACHER_CONFIG_PATH

C3_DETECTOR_AWARE_TEACHER_UTILITY_EXPORT_SUMMARY_JSON="${C3_DETECTOR_AWARE_TEACHER_UTILITY_EXPORT_SUMMARY_JSON:-${POLICY_DIR}/teacher_utility_export.summary.json}"
DETECTOR_AWARE_POLICY_CHECKPOINT="${DETECTOR_AWARE_POLICY_CHECKPOINT:-${POLICY_DIR}/detector_aware_policy.pth}"
DUCA_STAGE2_PRECHECK_SUMMARY_JSON="${DUCA_STAGE2_PRECHECK_SUMMARY_JSON:-${ROUTE_ROOT}/duca_stage2_precheck.summary.json}"
export C3_DETECTOR_AWARE_TEACHER_UTILITY_EXPORT_SUMMARY_JSON DETECTOR_AWARE_POLICY_CHECKPOINT DUCA_STAGE2_PRECHECK_SUMMARY_JSON

C3_DETECTOR_AWARE_SOURCE_ROOT="${C3_DETECTOR_AWARE_SOURCE_ROOT:-}"
if [[ -n "${C3_DETECTOR_AWARE_SOURCE_ROOT}" ]]; then
  C3_DETECTOR_AWARE_VAL_SOURCE_JSONL="${C3_DETECTOR_AWARE_VAL_SOURCE_JSONL:-${C3_DETECTOR_AWARE_SOURCE_ROOT}/val/samples.jsonl}"
  C3_DETECTOR_AWARE_TEST_SOURCE_JSONL="${C3_DETECTOR_AWARE_TEST_SOURCE_JSONL:-${C3_DETECTOR_AWARE_SOURCE_ROOT}/test/samples.jsonl}"
else
  C3_DETECTOR_AWARE_VAL_SOURCE_JSONL="${C3_DETECTOR_AWARE_VAL_SOURCE_JSONL:-${ROUTE_ROOT}/source/val.samples.jsonl}"
  C3_DETECTOR_AWARE_TEST_SOURCE_JSONL="${C3_DETECTOR_AWARE_TEST_SOURCE_JSONL:-${ROUTE_ROOT}/source/test.samples.jsonl}"
fi
export C3_DETECTOR_AWARE_VAL_SOURCE_JSONL C3_DETECTOR_AWARE_TEST_SOURCE_JSONL

assert_precheck_pass() {
  local summary_json="$1"
  "${PYTHON}" - "${summary_json}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"precheck summary missing: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("decision") != "DUCA_STAGE2_PRECHECK_PASS":
    raise SystemExit(f"precheck summary is not DUCA_STAGE2_PRECHECK_PASS: {payload.get('decision')}")
print(f"[DUCA_STAGE2_PRECHECK] full-run gate accepted {path}")
PY
}

if [[ "${DUCA_STAGE2_FULL_RUN:-0}" == "1" ]]; then
  assert_precheck_pass "${DUCA_STAGE2_PRECHECK_SUMMARY_JSON}"
  PRECHECK_ONLY=0 ALLOW_C3_DETECTOR_AWARE_ADATAD_FULLTRAIN=1 \
    bash scripts/run_c3_detector_aware_selector_adatad_full_train_gpu1.sh
  exit 0
fi

PRECHECK_ONLY=1 bash scripts/run_c3_detector_aware_selector_adatad_full_train_gpu1.sh

"${PYTHON}" tools/bata/validate_duca_stage23_precheck.py \
  --stage stage2 \
  --summary-json "${DUCA_STAGE2_PRECHECK_SUMMARY_JSON}" \
  --stage2-config "${CONFIG}" \
  --stage2-exec-config "${EXEC_CONFIG}" \
  --stage2-ledger-root "${LEDGER_ROOT}" \
  --require-stage2-ledgers \
  --require-stage2-teacher-evidence \
  --require-stage2-generator-manifest \
  --require-stage2-policy-evidence \
  --stage2-teacher-summary-json "${C3_DETECTOR_AWARE_TEACHER_UTILITY_EXPORT_SUMMARY_JSON}" \
  --stage2-teacher-checkpoint-path "${C3_DETECTOR_AWARE_TEACHER_CHECKPOINT_PATH}" \
  --stage2-teacher-config-path "${C3_DETECTOR_AWARE_TEACHER_CONFIG_PATH}" \
  --stage2-policy-summary-json "${POLICY_DIR}/train.summary.json" \
  --stage2-policy-checkpoint-path "${DETECTOR_AWARE_POLICY_CHECKPOINT}" \
  --stage2-val-source-jsonl "${C3_DETECTOR_AWARE_VAL_SOURCE_JSONL}" \
  --stage2-test-source-jsonl "${C3_DETECTOR_AWARE_TEST_SOURCE_JSONL}"

echo "[DUCA_STAGE2_PRECHECK] PASS summary=${DUCA_STAGE2_PRECHECK_SUMMARY_JSON}"
