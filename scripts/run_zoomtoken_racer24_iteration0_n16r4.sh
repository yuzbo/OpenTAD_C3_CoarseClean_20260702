#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  printf '[ZOOMTOKEN_RACER24_I0][FAIL] invoke this packet with bash, never sh\n' >&2
  exit 2
fi
set -euo pipefail

fail() {
  printf '[ZOOMTOKEN_RACER24_I0][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="${ZOOMTOKEN_RACER24_SOURCE_ROOT:?set the reviewed clean RACER24 source root}"
EXPECTED_COMMIT="${ZOOMTOKEN_RACER24_EXPECTED_COMMIT:?set the reviewed RACER24 commit}"
OUTPUT_ROOT="${ZOOMTOKEN_RACER24_OUTPUT_ROOT:?set one fresh RACER24 output root}"
PYTHON_BIN="${ZOOMTOKEN_RACER24_PYTHON:-python}"
CONFIG="${ROOT}/configs/adatad/thumos/georoute_official_r1_racer24_prebackbone_seed42_v001.py"
TEST="${ROOT}/tests/test_zoomtoken_racer24.py"
PROFILER="${ROOT}/tools/bata/profile_zoomtoken_racer24_block.py"
PROFILE_PATH="${OUTPUT_ROOT}/profile.json"
RECEIPT_PATH="${OUTPUT_ROOT}/terminal_receipt.json"
PROTOCOL_IDENTITY="zoomtoken_racer24_iteration0_n16r4_v001"
COMMAND_IDENTITY="racer24_block:B1:T8:K64:per_tubelet_Q24_KV64:total_Q192_KV512:warmup50:measurements200:gates1.08x_1.05x:seed42"

if [[ -e "${OUTPUT_ROOT}" && ! -d "${OUTPUT_ROOT}" ]]; then
  fail 'output root exists but is not a directory'
fi
if [[ -d "${OUTPUT_ROOT}" ]] && \
   [[ -n "$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  fail 'output root already exists and is non-empty'
fi
mkdir -p "${OUTPUT_ROOT}"

CURRENT_STEP="source_preflight"

write_receipt() {
  local receipt_status="$1"
  local exit_status="$2"
  RACER_RECEIPT_STATUS="${receipt_status}" \
  RACER_RECEIPT_EXIT_STATUS="${exit_status}" \
  RACER_RECEIPT_STEP="${CURRENT_STEP}" \
  RACER_RECEIPT_COMMIT="${EXPECTED_COMMIT}" \
  RACER_RECEIPT_SOURCE_ROOT="${ROOT}" \
  RACER_RECEIPT_OUTPUT_ROOT="${OUTPUT_ROOT}" \
  RACER_RECEIPT_COMMAND="${COMMAND_IDENTITY}" \
  RACER_RECEIPT_PROTOCOL="${PROTOCOL_IDENTITY}" \
  RACER_RECEIPT_PROFILE="${PROFILE_PATH}" \
  "${PYTHON_BIN}" - "${RECEIPT_PATH}" <<'PY'
import json
import os
import sys
from pathlib import Path

receipt = {
    "schema_version": "zoomtoken_racer24_terminal_receipt_v001",
    "status": os.environ["RACER_RECEIPT_STATUS"],
    "exit_status": int(os.environ["RACER_RECEIPT_EXIT_STATUS"]),
    "step": os.environ["RACER_RECEIPT_STEP"],
    "exact_commit": os.environ["RACER_RECEIPT_COMMIT"],
    "source_root": os.environ["RACER_RECEIPT_SOURCE_ROOT"],
    "output_root": os.environ["RACER_RECEIPT_OUTPUT_ROOT"],
    "command_identity": os.environ["RACER_RECEIPT_COMMAND"],
    "protocol_identity": os.environ["RACER_RECEIPT_PROTOCOL"],
    "profile_path": os.environ["RACER_RECEIPT_PROFILE"],
}
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("x", encoding="utf-8") as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
}

on_exit() {
  local exit_status="$?"
  trap - EXIT
  if [[ "${exit_status}" -eq 0 ]]; then
    write_receipt "success" 0
  else
    write_receipt "failure" "${exit_status}" || true
  fi
  exit "${exit_status}"
}

trap on_exit EXIT
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail 'expected commit must be a full SHA'
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail 'source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || \
  fail 'source snapshot is not clean'
for path in "${CONFIG}" "${TEST}" "${PROFILER}"; do
  [[ -f "${path}" ]] || fail "required Iteration-0 path is missing: ${path}"
done
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail 'one visible CUDA device is required'
case "${CUDA_VISIBLE_DEVICES}" in
  *,*) fail 'the matched block profiler requires exactly one visible CUDA device' ;;
esac

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
cd "${ROOT}"

CURRENT_STEP="focused_checks"
"${PYTHON_BIN}" -m py_compile \
  opentad/models/backbones/vit_adapter.py \
  configs/adatad/thumos/georoute_official_r1_racer24_prebackbone_seed42_v001.py \
  tools/bata/profile_zoomtoken_racer24_block.py \
  tests/test_zoomtoken_racer24.py
"${PYTHON_BIN}" -m pytest -q tests/test_zoomtoken_racer24.py
CURRENT_STEP="matched_block_profile"
"${PYTHON_BIN}" tools/bata/profile_zoomtoken_racer24_block.py \
  --device cuda \
  --warmup 50 \
  --measurements 200 \
  --min-speedup 1.08 \
  --max-memory-ratio 1.05 \
  --seed 42 \
  --output "${PROFILE_PATH}"
[[ -f "${PROFILE_PATH}" ]] || fail 'profiler did not publish profile.json'
CURRENT_STEP="complete"
