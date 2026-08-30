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
PYTHON_BIN="${ZOOMTOKEN_RACER24_PYTHON:-python}"
CONFIG="${ROOT}/configs/adatad/thumos/georoute_official_r1_racer24_prebackbone_seed42_v001.py"
TEST="${ROOT}/tests/test_zoomtoken_racer24.py"
PROFILER="${ROOT}/tools/bata/profile_zoomtoken_racer24_block.py"

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

"${PYTHON_BIN}" -m py_compile \
  opentad/models/backbones/vit_adapter.py \
  configs/adatad/thumos/georoute_official_r1_racer24_prebackbone_seed42_v001.py \
  tools/bata/profile_zoomtoken_racer24_block.py \
  tests/test_zoomtoken_racer24.py
"${PYTHON_BIN}" -m pytest -q tests/test_zoomtoken_racer24.py
exec "${PYTHON_BIN}" tools/bata/profile_zoomtoken_racer24_block.py \
  --device cuda \
  --warmup 50 \
  --measurements 200 \
  --min-speedup 1.08 \
  --max-memory-ratio 1.05 \
  --seed 42
