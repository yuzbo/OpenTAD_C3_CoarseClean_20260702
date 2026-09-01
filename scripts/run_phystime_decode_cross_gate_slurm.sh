#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PhysTime decode cross gate] ERROR: $*" >&2
  exit 1
}

BASE="${PHYSTIME_BASE:-/data/run01/sczc063/yuzibo}"
WORK_DIR="${PHYSTIME_WORK_DIR:?PHYSTIME_WORK_DIR is required}"
PYTHON="${PHYSTIME_PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
EXPECTED_COMMIT="${PHYSTIME_EXPECTED_COMMIT:?PHYSTIME_EXPECTED_COMMIT is required}"
EXPECTED_TREE="${PHYSTIME_EXPECTED_TREE:?PHYSTIME_EXPECTED_TREE is required}"
SELECTED_CONFIG="${PHYSTIME_DECODE_SELECTED_CONFIG:?selected config is required}"
PHYSICAL_CONFIG="${PHYSTIME_DECODE_PHYSICAL_CONFIG:?physical config is required}"
SELECTED_CHECKPOINT="${PHYSTIME_SELECTED_CHECKPOINT:?selected checkpoint is required}"
PHYSICAL_CHECKPOINT="${PHYSTIME_PHYSICAL_CHECKPOINT:?physical checkpoint is required}"
VIDEOMAE_CHECKPOINT="${PHYSTIME_VIDEOMAE_CHECKPOINT:?VideoMAE checkpoint is required}"
SELECTED_SOURCE_DIR="${PHYSTIME_SELECTED_SOURCE_DIR:?selected source dir is required}"
PHYSICAL_SOURCE_DIR="${PHYSTIME_PHYSICAL_SOURCE_DIR:?physical source dir is required}"
P0_RUN_ROOT="${PHYSTIME_P0_RUN_ROOT:?P0 run root is required}"
GATE_ROOT="${PHYSTIME_DECODE_GATE_ROOT:?gate root is required}"
GATE_OUTPUT="${PHYSTIME_DECODE_GATE_OUTPUT:?gate output is required}"
TEST_LOG="${PHYSTIME_DECODE_TEST_LOG:?test log is required}"
PREFLIGHT="${PHYSTIME_DECODE_PREFLIGHT:?preflight manifest is required}"
PREFLIGHT_SHA256="${PHYSTIME_DECODE_PREFLIGHT_SHA256:?preflight SHA256 is required}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "gate must run inside Slurm"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not assign a GPU"
[[ -n "${SLURM_JOB_NAME:-}" ]] || fail "Slurm job name is missing"
[[ -x "${PYTHON}" ]] || fail "fixed Python is missing"
for path in \
  "${SELECTED_CONFIG}" \
  "${PHYSICAL_CONFIG}" \
  "${SELECTED_CHECKPOINT}" \
  "${PHYSICAL_CHECKPOINT}" \
  "${VIDEOMAE_CHECKPOINT}" \
  "${PREFLIGHT}" \
  "${P0_RUN_ROOT}/P0_SUITE_COMPLETE.json"; do
  [[ -f "${path}" ]] || fail "required file is missing: ${path}"
done

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
source "${BASE}/conda_envs/opentad/bin/activate"
cd "${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PHYSTIME_SOURCE_COMMIT="${PHYSTIME_SOURCE_COMMIT:?source commit is required}"
export PHYSTIME_SOURCE_TREE="${PHYSTIME_SOURCE_TREE:?source tree is required}"
export PHYSTIME_CHECKPOINT_PATH="${PHYSICAL_CHECKPOINT}"

[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] \
  || fail "runtime commit mismatch"
[[ "$(git rev-parse 'HEAD^{tree}')" == "${EXPECTED_TREE}" ]] \
  || fail "runtime tree mismatch"
[[ -z "$(git status --porcelain)" ]] || fail "runtime snapshot is dirty"

mkdir -p "${GATE_ROOT}"
"${PYTHON}" -m py_compile \
  opentad/models/detectors/actionformer.py \
  opentad/models/utils/native_temporal_geometry.py \
  opentad/models/dense_heads/anchor_free_head.py \
  opentad/cores/phystime_decode_replay_capture.py \
  opentad/cores/test_engine.py \
  tools/bata/replay_phystime_decode_cross.py \
  tools/bata/validate_phystime_decode_cross_replay.py \
  tools/bata/validate_phystime_decode_cross_suite.py \
  tools/bata/preflight_phystime_decode_cross.py \
  tools/bata/run_phystime_decode_cross_gate.py

set +e
"${PYTHON}" -m pytest \
  tests/test_phystime_decode_cross_replay.py \
  tests/test_phystime_decode_cross_evidence_suite.py \
  tests/test_phystime_fullprecision_nms_replay.py \
  tests/test_phystime_performance_diagnostics.py \
  tests/test_phystime_prediction_diagnostics.py \
  tests/test_phystime_adatad_configs.py::test_g1a_native_configs_bind_geometry_to_actionformer_runtime \
  tests/test_phystime_native_tubelet_geometry.py::test_actionformer_consumes_native_geometry_without_changing_detector_family \
  -q 2>&1 | tee "${TEST_LOG}"
TEST_STATUS="${PIPESTATUS[0]}"
set -e
[[ "${TEST_STATUS}" == "0" ]] \
  || fail "focused tests failed with status ${TEST_STATUS}"

"${PYTHON}" tools/bata/run_phystime_decode_cross_gate.py \
  --selected-config "${SELECTED_CONFIG}" \
  --physical-config "${PHYSICAL_CONFIG}" \
  --selected-checkpoint "${SELECTED_CHECKPOINT}" \
  --physical-checkpoint "${PHYSICAL_CHECKPOINT}" \
  --videomae-checkpoint "${VIDEOMAE_CHECKPOINT}" \
  --selected-source-dir "${SELECTED_SOURCE_DIR}" \
  --physical-source-dir "${PHYSICAL_SOURCE_DIR}" \
  --p0-run-root "${P0_RUN_ROOT}" \
  --preflight-manifest "${PREFLIGHT}" \
  --expected-preflight-sha256 "${PREFLIGHT_SHA256}" \
  --focused-test-log "${TEST_LOG}" \
  --work-dir "${GATE_ROOT}/real_window" \
  --output "${GATE_OUTPUT}" \
  --expected-runtime-commit "${EXPECTED_COMMIT}" \
  --expected-runtime-tree "${EXPECTED_TREE}" \
  --seed 42

"${PYTHON}" - "${GATE_OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("gate_pass") is not True:
    raise SystemExit("decode cross real gate did not pass")
if payload.get("all_native_direct_exact_equivalence") is not True:
    raise SystemExit("four-condition native replay equivalence did not pass")
expected = {
    "selected_online",
    "selected_ema",
    "physical_online",
    "physical_ema",
}
if set(payload.get("real_windows", {})) != expected:
    raise SystemExit("four-condition real gate is incomplete")
if not all(
    record.get("raw_tensors_immutable") is True
    for record in payload["real_windows"].values()
):
    raise SystemExit("real gate tensor immutability did not pass")
PY

echo "[PhysTime decode cross gate] COMPLETE"
