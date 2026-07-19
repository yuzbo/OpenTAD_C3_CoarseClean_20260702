#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PhysTime P0 full-precision gate] ERROR: $*" >&2
  exit 1
}

WORK_DIR="${PHYSTIME_WORK_DIR:?PHYSTIME_WORK_DIR is required}"
BASE="${PHYSTIME_BASE:-/data/run01/sczc063/yuzibo}"
PYTHON="${PHYSTIME_PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
EXPECTED_COMMIT="${PHYSTIME_EXPECTED_COMMIT:?PHYSTIME_EXPECTED_COMMIT is required}"
EXPECTED_TREE="${PHYSTIME_EXPECTED_TREE:?PHYSTIME_EXPECTED_TREE is required}"
SELECTED_SOURCE_DIR="${PHYSTIME_SELECTED_SOURCE_DIR:?PHYSTIME_SELECTED_SOURCE_DIR is required}"
PHYSICAL_SOURCE_DIR="${PHYSTIME_PHYSICAL_SOURCE_DIR:?PHYSTIME_PHYSICAL_SOURCE_DIR is required}"
SELECTED_CHECKPOINT="${PHYSTIME_SELECTED_CHECKPOINT:?PHYSTIME_SELECTED_CHECKPOINT is required}"
PHYSICAL_CHECKPOINT="${PHYSTIME_PHYSICAL_CHECKPOINT:?PHYSTIME_PHYSICAL_CHECKPOINT is required}"
VIDEOMAE_CHECKPOINT="${PHYSTIME_VIDEOMAE_CHECKPOINT:?PHYSTIME_VIDEOMAE_CHECKPOINT is required}"
GATE_OUTPUT="${PHYSTIME_P0_GATE_OUTPUT:?PHYSTIME_P0_GATE_OUTPUT is required}"
TEST_LOG="${PHYSTIME_P0_TEST_LOG:?PHYSTIME_P0_TEST_LOG is required}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "gate must run inside Slurm"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not assign a visible GPU"
[[ -d "${WORK_DIR}" ]] || fail "runtime snapshot is missing"
[[ -d "${SELECTED_SOURCE_DIR}" && -d "${PHYSICAL_SOURCE_DIR}" ]] \
  || fail "source full60 run directories are missing"
[[ -f "${SELECTED_CHECKPOINT}" && -f "${PHYSICAL_CHECKPOINT}" \
    && -f "${VIDEOMAE_CHECKPOINT}" ]] \
  || fail "source epoch-59 or VideoMAE checkpoints are missing"

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
  PHYSTIME_ENV_INIT_MODE="module_cuda11.8_miniforge3_24.11"
else
  PHYSTIME_ENV_INIT_MODE="fixed_conda_path_no_module_command"
fi
export PHYSTIME_ENV_INIT_MODE
source "${BASE}/conda_envs/opentad/bin/activate"
cd "${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"

COMMIT="$(git rev-parse HEAD)"
TREE="$(git rev-parse 'HEAD^{tree}')"
[[ "${COMMIT}" == "${EXPECTED_COMMIT}" ]] || fail "runtime commit mismatch"
[[ "${TREE}" == "${EXPECTED_TREE}" ]] || fail "runtime tree mismatch"
[[ -z "$(git status --porcelain)" ]] || fail "runtime snapshot is dirty"

mkdir -p "$(dirname "${GATE_OUTPUT}")" "$(dirname "${TEST_LOG}")"
set +e
"${PYTHON}" -m pytest \
  tests/test_phystime_fullprecision_nms_replay.py \
  tests/test_phystime_p0_fullprecision_deployment.py \
  -q 2>&1 | tee "${TEST_LOG}"
TEST_STATUS="${PIPESTATUS[0]}"
set -e
[[ "${TEST_STATUS}" == "0" ]] \
  || fail "focused P0 tests failed with exit code ${TEST_STATUS}"

"${PYTHON}" tools/bata/run_phystime_p0_fullprecision_gate.py \
  --selected-source-dir "${SELECTED_SOURCE_DIR}" \
  --physical-source-dir "${PHYSICAL_SOURCE_DIR}" \
  --selected-checkpoint "${SELECTED_CHECKPOINT}" \
  --physical-checkpoint "${PHYSICAL_CHECKPOINT}" \
  --videomae-checkpoint "${VIDEOMAE_CHECKPOINT}" \
  --output "${GATE_OUTPUT}" \
  --expected-runtime-commit "${EXPECTED_COMMIT}" \
  --expected-runtime-tree "${EXPECTED_TREE}" \
  --focused-tests-log "${TEST_LOG}"

"${PYTHON}" - "${GATE_OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

gate = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if gate.get("gate_pass") is not True:
    raise SystemExit("P0 gate report did not pass")
PY

echo "[PhysTime P0 full-precision gate] PASS output=${GATE_OUTPUT}"
