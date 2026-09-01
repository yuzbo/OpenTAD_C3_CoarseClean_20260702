#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PhysTime G1b SDPQ gate] ERROR: $*" >&2
  exit 1
}

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${PHYSTIME_WORK_DIR:-${SCRIPT_ROOT}}"
BASE="${PHYSTIME_BASE:-/data/run01/sczc063/yuzibo}"
PYTHON="${PHYSTIME_PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
CONFIG="${PHYSTIME_G1B_CONFIG:-configs/adatad/thumos/phystime_g1b_sdpq_pool_native_j192.py}"
CHECKPOINT="${PHYSTIME_VIDEOMAE_CHECKPOINT:?PHYSTIME_VIDEOMAE_CHECKPOINT is required}"
OUTPUT="${PHYSTIME_G1B_GATE_OUTPUT:?PHYSTIME_G1B_GATE_OUTPUT is required}"
SEED="${PHYSTIME_SEED:-42}"
EXPECTED_COMMIT="${PHYSTIME_EXPECTED_COMMIT:?PHYSTIME_EXPECTED_COMMIT is required}"
EXPECTED_TREE="${PHYSTIME_EXPECTED_TREE:?PHYSTIME_EXPECTED_TREE is required}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "the G1b SDPQ gate must run inside Slurm"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not assign a visible GPU"
: "${OPENTAD_THUMOS14_ANNOTATION:?OPENTAD_THUMOS14_ANNOTATION is required}"
: "${OPENTAD_THUMOS14_CLASS_MAP:?OPENTAD_THUMOS14_CLASS_MAP is required}"
: "${OPENTAD_THUMOS14_TRAIN_VIDEOS:?OPENTAD_THUMOS14_TRAIN_VIDEOS is required}"
: "${OPENTAD_THUMOS14_TEST_VIDEOS:?OPENTAD_THUMOS14_TEST_VIDEOS is required}"
[[ -x "${PYTHON}" ]] || fail "Python environment not found: ${PYTHON}"
[[ -f "${CHECKPOINT}" ]] || fail "VideoMAE-S checkpoint not found: ${CHECKPOINT}"

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8 >/dev/null 2>&1 || true
  module load miniforge3/24.11 >/dev/null 2>&1 || true
fi
source "${BASE}/conda_envs/opentad/bin/activate"
cd "${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"
[[ -z "$(git status --porcelain)" ]] || fail "runtime snapshot is not clean"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "runtime commit changed after submission"
[[ "$(git rev-parse 'HEAD^{tree}')" == "${EXPECTED_TREE}" ]] || fail "runtime tree changed after submission"

"${PYTHON}" -m py_compile \
  opentad/models/dense_heads/support_decoupled_physical_query_head.py \
  opentad/models/projections/phystime_projection.py \
  opentad/models/detectors/phystime_tad.py \
  tools/bata/run_phystime_g1b_sdpq_real_gate.py
"${PYTHON}" -m pytest \
  tests/test_support_decoupled_physical_query_head.py \
  tests/test_phystime_measure_attention.py \
  tests/test_phystime_detector.py \
  tests/test_phystime_adatad_configs.py -q
"${PYTHON}" tools/bata/run_phystime_g1b_sdpq_real_gate.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --expected-tree "${EXPECTED_TREE}" \
  --device cuda:0 \
  --seed "${SEED}"

echo "[PhysTime G1b SDPQ gate] complete output=${OUTPUT}"
