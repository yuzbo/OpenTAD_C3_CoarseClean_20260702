#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PhysTime G1a gate] ERROR: $*" >&2
  exit 1
}

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${PHYSTIME_WORK_DIR:-${SCRIPT_ROOT}}"
BASE="${PHYSTIME_BASE:-/data/run01/sczc063/yuzibo}"
PYTHON="${PHYSTIME_PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
CHECKPOINT="${PHYSTIME_VIDEOMAE_CHECKPOINT:?PHYSTIME_VIDEOMAE_CHECKPOINT is required}"
CONTRACT_OUTPUT="${PHYSTIME_G1A_CONTRACT_JSON:?PHYSTIME_G1A_CONTRACT_JSON is required}"
G0_OUTPUT="${PHYSTIME_G0_OUTPUT:?PHYSTIME_G0_OUTPUT is required}"
GATE_OUTPUT="${PHYSTIME_G1A_GATE_OUTPUT:?PHYSTIME_G1A_GATE_OUTPUT is required}"
SEED="${PHYSTIME_SEED:-42}"
EXPECTED_COMMIT="${PHYSTIME_EXPECTED_COMMIT:?PHYSTIME_EXPECTED_COMMIT is required}"
EXPECTED_TREE="${PHYSTIME_EXPECTED_TREE:?PHYSTIME_EXPECTED_TREE is required}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "the formal G1a gate must run inside Slurm"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not assign a visible GPU"
: "${OPENTAD_THUMOS14_ANNOTATION:?OPENTAD_THUMOS14_ANNOTATION is required}"
: "${OPENTAD_THUMOS14_CLASS_MAP:?OPENTAD_THUMOS14_CLASS_MAP is required}"
: "${OPENTAD_THUMOS14_TRAIN_VIDEOS:?OPENTAD_THUMOS14_TRAIN_VIDEOS is required}"
: "${OPENTAD_THUMOS14_TEST_VIDEOS:?OPENTAD_THUMOS14_TEST_VIDEOS is required}"
[[ -x "${PYTHON}" ]] || fail "Python environment not found: ${PYTHON}"
[[ -f "${CHECKPOINT}" ]] || fail "VideoMAE-S checkpoint not found: ${CHECKPOINT}"
[[ -f "${OPENTAD_THUMOS14_ANNOTATION}" ]] || fail "annotation file not found"
[[ -f "${OPENTAD_THUMOS14_CLASS_MAP}" ]] || fail "class map not found"
[[ -d "${OPENTAD_THUMOS14_TRAIN_VIDEOS}" ]] || fail "training videos not found"
[[ -d "${OPENTAD_THUMOS14_TEST_VIDEOS}" ]] || fail "test videos not found"

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
mkdir -p "$(dirname "${CONTRACT_OUTPUT}")" "$(dirname "${G0_OUTPUT}")" "$(dirname "${GATE_OUTPUT}")"

"${PYTHON}" tools/bata/validate_phystime_g1a_track.py \
  --output "${CONTRACT_OUTPUT}"
"${PYTHON}" tools/bata/audit_phystime_g0_native_geometry.py \
  --output "${G0_OUTPUT}" --static-only
"${PYTHON}" tools/bata/run_phystime_g1a_real_gate.py \
  --checkpoint "${CHECKPOINT}" \
  --contract "${CONTRACT_OUTPUT}" \
  --static-g0 "${G0_OUTPUT}" \
  --output "${GATE_OUTPUT}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --expected-tree "${EXPECTED_TREE}" \
  --device cuda:0 \
  --seed "${SEED}"

echo "[PhysTime G1a gate] complete output=${GATE_OUTPUT}"
