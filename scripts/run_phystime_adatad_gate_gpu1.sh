#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PhysTime-AdaTAD gate] ERROR: $*" >&2
  exit 1
}

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES
[[ "${CUDA_VISIBLE_DEVICES}" == "1" ]] || fail "CUDA_VISIBLE_DEVICES must be physical GPU1"
[[ -n "${SLURM_JOB_ID:-}" || "${ALLOW_NON_SLURM_PHYSTIME_GATE:-0}" == "1" ]] || \
  fail "the real raw-video gate must run inside Slurm"

BASE="${PHYSTIME_BASE:-/data/run01/sczc063/yuzibo}"
WORK_DIR="${PHYSTIME_WORK_DIR:-$(pwd)}"
OUTPUT="${PHYSTIME_GATE_OUTPUT:-${WORK_DIR}/real_gate.json}"
PYTHON="${PHYSTIME_PYTHON:-${BASE}/conda_envs/opentad/bin/python}"

: "${OPENTAD_THUMOS14_ANNOTATION:?OPENTAD_THUMOS14_ANNOTATION is required}"
: "${OPENTAD_THUMOS14_CLASS_MAP:?OPENTAD_THUMOS14_CLASS_MAP is required}"
: "${OPENTAD_THUMOS14_TRAIN_VIDEOS:?OPENTAD_THUMOS14_TRAIN_VIDEOS is required}"
: "${OPENTAD_THUMOS14_TEST_VIDEOS:?OPENTAD_THUMOS14_TEST_VIDEOS is required}"
: "${PHYSTIME_VIDEOMAE_CHECKPOINT:?PHYSTIME_VIDEOMAE_CHECKPOINT is required}"

[[ -x "${PYTHON}" ]] || fail "Python environment not found: ${PYTHON}"
[[ -f "${OPENTAD_THUMOS14_ANNOTATION}" ]] || fail "annotation file not found"
[[ -f "${OPENTAD_THUMOS14_CLASS_MAP}" ]] || fail "class map not found"
[[ -d "${OPENTAD_THUMOS14_TRAIN_VIDEOS}" ]] || fail "training video directory not found"
[[ -d "${OPENTAD_THUMOS14_TEST_VIDEOS}" ]] || fail "test video directory not found"
[[ -f "${PHYSTIME_VIDEOMAE_CHECKPOINT}" ]] || fail "VideoMAE-S checkpoint not found"

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8 >/dev/null 2>&1 || true
  module load miniforge3/24.11 >/dev/null 2>&1 || true
fi
source "${BASE}/conda_envs/opentad/bin/activate"
cd "${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"
mkdir -p "$(dirname "${OUTPUT}")"

"${PYTHON}" tools/bata/validate_phystime_adatad_track.py \
  --output "$(dirname "${OUTPUT}")/matched_contract.json"

"${PYTHON}" tools/bata/run_phystime_adatad_real_gate.py \
  --checkpoint "${PHYSTIME_VIDEOMAE_CHECKPOINT}" \
  --output "${OUTPUT}" \
  --device cuda:0 \
  --seed "${PHYSTIME_SEED:-42}" \
  --sample-index "${PHYSTIME_GATE_SAMPLE_INDEX:--1}"
