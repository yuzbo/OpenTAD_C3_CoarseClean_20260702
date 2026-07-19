#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[NATIVE_CROP_S1_GATE][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
OUT_ROOT="${NATIVE_CROP_S1_OUT_ROOT:-${BASE}/native_crop_s1/gate}"
MANIFEST="${NATIVE_CROP_S1_MANIFEST:-}"
VIDEO_ROOT="${NATIVE_CROP_S1_VIDEO_ROOT:-}"
ANNOTATION="${NATIVE_CROP_S1_ANNOTATION:-}"
CLASS_MAP="${NATIVE_CROP_S1_CLASS_MAP:-${BASE}/thumos14/annotations/category_idx.txt}"
CONFIG="${ROOT}/configs/adatad/thumos/native_crop_s1_center_videomae_s_768x1_adapter.py"
EXPECTED_COMMIT="${NATIVE_CROP_S1_EXPECTED_COMMIT:-}"
AUDITED_SOURCE_PATHS=(
  "configs/_base_/datasets/thumos-14/e2e_train_trunc_test_sw_256x224x224.py"
  "configs/_base_/models/actionformer.py"
  "configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py"
  "configs/adatad/thumos/native_crop_s1_center_videomae_s_768x1_adapter.py"
  "opentad/datasets/transforms/__init__.py"
  "opentad/datasets/transforms/formatting.py"
  "opentad/datasets/transforms/native_crop.py"
  "opentad/models/backbones/__init__.py"
  "opentad/models/backbones/native_crop_wrapper.py"
  "opentad/models/builder.py"
  "tools/bata/build_native_crop_s1_development_annotation.py"
  "tools/bata/native_crop_s1_contract.py"
  "tools/bata/native_crop_s1_geometry_census.py"
  "tools/bata/run_native_crop_s1_precheck.py"
  "scripts/run_native_crop_s1_gate_slurm.sh"
  "tests/test_native_crop_s1_vertical_slice.py"
)

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "the full Native-Crop gate requires Slurm"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || \
  fail "the gate requires exactly one Slurm-visible GPU"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || \
  fail "NATIVE_CROP_S1_EXPECTED_COMMIT must be one full Git commit"
ACTUAL_COMMIT="$(git -C "${ROOT}" rev-parse HEAD)"
[[ "${ACTUAL_COMMIT}" == "${EXPECTED_COMMIT}" ]] || \
  fail "snapshot commit ${ACTUAL_COMMIT} != expected ${EXPECTED_COMMIT}"
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || \
  fail "formal Native-Crop snapshot is not completely clean"
git -C "${ROOT}" ls-files --error-unmatch -- "${AUDITED_SOURCE_PATHS[@]}" \
  >/dev/null || fail "formal Native-Crop executable source is not tracked"
[[ -n "${MANIFEST}" && -f "${MANIFEST}" ]] || fail "NATIVE_CROP_S1_MANIFEST is missing"
[[ -n "${VIDEO_ROOT}" && -d "${VIDEO_ROOT}" ]] || fail "NATIVE_CROP_S1_VIDEO_ROOT is missing"
[[ -n "${ANNOTATION}" && -f "${ANNOTATION}" ]] || \
  fail "NATIVE_CROP_S1_ANNOTATION must be the frozen development-only file"
[[ -f "${CLASS_MAP}" ]] || fail "Native-Crop class map is missing"
case "${OUT_ROOT}" in
  /data/run01/sczc063/yuzibo|/data/run01/sczc063/yuzibo/*) ;;
  *) fail "output must stay under /data/run01/sczc063/yuzibo" ;;
esac

cd "${ROOT}"
mkdir -p "${OUT_ROOT}"
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"

python -m py_compile \
  opentad/datasets/transforms/native_crop.py \
  opentad/models/backbones/native_crop_wrapper.py \
  tools/bata/build_native_crop_s1_development_annotation.py \
  tools/bata/native_crop_s1_contract.py \
  tools/bata/native_crop_s1_geometry_census.py \
  tools/bata/run_native_crop_s1_precheck.py

python -m pytest -p no:cacheprovider \
  tests/test_native_crop_s1_vertical_slice.py \
  -q

python tools/bata/native_crop_s1_geometry_census.py \
  --manifest "${MANIFEST}" \
  --video-root "${VIDEO_ROOT}" \
  --crop-sizes 96 112 128 \
  --output "${OUT_ROOT}/geometry_census.json"

python tools/bata/run_native_crop_s1_precheck.py \
  --config "${CONFIG}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --device cuda:0 \
  --amp \
  --manifest "${MANIFEST}" \
  --geometry-census "${OUT_ROOT}/geometry_census.json" \
  --annotation "${ANNOTATION}" \
  --class-map "${CLASS_MAP}" \
  --video-root "${VIDEO_ROOT}" \
  --output "${OUT_ROOT}/full_model_precheck.json"

printf '[NATIVE_CROP_S1_GATE] PASS output=%s\n' "${OUT_ROOT}"
