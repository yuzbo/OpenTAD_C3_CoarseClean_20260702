#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[SPATIAL_ZOOM_S1_POWER_DIAG][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="${SPATIAL_ZOOM_S1_PROFILE_SOURCE_ROOT:?set SPATIAL_ZOOM_S1_PROFILE_SOURCE_ROOT}"
OUTPUT="${SPATIAL_ZOOM_S1_POWER_DIAGNOSTIC_OUTPUT:?set SPATIAL_ZOOM_S1_POWER_DIAGNOSTIC_OUTPUT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "diagnostic requires a Slurm allocation"
[[ "${SLURM_GPUS_ON_NODE:-}" == "1" ]] || fail "diagnostic requires one allocated GPU"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || \
  fail "diagnostic requires exactly one CUDA-visible GPU"
[[ ! -e "${OUTPUT}" ]] || fail "diagnostic output already exists"

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"

[[ -d "${ROOT}/.git" || -f "${ROOT}/.git" ]] || fail "source root is not Git"
COMMIT="$(git -C "${ROOT}" rev-parse HEAD)"
[[ -z "$(git -C "${ROOT}" status --porcelain --untracked-files=all)" ]] || \
  fail "diagnostic source root must be clean"

mkdir -p "$(dirname "${OUTPUT}")"
cd "${ROOT}"
PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  python tools/bata/spatial_zoom_s1_power.py \
    --output "${OUTPUT}" \
    --logical-gpu-id 0 \
    --interval-ms 20 \
    --duration-seconds 10 \
    --code-commit "${COMMIT}"
