#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[GEOROUTE_GRADIENT_DECOMPOSITION][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${GEOROUTE_GRADIENT_DECOMPOSITION_RUN_ROOT:?set GEOROUTE_GRADIENT_DECOMPOSITION_RUN_ROOT}"
ARM="${GEOROUTE_GRADIENT_DECOMPOSITION_ARM:?set GEOROUTE_GRADIENT_DECOMPOSITION_ARM}"
EXPECTED_COMMIT="${GEOROUTE_EXPECTED_COMMIT:?set GEOROUTE_EXPECTED_COMMIT}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'gradient-decomposition stage requires Slurm'
if [[ "${GEOROUTE_INNER_STEP:-0}" != "1" && "${SLURM_GPUS_ON_NODE:-1}" != "1" ]]; then
  export GEOROUTE_INNER_STEP=1
  exec srun --exact --ntasks=1 --gpus=1 --cpus-per-task=5 --mem=96000M \
    bash "${ROOT}/scripts/run_georoute_gradient_decomposition_stage_slurm.sh"
fi
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || \
  fail 'gradient-decomposition stage requires one Slurm-visible GPU'
[[ -e "${ROOT}/.git" ]] || fail 'gradient-decomposition source is not a git checkout'
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail 'gradient-decomposition source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || \
  fail 'gradient-decomposition source snapshot is not clean'
case "${RUN_ROOT}" in
  /data/run01/sczc063/yuzibo/*) ;;
  *) fail 'gradient-decomposition run root left the write boundary' ;;
esac

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
cd "${ROOT}"
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"
python -c 'import numpy; assert numpy.__version__ == "1.23.5", numpy.__version__'

python -m tools.bata.georoute_gradient_decomposition_stage_runner \
  --arm "${ARM}" \
  --run-root "${RUN_ROOT}" \
  --source-config "${GEOROUTE_SOURCE_CONFIG}" \
  --manifest "${GEOROUTE_MANIFEST}" \
  --development-annotation "${GEOROUTE_DEVELOPMENT_ANNOTATION}" \
  --class-map "${GEOROUTE_CLASS_MAP}" \
  --development-video-root "${GEOROUTE_DEVELOPMENT_VIDEO_ROOT}" \
  --pretrained "${GEOROUTE_PRETRAINED}" \
  --official-reference-config "${GEOROUTE_OFFICIAL_REFERENCE_CONFIG}" \
  --expected-commit "${EXPECTED_COMMIT}"
