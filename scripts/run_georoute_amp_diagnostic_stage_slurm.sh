#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[GEOROUTE_AMP_DIAGNOSTIC][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${GEOROUTE_AMP_DIAGNOSTIC_RUN_ROOT:?set GEOROUTE_AMP_DIAGNOSTIC_RUN_ROOT}"
ARM="${GEOROUTE_AMP_DIAGNOSTIC_ARM:?set GEOROUTE_AMP_DIAGNOSTIC_ARM}"
EXPECTED_COMMIT="${GEOROUTE_EXPECTED_COMMIT:?set GEOROUTE_EXPECTED_COMMIT}"
PROTOCOL_PROFILE="${GEOROUTE_AMP_PROTOCOL_PROFILE:-diagnostic}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'diagnostic stage requires Slurm'
if [[ "${GEOROUTE_INNER_STEP:-0}" != "1" && "${SLURM_GPUS_ON_NODE:-1}" != "1" ]]; then
  export GEOROUTE_INNER_STEP=1
  exec srun --exact --ntasks=1 --gpus=1 --cpus-per-task=5 --mem=96000M \
    bash "${ROOT}/scripts/run_georoute_amp_diagnostic_stage_slurm.sh"
fi
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || \
  fail 'diagnostic stage requires one Slurm-visible GPU and logical cuda:0'
[[ -e "${ROOT}/.git" ]] || fail 'diagnostic source root is not a git checkout'
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail 'diagnostic source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || \
  fail 'diagnostic source snapshot is not clean'
case "${RUN_ROOT}" in
  /data/run01/sczc063/yuzibo/*) ;;
  *) fail 'diagnostic run root must stay inside the remote write boundary' ;;
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

EXTRA_ARGS=()
if [[ -n "${GEOROUTE_OFFICIAL_REFERENCE_CONFIG:-}" ]]; then
  EXTRA_ARGS+=(--official-reference-config "${GEOROUTE_OFFICIAL_REFERENCE_CONFIG}")
fi

python -m tools.bata.georoute_amp_diagnostic_stage_runner \
  --arm "${ARM}" \
  --run-root "${RUN_ROOT}" \
  --source-config "${GEOROUTE_SOURCE_CONFIG}" \
  --manifest "${GEOROUTE_MANIFEST}" \
  --development-annotation "${GEOROUTE_DEVELOPMENT_ANNOTATION}" \
  --class-map "${GEOROUTE_CLASS_MAP}" \
  --development-video-root "${GEOROUTE_DEVELOPMENT_VIDEO_ROOT}" \
  --pretrained "${GEOROUTE_PRETRAINED}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --protocol-profile "${PROTOCOL_PROFILE}" \
  "${EXTRA_ARGS[@]}"
