#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[GEOROUTE_HYBRID_CAUSAL_STAGE][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${GEOROUTE_HYBRID_CAUSAL_RUN_ROOT:?set GEOROUTE_HYBRID_CAUSAL_RUN_ROOT}"
ARM="${GEOROUTE_HYBRID_CAUSAL_ARM:?set GEOROUTE_HYBRID_CAUSAL_ARM}"
EXPECTED_COMMIT="${GEOROUTE_EXPECTED_COMMIT:?set GEOROUTE_EXPECTED_COMMIT}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'stage requires Slurm'
VISIBLE_COUNT="$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES:-}")"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${VISIBLE_COUNT}" == "2" ]] || \
  fail 'stage requires exactly two Slurm-visible GPUs'
[[ -e "${ROOT}/.git" ]] || fail 'source root is not a git checkout'
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail 'source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || \
  fail 'source snapshot is not clean'
case "${RUN_ROOT}" in
  /data/run01/sczc063/yuzibo/*) ;;
  *) fail 'run root leaves remote write boundary' ;;
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

python -m tools.bata.georoute_hybrid_causal_stage_runner \
  --arm "${ARM}" \
  --run-root "${RUN_ROOT}" \
  --source-config "${GEOROUTE_SOURCE_CONFIG}" \
  --manifest "${GEOROUTE_MANIFEST}" \
  --development-annotation "${GEOROUTE_DEVELOPMENT_ANNOTATION}" \
  --class-map "${GEOROUTE_CLASS_MAP}" \
  --development-video-root "${GEOROUTE_DEVELOPMENT_VIDEO_ROOT}" \
  --pretrained "${GEOROUTE_PRETRAINED}" \
  --expected-commit "${EXPECTED_COMMIT}"
