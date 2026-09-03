#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[GEOROUTE_OFFICIAL_DEVELOPMENT][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${GEOROUTE_OFFICIAL_DEVELOPMENT_RUN_ROOT:?set GEOROUTE_OFFICIAL_DEVELOPMENT_RUN_ROOT}"
ARM="${GEOROUTE_OFFICIAL_DEVELOPMENT_ARM:?set GEOROUTE_OFFICIAL_DEVELOPMENT_ARM}"
SEED="${GEOROUTE_OFFICIAL_DEVELOPMENT_SEED:?set GEOROUTE_OFFICIAL_DEVELOPMENT_SEED}"
EXPECTED_COMMIT="${GEOROUTE_EXPECTED_COMMIT:?set GEOROUTE_EXPECTED_COMMIT}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'formal development stage requires Slurm'
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ "${GEOROUTE_INNER_STEP:-0}" != "1" && "${#visible_gpus[@]}" -ne 2 ]]; then
  export GEOROUTE_INNER_STEP=1
  exec srun --exact --ntasks=1 --gpus=2 --cpus-per-task=10 --mem=192000M \
    bash "${ROOT}/scripts/run_georoute_official_development_stage_slurm.sh"
fi
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${#visible_gpus[@]}" -eq 2 ]] || \
  fail 'formal development requires two Slurm-visible GPUs'
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

python -m tools.bata.georoute_official_development_stage_runner \
  --arm "${ARM}" \
  --seed "${SEED}" \
  --run-root "${RUN_ROOT}" \
  --source-config "${GEOROUTE_SOURCE_CONFIG}" \
  --manifest "${GEOROUTE_MANIFEST}" \
  --development-annotation "${GEOROUTE_DEVELOPMENT_ANNOTATION}" \
  --class-map "${GEOROUTE_CLASS_MAP}" \
  --development-video-root "${GEOROUTE_DEVELOPMENT_VIDEO_ROOT}" \
  --pretrained "${GEOROUTE_PRETRAINED}" \
  --expected-commit "${EXPECTED_COMMIT}"
