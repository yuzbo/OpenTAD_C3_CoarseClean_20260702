#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[GEOROUTE_STAGE][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${GEOROUTE_RUN_ROOT:?set GEOROUTE_RUN_ROOT}"
STAGE="${GEOROUTE_STAGE:?set GEOROUTE_STAGE}"
VARIANT="${GEOROUTE_VARIANT:?set GEOROUTE_VARIANT}"
SEED="${GEOROUTE_SEED:?set GEOROUTE_SEED}"
TOKEN_BUDGET="${GEOROUTE_TOKEN_BUDGET:-}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'development stage requires Slurm'
if [[ "${GEOROUTE_INNER_STEP:-0}" != "1" && "${SLURM_GPUS_ON_NODE:-1}" != "1" ]]; then
  export GEOROUTE_INNER_STEP=1
  exec srun --exact --ntasks=1 --gpus=1 --cpus-per-task=5 --mem=96000M \
    bash "${ROOT}/scripts/run_georoute_stage_slurm.sh"
fi
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || \
  fail 'stage requires one Slurm-visible GPU and must use logical cuda:0'
[[ -d "${ROOT}/.git" ]] || fail 'GeoRoute source root is not a git checkout'
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${GEOROUTE_EXPECTED_COMMIT}" ]] || fail 'source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || fail 'source snapshot is not clean'
case "${RUN_ROOT}" in /data/run01/sczc063/yuzibo/*) ;; *) fail 'run root must stay inside remote write boundary' ;; esac

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

args=(
  --stage "${STAGE}"
  --variant "${VARIANT}"
  --seed "${SEED}"
  --run-root "${RUN_ROOT}"
  --source-config "${GEOROUTE_SOURCE_CONFIG}"
  --manifest "${GEOROUTE_MANIFEST}"
  --development-annotation "${GEOROUTE_DEVELOPMENT_ANNOTATION}"
  --class-map "${GEOROUTE_CLASS_MAP}"
  --development-video-root "${GEOROUTE_DEVELOPMENT_VIDEO_ROOT}"
  --pretrained "${GEOROUTE_PRETRAINED}"
  --expected-commit "${GEOROUTE_EXPECTED_COMMIT}"
)
if [[ -n "${TOKEN_BUDGET}" ]]; then
  args+=(--token-budget "${TOKEN_BUDGET}")
fi
python tools/bata/georoute_stage_runner.py "${args[@]}"
