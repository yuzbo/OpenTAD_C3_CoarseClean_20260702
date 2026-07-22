#!/usr/bin/env bash
set -euo pipefail

ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
[[ -n "${SLURM_JOB_ID:-}" ]] || { printf '[GEOROUTE_DISPATCH][FAIL] requires Slurm\n' >&2; exit 2; }
[[ -d "${ROOT}/.git" ]] || { printf '[GEOROUTE_DISPATCH][FAIL] source root is not git\n' >&2; exit 2; }
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${GEOROUTE_EXPECTED_COMMIT}" ]] || { printf '[GEOROUTE_DISPATCH][FAIL] source commit mismatch\n' >&2; exit 2; }
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || { printf '[GEOROUTE_DISPATCH][FAIL] source snapshot is not clean\n' >&2; exit 2; }

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
cd "${ROOT}"
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"

python tools/bata/georoute_dag_dispatch.py \
  --action "${GEOROUTE_DAG_ACTION}" \
  --run-root "${GEOROUTE_RUN_ROOT}" \
  --source-config "${GEOROUTE_SOURCE_CONFIG}" \
  --manifest "${GEOROUTE_MANIFEST}" \
  --development-annotation "${GEOROUTE_DEVELOPMENT_ANNOTATION}" \
  --class-map "${GEOROUTE_CLASS_MAP}" \
  --development-video-root "${GEOROUTE_DEVELOPMENT_VIDEO_ROOT}" \
  --pretrained "${GEOROUTE_PRETRAINED}" \
  --expected-commit "${GEOROUTE_EXPECTED_COMMIT}"
