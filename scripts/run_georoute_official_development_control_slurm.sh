#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[GEOROUTE_OFFICIAL_DEVELOPMENT_CONTROL][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${GEOROUTE_OFFICIAL_DEVELOPMENT_RUN_ROOT:?set GEOROUTE_OFFICIAL_DEVELOPMENT_RUN_ROOT}"
EXPECTED_COMMIT="${GEOROUTE_EXPECTED_COMMIT:?set GEOROUTE_EXPECTED_COMMIT}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'formal finalizer requires Slurm'
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

python -m tools.bata.finalize_georoute_official_development \
  --run-root "${RUN_ROOT}" \
  --expected-commit "${EXPECTED_COMMIT}"
