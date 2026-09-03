#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[GEOROUTE_ESTIMATOR_PILOT_CONTROL][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${GEOROUTE_PILOT_RUN_ROOT:?set GEOROUTE_PILOT_RUN_ROOT}"
EXPECTED_COMMIT="${GEOROUTE_EXPECTED_COMMIT:?set GEOROUTE_EXPECTED_COMMIT}"
ACTION="${GEOROUTE_PILOT_ACTION:?set GEOROUTE_PILOT_ACTION}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'pilot control action requires Slurm'
[[ -d "${ROOT}/.git" ]] || fail 'pilot source root is not a git checkout'
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail 'pilot source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || \
  fail 'pilot source snapshot is not clean'
case "${RUN_ROOT}" in
  /data/run01/sczc063/yuzibo/*) ;;
  *) fail 'pilot run root must stay inside the remote write boundary' ;;
esac

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
cd "${ROOT}"
if command -v module >/dev/null 2>&1; then
  module load miniforge3/24.11
fi
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"

case "${ACTION}" in
  p0-finalize)
    python -m tools.bata.finalize_georoute_estimator_pilot_p0 \
      --run-root "${RUN_ROOT}" \
      --expected-commit "${EXPECTED_COMMIT}"
    ;;
  finalize)
    python -m tools.bata.finalize_georoute_estimator_pilot \
      --run-root "${RUN_ROOT}" \
      --expected-commit "${EXPECTED_COMMIT}"
    ;;
  *)
    fail "unsupported pilot control action: ${ACTION}"
    ;;
esac
