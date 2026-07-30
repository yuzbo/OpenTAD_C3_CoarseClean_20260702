#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[GEOROUTE_GRADIENT_DECOMPOSITION_CONTROL][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${GEOROUTE_GRADIENT_DECOMPOSITION_RUN_ROOT:?set GEOROUTE_GRADIENT_DECOMPOSITION_RUN_ROOT}"
EXPECTED_COMMIT="${GEOROUTE_EXPECTED_COMMIT:?set GEOROUTE_EXPECTED_COMMIT}"
ACTION="${GEOROUTE_GRADIENT_DECOMPOSITION_ACTION:?set GEOROUTE_GRADIENT_DECOMPOSITION_ACTION}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'gradient-decomposition control requires Slurm'
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
  module load miniforge3/24.11
fi
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"

case "${ACTION}" in
  finalize)
    python -m tools.bata.finalize_georoute_gradient_decomposition \
      --run-root "${RUN_ROOT}" \
      --expected-commit "${EXPECTED_COMMIT}"
    ;;
  *)
    fail "unsupported gradient-decomposition control action: ${ACTION}"
    ;;
esac
