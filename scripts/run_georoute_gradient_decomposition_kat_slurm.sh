#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[GEOROUTE_GRADIENT_DECOMPOSITION_KAT][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${GEOROUTE_GRADIENT_DECOMPOSITION_KAT_ROOT:?set GEOROUTE_GRADIENT_DECOMPOSITION_KAT_ROOT}"
EXPECTED_COMMIT="${GEOROUTE_EXPECTED_COMMIT:?set GEOROUTE_EXPECTED_COMMIT}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'gradient-decomposition KAT requires Slurm'
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || \
  fail 'gradient-decomposition KAT requires one Slurm-visible GPU'
[[ -e "${ROOT}/.git" ]] || fail 'gradient-decomposition KAT source is not Git'
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail 'gradient-decomposition KAT source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || \
  fail 'gradient-decomposition KAT source snapshot is not clean'
case "${RUN_ROOT}" in
  /data/run01/sczc063/yuzibo/*) ;;
  *) fail 'gradient-decomposition KAT root left the write boundary' ;;
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
python -m tools.bata.run_georoute_gradient_decomposition_kat \
  --run-root "${RUN_ROOT}" \
  --expected-commit "${EXPECTED_COMMIT}"
