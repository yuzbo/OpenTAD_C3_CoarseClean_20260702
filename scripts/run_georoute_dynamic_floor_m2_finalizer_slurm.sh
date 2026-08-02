#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

set -euo pipefail

fail() {
  printf '[DYNAMIC_FLOOR_M2_FINALIZER] ERROR: %s\n' "$*" >&2
  exit 1
}

BASE="${GEOROUTE_BASE:-/data/run01/sczc063/yuzibo}"
ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
RUN_ROOT="${GEOROUTE_DYNAMIC_FLOOR_M2_RUN_ROOT:?set GEOROUTE_DYNAMIC_FLOOR_M2_RUN_ROOT}"
EXPECTED_COMMIT="${GEOROUTE_EXPECTED_COMMIT:?set GEOROUTE_EXPECTED_COMMIT}"
DEPLOYMENT="${RUN_ROOT}/control/deployment.json"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"

case "${PRECHECK_ONLY}" in
  0|1) ;;
  *) fail "PRECHECK_ONLY must be 0 or 1" ;;
esac
[[ -d "${ROOT}/.git" || -f "${ROOT}/.git" ]] || fail "source root is not a Git checkout"
if command -v module >/dev/null 2>&1; then
  module load miniforge3/24.11
fi
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"
cd "${ROOT}"

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  python -m py_compile \
    tools/bata/finalize_georoute_dynamic_floor_m2.py \
    tools/bata/georoute_dynamic_floor_m2_contract.py
  printf '[DYNAMIC_FLOOR_M2_FINALIZER] STATIC PRECHECK PASS\n'
  exit 0
fi

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "finalizer must run inside Slurm"
[[ -f "${DEPLOYMENT}" ]] || fail "deployment receipt is missing"
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  python tools/bata/finalize_georoute_dynamic_floor_m2.py \
    --run-root "${RUN_ROOT}" \
    --deployment "${DEPLOYMENT}" \
    --expected-commit "${EXPECTED_COMMIT}"
