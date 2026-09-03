#!/usr/bin/env bash
set -euo pipefail

# This control job seals P0 evidence only. It deliberately does not dispatch P1.
fail() {
  printf '[GEOROUTE_P0_FINALIZE][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${GEOROUTE_RUN_ROOT:?set GEOROUTE_RUN_ROOT}"
EXPECTED_COMMIT="${GEOROUTE_EXPECTED_COMMIT:?set GEOROUTE_EXPECTED_COMMIT}"
OUTPUT="${GEOROUTE_P0_FINAL_OUTPUT:-${RUN_ROOT}/control/p0_finalization.json}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'P0 finalization requires Slurm'
[[ -d "${ROOT}/.git" ]] || fail 'GeoRoute source root is not a git checkout'
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail 'source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || fail 'source snapshot is not clean'
case "${RUN_ROOT}" in /data/run01/sczc063/yuzibo/*) ;; *) fail 'run root must stay inside the remote write boundary' ;; esac
case "${OUTPUT}" in /data/run01/sczc063/yuzibo/*) ;; *) fail 'output must stay inside the remote write boundary' ;; esac
[[ ! -e "${OUTPUT}" ]] || fail 'P0 finalization namespace already exists'

DENSE="${RUN_ROOT}/p0/dense_native_parity.json"
HYBRID="${RUN_ROOT}/p0/hybrid_straight_through.json"
SCORE_FUNCTION="${RUN_ROOT}/p0/roi_score_function.json"
[[ -f "${DENSE}" && -f "${HYBRID}" && -f "${SCORE_FUNCTION}" ]] || \
  fail 'all three P0 reports must exist before finalization'

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

python -m tools.bata.finalize_georoute_p0_gate \
  --dense "${DENSE}" \
  --hybrid "${HYBRID}" \
  --score-function "${SCORE_FUNCTION}" \
  --output "${OUTPUT}"
