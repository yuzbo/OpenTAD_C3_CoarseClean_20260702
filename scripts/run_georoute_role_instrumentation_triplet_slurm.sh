#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[GEOROUTE_ROLE_TRIPLET][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
CELL_ROOT="${GEOROUTE_ROLE_TRIPLET_CELL_ROOT:?set GEOROUTE_ROLE_TRIPLET_CELL_ROOT}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'causal triplet requires Slurm'
[[ -d "${ROOT}/.git" ]] || fail 'source root is not a git checkout'
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${GEOROUTE_EXPECTED_COMMIT}" ]] || \
  fail 'source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || \
  fail 'source snapshot is not clean'
case "${CELL_ROOT}" in
  /data/run01/sczc063/yuzibo/*) ;;
  *) fail 'cell root must stay inside remote write boundary' ;;
esac
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || \
  fail 'causal triplet requires one Slurm-visible GPU and logical cuda:0'

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

python -m tools.bata.run_georoute_role_instrumentation_triplet \
  --variant "${GEOROUTE_ROLE_TRIPLET_VARIANT}" \
  --seed "${GEOROUTE_ROLE_TRIPLET_SEED}" \
  --cell-root "${CELL_ROOT}" \
  --source-run-root "${GEOROUTE_SOURCE_RUN_ROOT}" \
  --source-bound-config "${GEOROUTE_ROLE_TRIPLET_SOURCE_CONFIG}" \
  --source-bound-config-sha256 "${GEOROUTE_ROLE_TRIPLET_SOURCE_CONFIG_SHA256}" \
  --source-checkpoint "${GEOROUTE_ROLE_TRIPLET_SOURCE_CHECKPOINT}" \
  --source-checkpoint-sha256 "${GEOROUTE_ROLE_TRIPLET_SOURCE_CHECKPOINT_SHA256}" \
  --source-prediction "${GEOROUTE_ROLE_TRIPLET_SOURCE_PREDICTION}" \
  --source-prediction-sha256 "${GEOROUTE_ROLE_TRIPLET_SOURCE_PREDICTION_SHA256}" \
  --source-population-sha256 "${GEOROUTE_ROLE_TRIPLET_SOURCE_POPULATION_SHA256}" \
  --source-dataset-count "${GEOROUTE_ROLE_TRIPLET_SOURCE_DATASET_COUNT}" \
  --source-experiment-commit "${GEOROUTE_SOURCE_EXPERIMENT_COMMIT}" \
  --expected-commit "${GEOROUTE_EXPECTED_COMMIT}"
