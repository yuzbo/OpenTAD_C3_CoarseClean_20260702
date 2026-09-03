#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[SCNR_RESIDUAL_CENTERING_PROBE][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
CELL_ROOT="${SCNR_RESIDUAL_CENTERING_CELL_ROOT:?set SCNR_RESIDUAL_CENTERING_CELL_ROOT}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'residual-centering probe requires Slurm'
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
  fail 'residual-centering probe requires one Slurm-visible GPU and logical cuda:0'

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

python -m tools.bata.run_georoute_residual_centering_probe \
  --variant "${SCNR_RESIDUAL_CENTERING_VARIANT}" \
  --seed "${SCNR_RESIDUAL_CENTERING_SEED}" \
  --cell-root "${CELL_ROOT}" \
  --source-run-root "${GEOROUTE_SOURCE_RUN_ROOT}" \
  --source-bound-config "${SCNR_RESIDUAL_CENTERING_SOURCE_CONFIG}" \
  --source-bound-config-sha256 "${SCNR_RESIDUAL_CENTERING_SOURCE_CONFIG_SHA256}" \
  --source-checkpoint "${SCNR_RESIDUAL_CENTERING_SOURCE_CHECKPOINT}" \
  --source-checkpoint-sha256 "${SCNR_RESIDUAL_CENTERING_SOURCE_CHECKPOINT_SHA256}" \
  --source-prediction "${SCNR_RESIDUAL_CENTERING_SOURCE_PREDICTION}" \
  --source-prediction-sha256 "${SCNR_RESIDUAL_CENTERING_SOURCE_PREDICTION_SHA256}" \
  --source-population-sha256 "${SCNR_RESIDUAL_CENTERING_SOURCE_POPULATION_SHA256}" \
  --source-dataset-count "${SCNR_RESIDUAL_CENTERING_SOURCE_DATASET_COUNT}" \
  --source-experiment-commit "${GEOROUTE_SOURCE_EXPERIMENT_COMMIT}" \
  --expected-commit "${GEOROUTE_EXPECTED_COMMIT}"
