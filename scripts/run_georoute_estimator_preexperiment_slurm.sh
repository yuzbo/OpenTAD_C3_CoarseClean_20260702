#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '[GEOROUTE_PREEXPERIMENT][FAIL] %s\n' "$*" >&2
  exit 2
}

ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${GEOROUTE_PREEXPERIMENT_ROOT:?set GEOROUTE_PREEXPERIMENT_ROOT}"
ACTION="${GEOROUTE_PREEXPERIMENT_ACTION:?set GEOROUTE_PREEXPERIMENT_ACTION}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'preexperiment action requires Slurm'
[[ -d "${ROOT}/.git" ]] || fail 'source root is not a git checkout'
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${GEOROUTE_EXPECTED_COMMIT}" ]] || \
  fail 'source commit mismatch'
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || \
  fail 'source snapshot is not clean'
case "${RUN_ROOT}" in
  /data/run01/sczc063/yuzibo/*) ;;
  *) fail 'run root must stay inside remote write boundary' ;;
esac

if [[ "${ACTION}" == "phase-m" && "${GEOROUTE_INNER_STEP:-0}" != "1" && "${SLURM_GPUS_ON_NODE:-1}" != "1" ]]; then
  export GEOROUTE_INNER_STEP=1
  exec srun --exact --ntasks=1 --gpus=1 --cpus-per-task=5 --mem=96000M \
    bash "${ROOT}/scripts/run_georoute_estimator_preexperiment_slurm.sh"
fi
if [[ "${ACTION}" == "phase-m" ]]; then
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || \
    fail 'Phase M requires one Slurm-visible GPU and logical cuda:0'
fi

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

case "${ACTION}" in
  kat)
    python -m tools.bata.run_georoute_estimator_kat \
      --output "${RUN_ROOT}/control/estimator_representation_kat.json" \
      --expected-commit "${GEOROUTE_EXPECTED_COMMIT}"
    ;;
  decode-census)
    python -m tools.bata.run_georoute_decode_census \
      --bound-config "${GEOROUTE_CENSUS_BOUND_CONFIG}" \
      --expected-bound-config-sha256 "${GEOROUTE_CENSUS_BOUND_CONFIG_SHA256}" \
      --expected-commit "${GEOROUTE_EXPECTED_COMMIT}" \
      --source-experiment-commit "${GEOROUTE_SOURCE_EXPERIMENT_COMMIT}" \
      --passes "${GEOROUTE_CENSUS_PASSES:-2}" \
      --output "${RUN_ROOT}/control/decode_census.json"
    ;;
  phase-m)
    PHASE_M_ROLE_ARGS=()
    case "${GEOROUTE_PHASE_M_ROLE_CALIBRATION_TELEMETRY:-0}" in
      0) ;;
      1)
        [[ -n "${GEOROUTE_PHASE_M_SOURCE_POPULATION_SHA256:-}" ]] || \
          fail 'role calibration replay requires source population SHA-256'
        PHASE_M_ROLE_ARGS+=(
          --source-population-sha256 "${GEOROUTE_PHASE_M_SOURCE_POPULATION_SHA256}"
          --role-calibration-telemetry
        )
        ;;
      *) fail 'GEOROUTE_PHASE_M_ROLE_CALIBRATION_TELEMETRY must be 0 or 1' ;;
    esac
    python -m tools.bata.run_georoute_phase_m_replay \
      --variant "${GEOROUTE_PHASE_M_VARIANT}" \
      --seed "${GEOROUTE_PHASE_M_SEED}" \
      --cell-root "${RUN_ROOT}/phase_m/${GEOROUTE_PHASE_M_VARIANT}" \
      --source-run-root "${GEOROUTE_SOURCE_RUN_ROOT}" \
      --source-bound-config "${GEOROUTE_PHASE_M_SOURCE_CONFIG}" \
      --source-bound-config-sha256 "${GEOROUTE_PHASE_M_SOURCE_CONFIG_SHA256}" \
      --source-checkpoint "${GEOROUTE_PHASE_M_SOURCE_CHECKPOINT}" \
      --source-checkpoint-sha256 "${GEOROUTE_PHASE_M_SOURCE_CHECKPOINT_SHA256}" \
      --source-prediction "${GEOROUTE_PHASE_M_SOURCE_PREDICTION}" \
      --source-prediction-sha256 "${GEOROUTE_PHASE_M_SOURCE_PREDICTION_SHA256}" \
      --source-experiment-commit "${GEOROUTE_SOURCE_EXPERIMENT_COMMIT}" \
      --expected-commit "${GEOROUTE_EXPECTED_COMMIT}" \
      "${PHASE_M_ROLE_ARGS[@]}"
    ;;
  finalize)
    python -m tools.bata.finalize_georoute_estimator_preexperiment \
      --run-root "${RUN_ROOT}" \
      --expected-commit "${GEOROUTE_EXPECTED_COMMIT}"
    ;;
  *)
    fail "unsupported preexperiment action ${ACTION}"
    ;;
esac
