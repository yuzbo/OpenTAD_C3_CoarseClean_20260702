#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

source /etc/profile
set -euo pipefail

fail() {
  printf '[SCNR_RESIDUAL_CENTERING_FINALIZER] ERROR: %s\n' "$*" >&2
  exit 1
}

BASE="${GEOROUTE_BASE:-/data/run01/sczc063/yuzibo}"
ROOT="${GEOROUTE_SOURCE_ROOT:?set GEOROUTE_SOURCE_ROOT}"
RUN_ROOT="${SCNR_RESIDUAL_CENTERING_TRAINING_RUN_ROOT:?set SCNR_RESIDUAL_CENTERING_TRAINING_RUN_ROOT}"
EXPECTED_COMMIT="${GEOROUTE_EXPECTED_COMMIT:?set GEOROUTE_EXPECTED_COMMIT}"
NONE_JOB_ID="${SCNR_RESIDUAL_CENTERING_NONE_JOB_ID:?set SCNR_RESIDUAL_CENTERING_NONE_JOB_ID}"
CENTER_JOB_ID="${SCNR_RESIDUAL_CENTERING_CENTER_JOB_ID:?set SCNR_RESIDUAL_CENTERING_CENTER_JOB_ID}"
DEPLOYMENT="${SCNR_RESIDUAL_CENTERING_DEPLOYMENT:-${RUN_ROOT}/control/deployment.json}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"

case "${PRECHECK_ONLY}" in
  0|1) ;;
  *) fail 'PRECHECK_ONLY must be 0 or 1' ;;
esac
[[ -d "${ROOT}/.git" || -f "${ROOT}/.git" ]] || fail 'source root is not a Git checkout'
module load cuda/11.8
module load miniforge3/24.11
# shellcheck disable=SC1091
source "${BASE}/conda_envs/opentad/bin/activate"
cd "${ROOT}"

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  python -m py_compile \
    tools/bata/georoute_residual_centering_training_contract.py \
    tools/bata/run_georoute_residual_centering_training.py \
    tools/bata/deploy_georoute_residual_centering_training.py \
    tools/bata/finalize_georoute_residual_centering_training.py
  printf '[SCNR_RESIDUAL_CENTERING_FINALIZER] STATIC PRECHECK PASS\n'
  exit 0
fi

[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'finalizer must run inside Slurm'
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || \
  fail 'finalizer requires its one scheduling GPU'
[[ -f "${DEPLOYMENT}" ]] || fail 'deployment receipt is missing'
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  python tools/bata/finalize_georoute_residual_centering_training.py \
    --run-root "${RUN_ROOT}" \
    --expected-commit "${EXPECTED_COMMIT}" \
    --deployment "${DEPLOYMENT}" \
    --none-job-id "${NONE_JOB_ID}" \
    --center-job-id "${CENTER_JOB_ID}"
