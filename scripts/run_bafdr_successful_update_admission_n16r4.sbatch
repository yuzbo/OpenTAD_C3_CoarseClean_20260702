#!/usr/bin/env bash
#SBATCH --job-name=bafdr-update-admission
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
source /etc/profile
set -euo pipefail
cd "${PROJECT_DIR:?PROJECT_DIR is required}"
[[ "$(git rev-parse HEAD)" == "${BAFDR_EXPECTED_COMMIT:?exact SHA is required}" ]]
[[ -z "$(git status --porcelain)" ]]
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export YUZIBO_ROOT=/data/run01/sczc063/yuzibo
export ZOOMTOKEN_RUN_ROOT="${ZOOMTOKEN_RUN_ROOT:?external output root is required}"
python -m pytest tests/test_bafdr_successful_updates.py tests/test_bafdr_successful_updates_runtime.py tests/test_bafdr_k16_static_protocol.py tests/test_c3_coarse_classifier_model_matrix.py tests/test_c3_asformer_delta_ledger_full_train.py -q
export PRECHECK_ONLY=1 BAFDR_REQUIRE_SCREEN_GATE=0
if [[ "${BAFDR_ADMISSION_FULL_ONLY:-0}" == 1 ]]; then
  # FULL requires the new, actually complete D160 teacher and explicit bindings.
  arms=(full)
else
  arms=(d160 g96 u16_uniform_a0 late nokd)
fi
for arm in "${arms[@]}"; do
  bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh precheck "configs/adatad/thumos/bafdr_k16_${arm}_seed4407.py"
done
echo BAFDR_SUCCESSFUL_UPDATE_PRECHECKS_OK
