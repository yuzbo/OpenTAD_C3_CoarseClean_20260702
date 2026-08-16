#!/usr/bin/env bash
#SBATCH --job-name=zt-q-diag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/data/run01/sczc063/yuzibo/logs/zoomtoken_q_diag_%j.out
#SBATCH --error=/data/run01/sczc063/yuzibo/logs/zoomtoken_q_diag_%j.err

source /etc/profile
set -euo pipefail

BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
ROOT="${ZOOMTOKEN_SOURCE_ROOT:?set ZOOMTOKEN_SOURCE_ROOT}"
EXPECTED_COMMIT="${ZOOMTOKEN_EXPECTED_COMMIT:?set ZOOMTOKEN_EXPECTED_COMMIT}"
CHECKPOINT="${ZOOMTOKEN_Q_CHECKPOINT:?set ZOOMTOKEN_Q_CHECKPOINT}"
DIAGNOSTIC_ROOT="${ZOOMTOKEN_Q_DIAGNOSTIC_ROOT:?set ZOOMTOKEN_Q_DIAGNOSTIC_ROOT}"
IMAGE="${ZOOMTOKEN_CONTAINER_IMAGE:-${BASE}/containers/zoomtoken_p1_ubi9_20260815.sif}"

if [[ -z "${APPTAINER_CONTAINER:-${SINGULARITY_CONTAINER:-}}" ]]; then
  [[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
  [[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]]
  [[ -s "${CHECKPOINT}" ]]
  [[ ! -e "${DIAGNOSTIC_ROOT}" ]]
  export ZOOMTOKEN_SOURCE_VERIFIED=1
  module load apptainer/1.2.4
  exec apptainer exec --nv --bind "${BASE}:${BASE}" "${IMAGE}" \
    bash "${ROOT}/scripts/run_zoomtoken_q_seed3408_telemetry_replay_n16r4.sbatch"
fi

[[ "${ZOOMTOKEN_SOURCE_VERIFIED:-0}" == "1" ]]
source "${BASE}/conda_envs/opentad/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export ZOOMTOKEN_Q_DIAGNOSTIC_RUN_ROOT="${DIAGNOSTIC_ROOT}"
mkdir -p "${BASE}/logs"
cd "${ROOT}"

torchrun --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port="$((24000 + SLURM_JOB_ID % 20000))" \
  tools/test.py configs/adatad/thumos/zoomtoken_q_seed3408_telemetry_replay_v001.py \
  --checkpoint "${CHECKPOINT}" --seed 3408 --id 0 \
  2>&1 | tee "${BASE}/logs/zoomtoken_q_diag_${SLURM_JOB_ID}.test.log"

telemetry="${DIAGNOSTIC_ROOT}/gpu1_id0/georoute_diagnostic_telemetry.json"
[[ -s "${telemetry}" ]]
python tools/bata/summarize_zoomtoken_q_telemetry.py "${telemetry}" \
  --output "${DIAGNOSTIC_ROOT}/gpu1_id0/zoomtoken_q_telemetry_summary.json"

