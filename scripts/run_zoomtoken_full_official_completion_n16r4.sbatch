#!/usr/bin/env bash
#SBATCH --job-name=zt-complete
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --time=72:00:00
#SBATCH --array=0-14%8
#SBATCH --output=/data/run01/sczc063/yuzibo/logs/zoomtoken_complete_%A_%a.out
#SBATCH --error=/data/run01/sczc063/yuzibo/logs/zoomtoken_complete_%A_%a.err

source /etc/profile
set -euo pipefail

BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
ROOT="${ZOOMTOKEN_SOURCE_ROOT:?set ZOOMTOKEN_SOURCE_ROOT}"
EXPECTED_COMMIT="${ZOOMTOKEN_EXPECTED_COMMIT:?set ZOOMTOKEN_EXPECTED_COMMIT}"
RUN_ROOT="${ZOOMTOKEN_COMPLETION_RUN_ROOT:?set ZOOMTOKEN_COMPLETION_RUN_ROOT}"
IMAGE="${ZOOMTOKEN_CONTAINER_IMAGE:-${BASE}/containers/zoomtoken_p1_ubi9_20260815.sif}"

if [[ -z "${APPTAINER_CONTAINER:-${SINGULARITY_CONTAINER:-}}" ]]; then
  [[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || {
    echo "source commit mismatch" >&2
    exit 2
  }
  [[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || {
    echo "source checkout is not clean" >&2
    exit 2
  }
  export ZOOMTOKEN_SOURCE_VERIFIED=1
  module load apptainer/1.2.4
  exec apptainer exec --nv --bind "${BASE}:${BASE}" "${IMAGE}" \
    bash "${ROOT}/scripts/run_zoomtoken_full_official_completion_n16r4.sbatch"
fi

[[ "${ZOOMTOKEN_SOURCE_VERIFIED:-0}" == "1" ]] || {
  echo "source checkout was not verified before entering the container" >&2
  exit 2
}
source "${BASE}/conda_envs/opentad/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

arms=(DO DN U R Q)
configs=(
  zoomtoken_full_official_do_seed3407_v001.py
  zoomtoken_full_official_dn_seed3407_v001.py
  zoomtoken_full_official_u_seed3407_v001.py
  zoomtoken_full_official_r_seed3407_v001.py
  zoomtoken_full_official_q_seed3407_v001.py
)
seeds=(3407 3408 3409)
source_roots=(
  "${BASE}/zoomtoken_full_official_b88a11ba_seed3407_20260815_070546"
  "${BASE}/zoomtoken_full_official_8b4f4ce9_seed3408_20260815_071707"
  "${BASE}/zoomtoken_full_official_8b4f4ce9_seed3409_20260815_071707"
)

arm_index=$((SLURM_ARRAY_TASK_ID % 5))
seed_index=$((SLURM_ARRAY_TASK_ID / 5))
arm="${arms[${arm_index}]}"
config="${ROOT}/configs/adatad/thumos/${configs[${arm_index}]}"
seed="${seeds[${seed_index}]}"
source_root="${source_roots[${seed_index}]}"
export ZOOMTOKEN_FULL_RUN_ROOT="${RUN_ROOT}/seed_${seed}"
work_dir="${ZOOMTOKEN_FULL_RUN_ROOT}/${arm}"
source_checkpoint="${source_root}/${arm}/gpu2_id0/checkpoint/epoch_59.pth"
master_port="$((22000 + (SLURM_ARRAY_JOB_ID % 20000) + SLURM_ARRAY_TASK_ID))"
cfg_options=()
if [[ "${arm}" != "DO" ]]; then
  cfg_options=(--cfg-options "model.backbone.custom.georoute_random_seed=${seed}")
fi

[[ ! -e "${work_dir}" ]] || {
  echo "refusing to overwrite ${work_dir}" >&2
  exit 2
}
mkdir -p "${ZOOMTOKEN_FULL_RUN_ROOT}/logs" "${BASE}/logs"
cd "${ROOT}"

if [[ -s "${source_checkpoint}" ]]; then
  checkpoint="${source_checkpoint}"
  echo "MODE=EVALUATE_COMPLETED_FULL_TRAIN CHECKPOINT=${checkpoint}"
else
  echo "MODE=FULL_TRAIN_AND_EVALUATE"
  torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port="${master_port}" \
    tools/train.py "${config}" --seed "${seed}" --id 0 "${cfg_options[@]}" \
    2>&1 | tee "${ZOOMTOKEN_FULL_RUN_ROOT}/logs/${arm}_train.log"
  checkpoint="${work_dir}/gpu2_id0/checkpoint/epoch_59.pth"
  [[ -s "${checkpoint}" ]] || {
    echo "missing final checkpoint ${checkpoint}" >&2
    exit 2
  }
fi

torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port="${master_port}" \
  tools/test.py "${config}" --checkpoint "${checkpoint}" --seed "${seed}" --id 0 "${cfg_options[@]}" \
  2>&1 | tee "${ZOOMTOKEN_FULL_RUN_ROOT}/logs/${arm}_test.log"
