#!/usr/bin/env bash
#SBATCH --job-name=zt_full_official
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=72:00:00
#SBATCH --array=0-4

source /etc/profile
set -euo pipefail

BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
ROOT="${ZOOMTOKEN_SOURCE_ROOT:?set ZOOMTOKEN_SOURCE_ROOT}"
EXPECTED_COMMIT="${ZOOMTOKEN_EXPECTED_COMMIT:?set ZOOMTOKEN_EXPECTED_COMMIT}"
export ZOOMTOKEN_FULL_RUN_ROOT="${ZOOMTOKEN_FULL_RUN_ROOT:?set ZOOMTOKEN_FULL_RUN_ROOT}"
IMAGE="${ZOOMTOKEN_CONTAINER_IMAGE:-${BASE}/containers/zoomtoken_p1_ubi9_20260815.sif}"

if [[ -z "${APPTAINER_CONTAINER:-${SINGULARITY_CONTAINER:-}}" ]]; then
  module load apptainer/1.2.4
  exec apptainer exec --nv --bind "${BASE}:${BASE}" "${IMAGE}" \
    bash "${ROOT}/scripts/run_zoomtoken_full_official_matrix_n16r4.sbatch"
fi

module load cuda/11.8
module load miniforge3/24.11
source "${BASE}/conda_envs/opentad/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || {
  echo "source commit mismatch" >&2
  exit 2
}
[[ -z "$(git -C "${ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "source checkout is not clean" >&2
  exit 2
}

arms=(DO DN U R Q)
configs=(
  zoomtoken_full_official_do_seed3407_v001.py
  zoomtoken_full_official_dn_seed3407_v001.py
  zoomtoken_full_official_u_seed3407_v001.py
  zoomtoken_full_official_r_seed3407_v001.py
  zoomtoken_full_official_q_seed3407_v001.py
)
arm="${arms[${SLURM_ARRAY_TASK_ID}]}"
config="${ROOT}/configs/adatad/thumos/${configs[${SLURM_ARRAY_TASK_ID}]}"
work_dir="${ZOOMTOKEN_FULL_RUN_ROOT}/${arm}"
[[ ! -e "${work_dir}" ]] || {
  echo "refusing to overwrite ${work_dir}" >&2
  exit 2
}
mkdir -p "${ZOOMTOKEN_FULL_RUN_ROOT}/logs"
cd "${ROOT}"

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  tools/train.py "${config}" --seed 3407 --id 0 \
  2>&1 | tee "${ZOOMTOKEN_FULL_RUN_ROOT}/logs/${arm}_train.log"

checkpoint="${work_dir}/checkpoint/epoch_59.pth"
[[ -f "${checkpoint}" ]] || {
  echo "missing final checkpoint ${checkpoint}" >&2
  exit 2
}
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  tools/test.py "${config}" --checkpoint "${checkpoint}" --seed 3407 --id 0 \
  2>&1 | tee "${ZOOMTOKEN_FULL_RUN_ROOT}/logs/${arm}_test.log"
