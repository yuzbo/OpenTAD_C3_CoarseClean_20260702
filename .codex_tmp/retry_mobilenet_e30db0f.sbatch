#!/usr/bin/env bash
#SBATCH --clusters=n16r4
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --gpus=3
#SBATCH --time=7-00:00:00

source /etc/profile
set -euo pipefail
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

REPO=/data/run01/sczc063/yuzibo/projects/opentad_duca_t1_e30db0f_20260723
EXPECTED=e30db0f3987128798da6bc8ff446065b818b1a7f
: "${RUN_ROOT:?set RUN_ROOT}"
cd "${REPO}"
test "$(git rev-parse HEAD)" = "${EXPECTED}"
test -z "$(git status --porcelain)"

export TORCH_HOME=/data/run01/sczc063/yuzibo/.cache/torch
variants=(
  trainfree_mobilenet_feature_change
  trainfree_mobilenet_semantic
  trainfree_mobilenet_fusion_r2q3
)
pids=()
for variant in "${variants[@]}"; do
  srun --exact --exclusive --ntasks=1 --gpus=1 --gpus-per-task=1 --cpus-per-task=4 \
    env BASE=/data/run01/sczc063/yuzibo \
      TORCH_HOME="${TORCH_HOME}" \
      DUCA_REPO_ROOT="${REPO}" \
      DUCA_EXPECTED_COMMIT="${EXPECTED}" \
      DUCA_ADATAD_PRETRAIN_PATH=/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth \
      DUCA_ADATAD_PRETRAIN_SHA256=4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251 \
      DUCA_INDEPENDENT_VARIANT="${variant}" \
      DUCA_INDEPENDENT_ARM_ROOT="${RUN_ROOT}/arms/${variant}" \
      bash scripts/run_duca_independent_official60_gpu1.sh \
      > "${RUN_ROOT}/logs/${variant}.out" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
test "${failed}" = 0
