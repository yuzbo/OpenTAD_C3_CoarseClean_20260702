#!/usr/bin/env bash
#SBATCH --clusters=n16r4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00

source /etc/profile
set -euo pipefail
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

REPO=/data/run01/sczc063/yuzibo/projects/opentad_duca_t1_919aa55_20260723
EXPECTED=919aa555d1aa36191ee318477409dfbfdfb0e807
P0_SHA=acb6e30673e811f34ce84d710442581bec8a74ca68e9187eb71e005e01536c9b
: "${RUN_ROOT:?set RUN_ROOT}"
: "${VARIANT:?set VARIANT}"
: "${P0_CHECKPOINT:?set P0_CHECKPOINT}"
cd "${REPO}"
test "$(git rev-parse HEAD)" = "${EXPECTED}"
test -z "$(git status --porcelain)"

env BASE=/data/run01/sczc063/yuzibo \
  DUCA_REPO_ROOT="${REPO}" \
  DUCA_EXPECTED_COMMIT="${EXPECTED}" \
  DUCA_ADATAD_PRETRAIN_PATH=/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth \
  DUCA_ADATAD_PRETRAIN_SHA256=4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251 \
  DUCA_INDEPENDENT_VARIANT="${VARIANT}" \
  DUCA_INDEPENDENT_ARM_ROOT="${RUN_ROOT}/arms/${VARIANT}" \
  DUCA_REUSE_P0_CHECKPOINT="${P0_CHECKPOINT}" \
  DUCA_REUSE_P0_CHECKPOINT_SHA256="${P0_SHA}" \
  bash scripts/run_duca_independent_official60_gpu1.sh
