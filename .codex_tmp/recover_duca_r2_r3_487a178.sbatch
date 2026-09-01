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

REPO=/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_da4adf4_20260723
EXPECTED_COMMIT=487a1784554b8c07cbaf8e3948c5aea785a2d8e1
cd "${REPO}"

: "${RECOVERY_VARIANT:?set RECOVERY_VARIANT}"
: "${RECOVERY_P0:?set RECOVERY_P0}"
: "${RECOVERY_P0_SHA:?set RECOVERY_P0_SHA}"
: "${RECOVERY_ARM_ROOT:?set RECOVERY_ARM_ROOT}"

export BASE=/data/run01/sczc063/yuzibo
export DUCA_REPO_ROOT="${REPO}"
export DUCA_EXPECTED_COMMIT="${EXPECTED_COMMIT}"
export DUCA_INDEPENDENT_VARIANT="${RECOVERY_VARIANT}"
export DUCA_INDEPENDENT_ARM_ROOT="${RECOVERY_ARM_ROOT}"
export DUCA_REUSE_P0_CHECKPOINT="${RECOVERY_P0}"
export DUCA_REUSE_P0_CHECKPOINT_SHA256="${RECOVERY_P0_SHA}"
export DUCA_ADATAD_PRETRAIN_PATH=/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth
export DUCA_ADATAD_PRETRAIN_SHA256=4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251
export DUCA_FRONTEND_SPLIT_MANIFEST=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_9f97f2c_formal_20260722_2343/frontend_split/frontend_split_manifest.json
export DUCA_FRONTEND_SPLIT_MANIFEST_SHA256=89c4f103934896b38b76011dc2753e0065fced1654ac8edcc4a896ed96366c40
export DUCA_SPLIT_ANNOTATION_PATH=/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json
export DUCA_SPLIT_ANNOTATION_SHA256=ee526d55aa4315a8adc68c501d0331f96a56ce16fa960f1d2ea182b9381ab9ad
export DUCA_FRONTEND_TRAIN_BLOCK_LIST=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_9f97f2c_formal_20260722_2343/frontend_split/frontend_train_block_list.txt
export DUCA_FRONTEND_TRAIN_BLOCK_LIST_SHA256=a5c44379bf34d8506a93b28985d4322691d46f7223f44a82dfbdbd378d877194
export DUCA_FRONTEND_HOLDOUT_BLOCK_LIST=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_9f97f2c_formal_20260722_2343/frontend_split/frontend_holdout_block_list.txt
export DUCA_FRONTEND_HOLDOUT_BLOCK_LIST_SHA256=31007c0ef35aa818d56c6e5586ee9819c14277cf0d24fb817d11829ebfbe4c17
export DUCA_PREREGISTERED_PROJECTED_FAMILY=R2Q3_privileged_boundary_burst

bash scripts/run_duca_independent_official60_gpu1.sh
