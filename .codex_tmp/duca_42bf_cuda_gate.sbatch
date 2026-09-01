#!/usr/bin/env bash
#SBATCH --job-name=duca62eb_gate
#SBATCH --clusters=n16r4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_62eb52b_cuda_gate_20260722/gate-%j.out
#SBATCH --error=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_62eb52b_cuda_gate_20260722/gate-%j.err

set -eo pipefail
set +u
source /etc/profile
set -u
module load cuda/11.8
module load miniforge3/24.11
BASE=/data/run01/sczc063/yuzibo
source ${BASE}/conda_envs/opentad/bin/activate
REPO=${BASE}/projects/opentad_duca_boundary_62eb52b_20260722
ROOT=${BASE}/projects/c3_lowres_action_probe/duca_boundary_62eb52b_cuda_gate_20260722
SPLIT_ROOT=${BASE}/projects/c3_lowres_action_probe/duca_boundary_62eb52b_verify_20260722_2015/parallel_precheck/frontend_split
COMMIT=62eb52bbc9c68d07c68aef7e53517b4716872870
mkdir -p ${ROOT}
cd ${REPO}
[[ "$(git rev-parse HEAD)" == "${COMMIT}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
export BASE
export DUCA_CELLCF_TRAINING_PROFILE=official60
source scripts/duca_cellcf_canonical_env.sh
export DUCA_FRONTEND_TRAIN_BLOCK_LIST=${SPLIT_ROOT}/frontend_train_block_list.txt
export DUCA_FRONTEND_HOLDOUT_BLOCK_LIST=${SPLIT_ROOT}/frontend_holdout_block_list.txt
SPLIT_MANIFEST=${SPLIT_ROOT}/frontend_split_manifest.json
SPLIT_SHA256="$(sha256sum ${SPLIT_MANIFEST} | awk '{print $1}')"

python -m torch.distributed.run --standalone --nproc_per_node=1 \
  tools/bata/run_duca_frontend_p0_real_gate.py \
  --config configs/adatad/thumos/duca_boundary_burst_frontend_pretrain_fixed384.py \
  --variant-config configs/adatad/thumos/duca_boundary_burst_frontend_pretrain_fixed384.py \
  --variant-config configs/adatad/thumos/duca_boundary_burst_hard_detached_frontend_pretrain_fixed384.py \
  --variant-config configs/adatad/thumos/duca_boundary_burst_soft_detached_frontend_pretrain_fixed384.py \
  --variant-config configs/adatad/thumos/duca_boundary_burst_soft_adapted_frontend_pretrain_fixed384.py \
  --expected-commit ${COMMIT} \
  --checkpoint ${ADATAD_PRETRAIN_PATH} \
  --official-repos-root ${C3_OFFICIAL_ACTION_SEG_REPOS} \
  --split-manifest ${SPLIT_MANIFEST} \
  --expected-split-sha256 ${SPLIT_SHA256} \
  --output-json ${ROOT}/formal_full_model_gate.json

sha256sum ${ROOT}/formal_full_model_gate.json > ${ROOT}/formal_full_model_gate.json.sha256
echo "DUCA_62EB_CUDA_GATE_OK ${ROOT}/formal_full_model_gate.json"
