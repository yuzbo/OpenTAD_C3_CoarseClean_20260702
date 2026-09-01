#!/usr/bin/env bash
set -eo pipefail

BASE=/data/run01/sczc063/yuzibo
COMMIT=cd68d89dcc0854baa3c0107607086e801509b552
BRANCH=codex/duca-boundary-burst-20260722
SNAPSHOT=${BASE}/projects/opentad_duca_boundary_cd68d89_20260722
R0_CHECKPOINT=${BASE}/projects/c3_lowres_action_probe/duca_cellcf_1642f26_formal_seed0_20260717_0200/work_dirs/transition_beta0/gpu1_id0/checkpoint/epoch_131.pth
PRETRAIN=${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth
DENSE_COMMIT=b3de5d8fac23d67cd9cae9c8c08bb60ba217f64f
DENSE_SNAPSHOT=${BASE}/projects/opentad_dense_teacher_b3de5d8_cost_20260722
DENSE_RUN=${BASE}/projects/c3_lowres_action_probe/dense_adatad_teacher/c3_dense_adatad_teacher_full_b3de5d8_20260707_094642_+0800
DENSE_CONFIG=${DENSE_SNAPSHOT}/configs/adatad/thumos/c3_dense_adatad_teacher_full_train.py
DENSE_CHECKPOINT=${DENSE_RUN}/work_dir/gpu1_id0/checkpoint/epoch_59.pth
DENSE_TRAINING=${DENSE_RUN}/work_dir/gpu1_id0/log.json
DENSE_EVALUATION=${DENSE_RUN}/train.out
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT=${BASE}/projects/c3_lowres_action_probe/duca_boundary_cd68d89_parallel_${STAMP}
DENSE_BINDING=${RUN_ROOT}_dense_checkpoint_binding.json

set +u
source /etc/profile
set -u
module load cuda/11.8 >/dev/null
module load miniforge3/24.11 >/dev/null
source ${BASE}/conda_envs/opentad/bin/activate

if [[ ! -d ${SNAPSHOT}/.git ]]; then
  git clone --branch ${BRANCH} --single-branch \
    https://ghfast.top/https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git \
    ${SNAPSHOT}
fi
cd ${SNAPSHOT}
git fetch origin ${BRANCH}
git checkout --detach ${COMMIT}
[[ "$(git rev-parse HEAD)" == "${COMMIT}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
[[ ! -e ${RUN_ROOT} ]]
[[ -f ${R0_CHECKPOINT} && -f ${PRETRAIN} ]]
[[ -f ${DENSE_CONFIG} && -f ${DENSE_CHECKPOINT} && -f ${DENSE_TRAINING} && -f ${DENSE_EVALUATION} ]]

python -c "from mmengine.config import Config; from tools.bata.profile_duca_full_stack_cost import _payload_fingerprint; from tools.bata.duca_trained_checkpoint_binding import build_trained_checkpoint_binding,write_trained_checkpoint_binding; cfg=Config.fromfile('${DENSE_CONFIG}'); payload=build_trained_checkpoint_binding(role='dense_adatad_baseline',git_commit='${DENSE_COMMIT}',config_path='${DENSE_CONFIG}',resolved_config_sha256=_payload_fingerprint(cfg),checkpoint_path='${DENSE_CHECKPOINT}',checkpoint_epoch=59,checkpoint_state_key='state_dict_ema',training_evidence_path='${DENSE_TRAINING}',evaluation_evidence_path='${DENSE_EVALUATION}'); write_trained_checkpoint_binding('${DENSE_BINDING}',payload)"
sha256sum ${DENSE_BINDING} | awk '{print $1}' > ${DENSE_BINDING}.sha256

export BASE
export RUN_ROOT
export DUCA_EXPECTED_COMMIT=${COMMIT}
export DUCA_R0_CHECKPOINT=${R0_CHECKPOINT}
export DUCA_R0_CHECKPOINT_EPOCH=131
export ADATAD_PRETRAIN_PATH=${PRETRAIN}
export DUCA_TARGET_CLUSTER=n16r4
export DUCA_R5_DENSE_CONFIG=${DENSE_CONFIG}
export DUCA_R5_DENSE_CHECKPOINT=${DENSE_CHECKPOINT}
export DUCA_R5_DENSE_CHECKPOINT_EVIDENCE=${DENSE_BINDING}
export DUCA_R5_DENSE_TRAINED_COMMIT=${DENSE_COMMIT}
bash scripts/submit_duca_r0_r5_parallel_bundles.sh

echo DEPLOYMENT_OK
echo COMMIT=${COMMIT}
echo SNAPSHOT=${SNAPSHOT}
echo RUN_ROOT=${RUN_ROOT}
echo DENSE_BINDING=${DENSE_BINDING}
cat ${RUN_ROOT}/jobs.tsv
