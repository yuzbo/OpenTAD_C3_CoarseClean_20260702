#!/usr/bin/env bash
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo
COMMIT=e49ef69605e1f98a7217957483f93a8a64bfc348
DENSE_COMMIT=b3de5d8fac23d67cd9cae9c8c08bb60ba217f64f
SNAPSHOT=${BASE}/projects/opentad_duca_boundary_e49ef69_20260722
DENSE_SNAPSHOT=${BASE}/projects/opentad_dense_teacher_b3de5d8_cost_20260722
PRETRAIN=${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth
R0_CHECKPOINT=${BASE}/projects/c3_lowres_action_probe/duca_cellcf_1642f26_formal_seed0_20260717_0200/work_dirs/transition_beta0/gpu1_id0/checkpoint/epoch_131.pth
DENSE_RUN=${BASE}/projects/c3_lowres_action_probe/dense_adatad_teacher/c3_dense_adatad_teacher_full_b3de5d8_20260707_094642_+0800
DENSE_CONFIG=${DENSE_SNAPSHOT}/configs/adatad/thumos/c3_dense_adatad_teacher_full_train.py
DENSE_CHECKPOINT=${DENSE_RUN}/work_dir/gpu1_id0/checkpoint/epoch_59.pth
DENSE_TRAINING=${DENSE_RUN}/work_dir/gpu1_id0/log.json
DENSE_EVALUATION=${DENSE_RUN}/train.out
STAMP=$(date +%Y%m%d_%H%M%S)
PREFIX=${BASE}/projects/c3_lowres_action_probe/duca_boundary_e49ef69_formal_${STAMP}
R03_ROOT=${PREFIX}_r0_r3
R4_ROOT=${PREFIX}_r4
R5_ROOT=${PREFIX}_r5
DENSE_BINDING=${PREFIX}_dense_checkpoint_binding.json

set +u
source /etc/profile
set -u
module load cuda/11.8 >/dev/null
module load miniforge3/24.11 >/dev/null
source ${BASE}/conda_envs/opentad/bin/activate

cd ${SNAPSHOT}
[[ "$(git rev-parse HEAD)" == "${COMMIT}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
[[ "$(git -C ${DENSE_SNAPSHOT} rev-parse HEAD)" == "${DENSE_COMMIT}" ]]
[[ -z "$(git -C ${DENSE_SNAPSHOT} status --porcelain --untracked-files=normal)" ]]
[[ -f ${PRETRAIN} && -f ${R0_CHECKPOINT} ]]
[[ -f ${DENSE_CONFIG} && -f ${DENSE_CHECKPOINT} && -f ${DENSE_TRAINING} && -f ${DENSE_EVALUATION} ]]

python -c "from mmengine.config import Config; from tools.bata.profile_duca_full_stack_cost import _payload_fingerprint; from tools.bata.duca_trained_checkpoint_binding import build_trained_checkpoint_binding,write_trained_checkpoint_binding; cfg=Config.fromfile('${DENSE_CONFIG}'); payload=build_trained_checkpoint_binding(role='dense_adatad_baseline',git_commit='${DENSE_COMMIT}',config_path='${DENSE_CONFIG}',resolved_config_sha256=_payload_fingerprint(cfg),checkpoint_path='${DENSE_CHECKPOINT}',checkpoint_epoch=59,checkpoint_state_key='state_dict_ema',training_evidence_path='${DENSE_TRAINING}',evaluation_evidence_path='${DENSE_EVALUATION}'); write_trained_checkpoint_binding('${DENSE_BINDING}',payload)"
sha256sum ${DENSE_BINDING} | awk '{print $1}' > ${DENSE_BINDING}.sha256

export BASE
export RUN_ROOT=${R03_ROOT}
export DUCA_EXPECTED_COMMIT=${COMMIT}
export DUCA_R0_CHECKPOINT=${R0_CHECKPOINT}
export DUCA_R0_CHECKPOINT_EPOCH=131
export DUCA_TARGET_CLUSTER=n16r4
bash scripts/submit_duca_boundary_burst_official60_suite.sh

R3_AGG_JOB=$(awk -F '\t' '$1 == "aggregate" {print $2}' ${R03_ROOT}/jobs.tsv)
[[ ${R3_AGG_JOB} =~ ^[1-9][0-9]*$ ]]
mkdir -p ${R4_ROOT}/slurm
R4_COMMAND="export BASE=${BASE} DUCA_REPO_ROOT=${SNAPSHOT} DUCA_EXPECTED_COMMIT=${COMMIT} DUCA_BOUNDARY_BURST_TERMINAL_SUITE=${R03_ROOT}/final_suite_results.json DUCA_BOUNDARY_BURST_R4_ROOT=${R4_ROOT}; export DUCA_BOUNDARY_BURST_TERMINAL_SUITE_SHA256=\$(sha256sum \${DUCA_BOUNDARY_BURST_TERMINAL_SUITE} | awk '{print \$1}'); bash ${SNAPSHOT}/scripts/run_duca_boundary_burst_r4_gpu1.sbatch"
R4_RAW=$(sbatch --parsable --clusters=n16r4 --dependency=afterok:${R3_AGG_JOB} \
  --job-name=burst_r4_${COMMIT:0:7} --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gpus=1 --time=7-00:00:00 \
  --output=${R4_ROOT}/slurm/burst_r4-%j.out \
  --error=${R4_ROOT}/slurm/burst_r4-%j.err --wrap="${R4_COMMAND}")
R4_JOB=${R4_RAW%%;*}
[[ ${R4_JOB} =~ ^[1-9][0-9]*$ ]]

export DUCA_REPO_ROOT=${SNAPSHOT}
export ADATAD_PRETRAIN_PATH=${PRETRAIN}
export ADATAD_PRETRAIN_SHA256=$(sha256sum ${PRETRAIN} | awk '{print $1}')
export R5_OUTPUT_DIR=${R5_ROOT}
export R5_LEARNED_CONFIG=${SNAPSHOT}/configs/adatad/thumos/duca_boundary_burst_g1_protected_fixed384_official60.py
export R5_FRONTEND_DECISION=${R03_ROOT}/frontend_decision.json
export R5_FRONTEND_DECISION_SHA256_FILE=${R03_ROOT}/frontend_decision.sha256
export R5_ALIGNMENT_JSON=${R4_ROOT}/alignment/alignment.json
export R5_ALIGNMENT_SHA256_FILE=${R4_ROOT}/alignment/alignment.json.sha256
export R5_DENSE_CONFIG=${DENSE_CONFIG}
export R5_DENSE_CHECKPOINT=${DENSE_CHECKPOINT}
export R5_DENSE_CHECKPOINT_EVIDENCE=${DENSE_BINDING}
export R5_DENSE_TRAINED_COMMIT=${DENSE_COMMIT}
export R5_UPSTREAM_DEPENDENCY=afterok:${R4_JOB}
export R5_SUBMIT=1
export TARGET_CLUSTER=n16r4
bash scripts/launch_duca_r5_paper_matrix.sh

RECEIPT=${PREFIX}_deployment.tsv
printf 'field\tvalue\n' > ${RECEIPT}
printf 'exact_commit\t%s\n' ${COMMIT} >> ${RECEIPT}
printf 'snapshot\t%s\n' ${SNAPSHOT} >> ${RECEIPT}
printf 'dense_commit\t%s\n' ${DENSE_COMMIT} >> ${RECEIPT}
printf 'dense_snapshot\t%s\n' ${DENSE_SNAPSHOT} >> ${RECEIPT}
printf 'dense_binding\t%s\n' ${DENSE_BINDING} >> ${RECEIPT}
printf 'r0_r3_root\t%s\n' ${R03_ROOT} >> ${RECEIPT}
printf 'r3_aggregate_job\t%s\n' ${R3_AGG_JOB} >> ${RECEIPT}
printf 'r4_root\t%s\n' ${R4_ROOT} >> ${RECEIPT}
printf 'r4_job\t%s\n' ${R4_JOB} >> ${RECEIPT}
printf 'r4_dependency\tafterok:%s\n' ${R3_AGG_JOB} >> ${RECEIPT}
printf 'r5_root\t%s\n' ${R5_ROOT} >> ${RECEIPT}
printf 'r5_dependency\tafterok:%s\n' ${R4_JOB} >> ${RECEIPT}
printf 'r0_r3_jobs\t%s\n' ${R03_ROOT}/jobs.tsv >> ${RECEIPT}
printf 'r5_jobs\t%s\n' ${R5_ROOT}/jobs.tsv >> ${RECEIPT}
sha256sum ${RECEIPT} | awk '{print $1}' > ${RECEIPT}.sha256

echo DEPLOYMENT_OK
echo PREFIX=${PREFIX}
echo R03_ROOT=${R03_ROOT}
echo R3_AGG_JOB=${R3_AGG_JOB}
echo R4_ROOT=${R4_ROOT}
echo R4_JOB=${R4_JOB}
echo R5_ROOT=${R5_ROOT}
echo RECEIPT=${RECEIPT}
cat ${R03_ROOT}/jobs.tsv
cat ${R5_ROOT}/jobs.tsv
