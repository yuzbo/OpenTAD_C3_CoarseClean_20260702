#!/usr/bin/env bash
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo
COMMIT=e49ef69605e1f98a7217957483f93a8a64bfc348
DENSE_COMMIT=b3de5d8fac23d67cd9cae9c8c08bb60ba217f64f
SNAPSHOT=${BASE}/projects/opentad_duca_boundary_e49ef69_20260722
DENSE_SNAPSHOT=${BASE}/projects/opentad_dense_teacher_b3de5d8_cost_20260722
PRETRAIN=${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth
DENSE_RUN=${BASE}/projects/c3_lowres_action_probe/dense_adatad_teacher/c3_dense_adatad_teacher_full_b3de5d8_20260707_094642_+0800
DENSE_CONFIG=${DENSE_SNAPSHOT}/configs/adatad/thumos/c3_dense_adatad_teacher_full_train.py
DENSE_CHECKPOINT=${DENSE_RUN}/work_dir/gpu1_id0/checkpoint/epoch_59.pth
R03_ROOT=${BASE}/projects/c3_lowres_action_probe/duca_boundary_e49ef69_formal_20260722_155037_r0_r3
PREFIX=${R03_ROOT%_r0_r3}
R4_ROOT=${PREFIX}_r4
R5_ROOT=${PREFIX}_r5
DENSE_BINDING=${PREFIX}_dense_checkpoint_binding.json
JOURNAL=${R03_ROOT}/jobs.tsv
SEAL=${R03_ROOT}/jobs.complete.json

set +u
source /etc/profile
set -u
module load cuda/11.8 >/dev/null
module load miniforge3/24.11 >/dev/null
source ${BASE}/conda_envs/opentad/bin/activate

cd ${SNAPSHOT}
[[ "$(git rev-parse HEAD)" == "${COMMIT}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
[[ -f ${DENSE_BINDING} && -f ${DENSE_BINDING}.sha256 ]]
[[ "$(sha256sum ${DENSE_BINDING} | awk '{print $1}')" == "$(tr -d '[:space:]' < ${DENSE_BINDING}.sha256)" ]]

journal() {
  python -m tools.bata.duca_boundary_burst_submission_journal \
    --journal ${JOURNAL} \
    --seal ${SEAL} \
    --expected-commit ${COMMIT} \
    --target-cluster n16r4 "$@"
}

aggregate_dependency=afterok:1179798:1179799
aggregate_state=$(awk -F '\t' '$1 == "aggregate" {print $2}' ${JOURNAL})
if [[ ${aggregate_state} == PENDING ]]; then
  aggregate_raw=$(sbatch --parsable --clusters=n16r4 --gpus=1 \
    --dependency=${aggregate_dependency} \
    ${R03_ROOT}/submission/aggregate.sbatch)
  R3_AGG_JOB=${aggregate_raw%%;*}
  [[ ${R3_AGG_JOB} =~ ^[1-9][0-9]*$ ]]
  journal record --role aggregate --job-id ${R3_AGG_JOB} \
    --dependency ${aggregate_dependency}
  journal seal
else
  R3_AGG_JOB=${aggregate_state}
  [[ ${R3_AGG_JOB} =~ ^[1-9][0-9]*$ ]]
fi
[[ "$(journal inspect)" == COMPLETE ]]

[[ ! -e ${R4_ROOT} ]]
mkdir -p ${R4_ROOT}/slurm
R4_COMMAND="export BASE=${BASE} DUCA_REPO_ROOT=${SNAPSHOT} DUCA_EXPECTED_COMMIT=${COMMIT} DUCA_BOUNDARY_BURST_TERMINAL_SUITE=${R03_ROOT}/final_suite_results.json DUCA_BOUNDARY_BURST_R4_ROOT=${R4_ROOT}; export DUCA_BOUNDARY_BURST_TERMINAL_SUITE_SHA256=\$(sha256sum \${DUCA_BOUNDARY_BURST_TERMINAL_SUITE} | awk '{print \$1}'); bash ${SNAPSHOT}/scripts/run_duca_boundary_burst_r4_gpu1.sbatch"
R4_RAW=$(sbatch --parsable --clusters=n16r4 \
  --dependency=afterok:${R3_AGG_JOB} \
  --job-name=burst_r4_${COMMIT:0:7} --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --gpus=1 --time=7-00:00:00 \
  --output=${R4_ROOT}/slurm/burst_r4-%j.out \
  --error=${R4_ROOT}/slurm/burst_r4-%j.err --wrap="${R4_COMMAND}")
R4_JOB=${R4_RAW%%;*}
[[ ${R4_JOB} =~ ^[1-9][0-9]*$ ]]

export DUCA_REPO_ROOT=${SNAPSHOT}
export DUCA_EXPECTED_COMMIT=${COMMIT}
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
{
  printf 'field\tvalue\n'
  printf 'exact_commit\t%s\n' ${COMMIT}
  printf 'snapshot\t%s\n' ${SNAPSHOT}
  printf 'dense_commit\t%s\n' ${DENSE_COMMIT}
  printf 'dense_snapshot\t%s\n' ${DENSE_SNAPSHOT}
  printf 'dense_binding\t%s\n' ${DENSE_BINDING}
  printf 'r0_r3_root\t%s\n' ${R03_ROOT}
  printf 'r3_aggregate_job\t%s\n' ${R3_AGG_JOB}
  printf 'r4_root\t%s\n' ${R4_ROOT}
  printf 'r4_job\t%s\n' ${R4_JOB}
  printf 'r4_dependency\tafterok:%s\n' ${R3_AGG_JOB}
  printf 'r5_root\t%s\n' ${R5_ROOT}
  printf 'r5_dependency\tafterok:%s\n' ${R4_JOB}
  printf 'r0_r3_jobs\t%s\n' ${R03_ROOT}/jobs.tsv
  printf 'r5_jobs\t%s\n' ${R5_ROOT}/jobs.tsv
} > ${RECEIPT}
sha256sum ${RECEIPT} | awk '{print $1}' > ${RECEIPT}.sha256

echo DEPLOYMENT_OK
echo PREFIX=${PREFIX}
echo R03_ROOT=${R03_ROOT}
echo R3_AGG_JOB=${R3_AGG_JOB}
echo R4_ROOT=${R4_ROOT}
echo R4_JOB=${R4_JOB}
echo R5_ROOT=${R5_ROOT}
echo RECEIPT=${RECEIPT}
echo R03_JOBS
cat ${R03_ROOT}/jobs.tsv
echo R5_JOBS
cat ${R5_ROOT}/jobs.tsv
