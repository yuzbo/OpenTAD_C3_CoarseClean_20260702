#!/usr/bin/env bash
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo
COMMIT=5426dff5f3fae03c74b3baff9d15d38527a47f11
SNAPSHOT=${BASE}/projects/opentad_duca_boundary_5426dff_20260722
PRETRAIN=${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth
R0_CHECKPOINT=${BASE}/projects/c3_lowres_action_probe/duca_cellcf_1642f26_formal_seed0_20260717_0200/work_dirs/transition_beta0/gpu1_id0/checkpoint/epoch_131.pth
STAMP=$(date +%Y%m%d_%H%M%S)
PREFIX=${BASE}/projects/c3_lowres_action_probe/duca_boundary_5426dff_formal_${STAMP}
R03_ROOT=${PREFIX}_r0_r3
R4_ROOT=${PREFIX}_r4
R5_ROOT=${PREFIX}_r5

set +u
source /etc/profile
set -u
module load cuda/11.8 >/dev/null
module load miniforge3/24.11 >/dev/null
source ${BASE}/conda_envs/opentad/bin/activate

cd ${SNAPSHOT}
[[ "$(git rev-parse HEAD)" == "${COMMIT}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
[[ -f ${PRETRAIN} ]]
[[ -f ${R0_CHECKPOINT} ]]

export BASE
export RUN_ROOT=${R03_ROOT}
export DUCA_EXPECTED_COMMIT=${COMMIT}
export DUCA_R0_CHECKPOINT=${R0_CHECKPOINT}
export DUCA_R0_CHECKPOINT_EPOCH=131
export DUCA_TARGET_CLUSTER=n16r4
bash scripts/submit_duca_boundary_burst_official60_suite.sh

R3_AGG_JOB=$(awk -F '\t' '$1 == "aggregate" {print $2}' ${R03_ROOT}/jobs.tsv)
[[ ${R3_AGG_JOB} =~ ^[1-9][0-9]*$ ]]

R4_WRAPPER=${R03_ROOT}/submission/r4_after_r3.sbatch
cat >${R4_WRAPPER} <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=burst_r4_${COMMIT:0:7}
#SBATCH --clusters=n16r4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00
#SBATCH --output=${R03_ROOT}/logs/burst_r4_${COMMIT:0:7}-%j.out
#SBATCH --error=${R03_ROOT}/logs/burst_r4_${COMMIT:0:7}-%j.err
set -euo pipefail
source /etc/profile
module load cuda/11.8
module load miniforge3/24.11
source ${BASE}/conda_envs/opentad/bin/activate
export BASE=${BASE}
export DUCA_REPO_ROOT=${SNAPSHOT}
export DUCA_EXPECTED_COMMIT=${COMMIT}
export DUCA_BOUNDARY_BURST_TERMINAL_SUITE=${R03_ROOT}/final_suite_results.json
export DUCA_BOUNDARY_BURST_TERMINAL_SUITE_SHA256=\$(sha256sum \${DUCA_BOUNDARY_BURST_TERMINAL_SUITE} | awk '{print \$1}')
export DUCA_BOUNDARY_BURST_R4_ROOT=${R4_ROOT}
bash ${SNAPSHOT}/scripts/run_duca_boundary_burst_r4_gpu1.sbatch
EOF
bash -n ${R4_WRAPPER}
R4_RAW=$(sbatch --parsable --clusters=n16r4 --dependency=afterok:${R3_AGG_JOB} ${R4_WRAPPER})
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
export R5_UPSTREAM_DEPENDENCY=afterok:${R4_JOB}
export R5_SUBMIT=1
export TARGET_CLUSTER=n16r4
bash scripts/launch_duca_r5_paper_matrix.sh

RECEIPT=${PREFIX}_deployment.tsv
{
  printf 'field\tvalue\n'
  printf 'exact_commit\t%s\n' ${COMMIT}
  printf 'snapshot\t%s\n' ${SNAPSHOT}
  printf 'r0_r3_root\t%s\n' ${R03_ROOT}
  printf 'r3_aggregate_job\t%s\n' ${R3_AGG_JOB}
  printf 'r4_root\t%s\n' ${R4_ROOT}
  printf 'r4_job\t%s\n' ${R4_JOB}
  printf 'r4_dependency\tafterok:%s\n' ${R3_AGG_JOB}
  printf 'r5_root\t%s\n' ${R5_ROOT}
  printf 'r5_dependency\tafterok:%s\n' ${R4_JOB}
  printf 'r0_r3_jobs\t%s\n' ${R03_ROOT}/jobs.tsv
  printf 'r5_jobs\t%s\n' ${R5_ROOT}/jobs.tsv
} >${RECEIPT}
sha256sum ${RECEIPT} | awk '{print $1}' >${RECEIPT}.sha256

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
