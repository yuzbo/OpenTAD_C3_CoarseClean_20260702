#!/usr/bin/env bash
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo
COMMIT=e49ef69605e1f98a7217957483f93a8a64bfc348
SNAPSHOT=${BASE}/projects/opentad_duca_boundary_e49ef69_20260722
PRETRAIN=${BASE}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth
PREFIX=${BASE}/projects/c3_lowres_action_probe/duca_boundary_e49ef69_formal_20260722_155037
R03=${PREFIX}_r0_r3
R4=${PREFIX}_r4
R5=${PREFIX}_r5
R4_JOB=1179826
GATE_JOB=1179827
DENSE_SNAPSHOT=${BASE}/projects/opentad_dense_teacher_b3de5d8_cost_20260722
DENSE_RUN=${BASE}/projects/c3_lowres_action_probe/dense_adatad_teacher/c3_dense_adatad_teacher_full_b3de5d8_20260707_094642_+0800
DENSE_CONFIG=${DENSE_SNAPSHOT}/configs/adatad/thumos/c3_dense_adatad_teacher_full_train.py
DENSE_CHECKPOINT=${DENSE_RUN}/work_dir/gpu1_id0/checkpoint/epoch_59.pth
DENSE_BINDING=${PREFIX}_dense_checkpoint_binding.json

cd ${SNAPSHOT}
[[ "$(git rev-parse HEAD)" == "${COMMIT}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
[[ -f ${R5}/jobs.tsv && ! -e ${R5}/jobs.tsv.sha256 ]]
[[ $(awk 'END {print NR-1}' ${R5}/jobs.tsv) -eq 7 ]]

PARTIAL=${R5}/jobs.individual_partial.tsv
cp ${R5}/jobs.tsv ${PARTIAL}
sha256sum ${PARTIAL} | awk '{print $1}' > ${PARTIAL}.sha256
mapfile -t cancelled_ids < <(awk -F '\t' 'NR > 2 {print $2}' ${PARTIAL})
[[ ${#cancelled_ids[@]} -eq 6 ]]
scancel "${cancelled_ids[@]}"
sleep 5

PRETRAIN_SHA=$(sha256sum ${PRETRAIN} | awk '{print $1}')
COMMON_EXPORTS="export DUCA_REPO_ROOT=${SNAPSHOT} DUCA_EXPECTED_COMMIT=${COMMIT} ADATAD_PRETRAIN_PATH=${PRETRAIN} ADATAD_PRETRAIN_SHA256=${PRETRAIN_SHA} R5_FRONTEND_DECISION=${R03}/frontend_decision.json R5_FRONTEND_DECISION_SHA256_FILE=${R03}/frontend_decision.sha256 R5_ALIGNMENT_JSON=${R4}/alignment/alignment.json R5_ALIGNMENT_SHA256_FILE=${R4}/alignment/alignment.json.sha256 R5_DENSE_CONFIG=${DENSE_CONFIG} R5_DENSE_CHECKPOINT=${DENSE_CHECKPOINT} R5_DENSE_CHECKPOINT_EVIDENCE=${DENSE_BINDING} R5_DENSE_TRAINED_COMMIT=b3de5d8fac23d67cd9cae9c8c08bb60ba217f64f"

MEMBERS=${R5}/bundle_members.tsv
printf 'bundle\tmember\tscript\n' > ${MEMBERS}
mapfile -t cell_rows < <(tail -n +2 ${R5}/cells.tsv)
[[ ${#cell_rows[@]} -eq 24 ]]

declare -a bundle_scripts bundle_jobs
for bundle in 0 1 2 3; do
  script=${R5}/jobs/train_bundle_${bundle}.sbatch
  bundle_scripts+=("${script}")
  {
    printf '#!/usr/bin/env bash\n'
    printf '#SBATCH --job-name=r5-train-bundle-%s\n' ${bundle}
    printf '#SBATCH --clusters=n16r4\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n'
    printf '#SBATCH --cpus-per-task=8\n#SBATCH --gpus=1\n#SBATCH --time=7-00:00:00\n'
    printf '#SBATCH --output=%s/logs/r5-train-bundle-%s-%%j.out\n' ${R5} ${bundle}
    printf '#SBATCH --error=%s/logs/r5-train-bundle-%s-%%j.err\n\n' ${R5} ${bundle}
    printf 'set -euo pipefail\n%s\n' "${COMMON_EXPORTS}"
    printf 'echo BUNDLE_START=%s JOB=${SLURM_JOB_ID}\n' ${bundle}
    for ((index=bundle; index<24; index+=4)); do
      IFS=$'\t' read -r role backend arm budget seed config config_sha job job_sha <<< "${cell_rows[index]}"
      [[ -f ${job} && "$(sha256sum ${job} | awk '{print $1}')" == "${job_sha}" ]]
      printf 'echo CELL_START=%s\n' ${role}
      printf 'bash %q\n' ${job}
      printf 'echo CELL_DONE=%s\n' ${role}
      printf '%s\t%s\t%s\n' train_bundle_${bundle} ${role} ${job} >> ${MEMBERS}
    done
    printf 'echo BUNDLE_DONE=%s\n' ${bundle}
  } > ${script}
  chmod 750 ${script}
  sha256sum ${script} | awk '{print $1}' > ${script}.sha256
done

POST=${R5}/jobs/postprocess_bundle.sbatch
{
  printf '#!/usr/bin/env bash\n'
  printf '#SBATCH --job-name=r5-postprocess\n'
  printf '#SBATCH --clusters=n16r4\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n'
  printf '#SBATCH --cpus-per-task=8\n#SBATCH --gpus=1\n#SBATCH --time=7-00:00:00\n'
  printf '#SBATCH --output=%s/logs/r5-postprocess-%%j.out\n' ${R5}
  printf '#SBATCH --error=%s/logs/r5-postprocess-%%j.err\n\n' ${R5}
  printf 'set -euo pipefail\n%s\n' "${COMMON_EXPORTS}"
  printf 'echo POSTPROCESS_START JOB=${SLURM_JOB_ID}\n'
  while IFS=$'\t' read -r cost kind source job summary; do
    [[ ${cost} == id ]] && continue
    [[ -f ${job} ]]
    printf 'echo COST_START=%s\n' ${cost}
    printf 'bash %q\n' ${job}
    printf 'echo COST_DONE=%s\n' ${cost}
    printf '%s\t%s\t%s\n' postprocess_bundle ${cost} ${job} >> ${MEMBERS}
  done < ${R5}/costs.tsv
  printf 'echo AGGREGATE_START\n'
  printf 'bash %q\n' ${R5}/jobs/aggregate.sbatch
  printf 'echo POSTPROCESS_DONE\n'
  printf '%s\t%s\t%s\n' postprocess_bundle aggregate ${R5}/jobs/aggregate.sbatch >> ${MEMBERS}
} > ${POST}
chmod 750 ${POST}
sha256sum ${POST} | awk '{print $1}' > ${POST}.sha256
sha256sum ${MEMBERS} | awk '{print $1}' > ${MEMBERS}.sha256

JOBS_TMP=${R5}/.jobs.tsv.bundle.tmp
printf 'role\tjob_id\tdependency\tsbatch\n' > ${JOBS_TMP}
printf 'temporalmaxer_one_step\t%s\tafterok:%s\t%s\n' \
  ${GATE_JOB} ${R4_JOB} ${R5}/jobs/temporalmaxer_one_step.sbatch >> ${JOBS_TMP}

for bundle in 0 1 2 3; do
  raw=$(sbatch --parsable --clusters=n16r4 --dependency=afterok:${GATE_JOB} \
    ${R5}/jobs/train_bundle_${bundle}.sbatch)
  job=${raw%%;*}
  [[ ${job} =~ ^[1-9][0-9]*$ ]]
  bundle_jobs+=("${job}")
  printf 'train_bundle_%s\t%s\tafterok:%s\t%s\n' \
    ${bundle} ${job} ${GATE_JOB} ${R5}/jobs/train_bundle_${bundle}.sbatch >> ${JOBS_TMP}
done

post_dependency="afterok:$(IFS=:; echo "${bundle_jobs[*]}")"
raw=$(sbatch --parsable --clusters=n16r4 --dependency=${post_dependency} ${POST})
POST_JOB=${raw%%;*}
[[ ${POST_JOB} =~ ^[1-9][0-9]*$ ]]
printf 'postprocess_bundle\t%s\t%s\t%s\n' \
  ${POST_JOB} ${post_dependency} ${POST} >> ${JOBS_TMP}
mv ${JOBS_TMP} ${R5}/jobs.tsv
sha256sum ${R5}/jobs.tsv | awk '{print $1}' > ${R5}/jobs.tsv.sha256

python - ${R5} ${COMMIT} ${GATE_JOB} ${POST_JOB} ${MEMBERS} "${cancelled_ids[*]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
members = Path(sys.argv[5]).resolve()
payload = {
    "schema": "duca_r5_site_bundle_submission_v1",
    "task": "offline_temporal_action_detection",
    "git_commit": sys.argv[2],
    "reason": "n16r4 AssocMaxSubmitJobLimit",
    "experiment_content_changed": False,
    "mechanism_gate_job": sys.argv[3],
    "postprocess_job": sys.argv[4],
    "cancelled_duplicate_individual_job_ids": sys.argv[6].split(),
    "train_bundle_count": 4,
    "cells_executed": 24,
    "cost_profiles_executed": 9,
    "final_aggregate_executed": True,
    "bundle_members_path": str(members),
    "bundle_members_sha256": hashlib.sha256(members.read_bytes()).hexdigest(),
    "jobs_path": str(root / "jobs.tsv"),
    "jobs_sha256": hashlib.sha256((root / "jobs.tsv").read_bytes()).hexdigest(),
}
output = root / "site_bundle_submission.json"
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(root / "site_bundle_submission.json.sha256").write_text(
    hashlib.sha256(output.read_bytes()).hexdigest() + "\n", encoding="utf-8"
)
PY

RECEIPT=${PREFIX}_deployment.tsv
{
  printf 'field\tvalue\n'
  printf 'exact_commit\t%s\n' ${COMMIT}
  printf 'snapshot\t%s\n' ${SNAPSHOT}
  printf 'r0_r3_root\t%s\n' ${R03}
  printf 'r3_aggregate_job\t1179825\n'
  printf 'r4_root\t%s\n' ${R4}
  printf 'r4_job\t%s\n' ${R4_JOB}
  printf 'r5_root\t%s\n' ${R5}
  printf 'r5_gate_job\t%s\n' ${GATE_JOB}
  printf 'r5_postprocess_job\t%s\n' ${POST_JOB}
  printf 'r5_submission_mode\tsite_bundle_equivalent\n'
  printf 'r5_cells\t24\n'
  printf 'r5_cost_profiles\t9\n'
  printf 'r5_jobs\t%s\n' ${R5}/jobs.tsv
  printf 'r5_bundle_manifest\t%s\n' ${R5}/site_bundle_submission.json
} > ${RECEIPT}
sha256sum ${RECEIPT} | awk '{print $1}' > ${RECEIPT}.sha256

echo BUNDLE_DEPLOYMENT_OK
echo CANCELLED_DUPLICATES="${cancelled_ids[*]}"
echo BUNDLE_JOBS="${bundle_jobs[*]}"
echo POSTPROCESS_JOB=${POST_JOB}
cat ${R5}/jobs.tsv
cat ${R5}/site_bundle_submission.json
