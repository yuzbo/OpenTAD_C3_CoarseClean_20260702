#!/usr/bin/env bash
set -euo pipefail

R5=/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_e49ef69_formal_20260722_155037_r5
GATE=$(awk -F '\t' '$1 == "temporalmaxer_one_step" {print $2}' ${R5}/jobs.tsv)
[[ ${GATE} =~ ^[1-9][0-9]*$ ]]
next_role=
next_sbatch=
while IFS=$'\t' read -r role backend arm budget seed config config_sha sbatch_file sbatch_sha; do
  [[ ${role} == id ]] && continue
  if ! awk -F '\t' -v role="${role}" '$1 == role {found=1} END {exit !found}' ${R5}/jobs.tsv; then
    next_role=${role}
    next_sbatch=${sbatch_file}
    break
  fi
done < ${R5}/cells.tsv
echo "NEXT_ROLE=${next_role}"
echo "NEXT_SBATCH=${next_sbatch}"
[[ -n ${next_role} && -f ${next_sbatch} ]]
set +e
raw=$(sbatch --parsable --clusters=n16r4 --dependency=afterok:${GATE} ${next_sbatch} 2>&1)
status=$?
set -e
echo "SBATCH_STATUS=${status}"
echo "SBATCH_RAW=${raw}"
if [[ ${status} -eq 0 ]]; then
  job=${raw%%;*}
  [[ ${job} =~ ^[1-9][0-9]*$ ]]
  printf '%s\t%s\tafterok:%s\t%s\n' ${next_role} ${job} ${GATE} ${next_sbatch} >> ${R5}/jobs.tsv
  echo "RECORDED_JOB=${job}"
fi
