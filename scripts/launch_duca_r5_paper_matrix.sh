#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd -P)}"
OUTPUT_DIR="${R5_OUTPUT_DIR:?set R5_OUTPUT_DIR}"
LEARNED_CONFIG="${R5_LEARNED_CONFIG:?set R5_LEARNED_CONFIG to the R0-selected reviewed G1 config}"
UNIFORM_CONFIG="${R5_UNIFORM_CONFIG:-${REPO_ROOT}/configs/adatad/thumos/duca_two_stage_exact_uniform_fixed384_official60.py}"
TARGET_CLUSTER="${TARGET_CLUSTER:-n16r4}"
R5_SUBMIT="${R5_SUBMIT:-0}"
R5_UPSTREAM_DEPENDENCY="${R5_UPSTREAM_DEPENDENCY:-}"

cd "${REPO_ROOT}"
python -m tools.bata.duca_r5_paper_matrix \
  --repo-root "${REPO_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --uniform-config "${UNIFORM_CONFIG}" \
  --learned-config "${LEARNED_CONFIG}" \
  --cluster "${TARGET_CLUSTER}"

if [[ "${R5_SUBMIT}" != 1 ]]; then
  echo "Generated only; set R5_SUBMIT=1 to submit the matrix."
  echo "Mechanism gate: sbatch ${OUTPUT_DIR}/jobs/temporalmaxer_one_step.sbatch"
  echo "Matrix index:  ${OUTPUT_DIR}/cells.tsv"
  exit 0
fi

[[ "${R5_UPSTREAM_DEPENDENCY}" =~ ^afterok:[1-9][0-9]*$ ]] \
  || { echo "R5_UPSTREAM_DEPENDENCY must be afterok:<job_id>" >&2; exit 1; }
[[ ! -e "${OUTPUT_DIR}/jobs.tsv" ]] \
  || { echo "refusing to overwrite ${OUTPUT_DIR}/jobs.tsv" >&2; exit 1; }

gate_raw="$(sbatch --parsable --clusters="${TARGET_CLUSTER}" \
  --dependency="${R5_UPSTREAM_DEPENDENCY}" \
  "${OUTPUT_DIR}/jobs/temporalmaxer_one_step.sbatch")"
gate_job="${gate_raw%%;*}"
[[ "${gate_job}" =~ ^[1-9][0-9]*$ ]]
printf 'role\tjob_id\tdependency\tsbatch\n' > "${OUTPUT_DIR}/jobs.tsv"
printf 'temporalmaxer_one_step\t%s\t%s\t%s\n' \
  "${gate_job}" "${R5_UPSTREAM_DEPENDENCY}" \
  "${OUTPUT_DIR}/jobs/temporalmaxer_one_step.sbatch" >> "${OUTPUT_DIR}/jobs.tsv"

tail -n +2 "${OUTPUT_DIR}/cells.tsv" | while IFS=$'\t' read -r \
  cell_id backend arm budget seed config config_sha256 sbatch_file sbatch_sha256; do
  raw="$(sbatch --parsable --clusters="${TARGET_CLUSTER}" \
    --dependency="afterok:${gate_job}" "${sbatch_file}")"
  job_id="${raw%%;*}"
  [[ "${job_id}" =~ ^[1-9][0-9]*$ ]]
  printf '%s\t%s\tafterok:%s\t%s\n' \
    "${cell_id}" "${job_id}" "${gate_job}" "${sbatch_file}" \
    >> "${OUTPUT_DIR}/jobs.tsv"
done

tail -n +2 "${OUTPUT_DIR}/costs.tsv" | while IFS=$'\t' read -r \
  cost_id source_cell sbatch_file summary; do
  source_job="$(awk -F '\t' -v role="${source_cell}" '$1 == role {print $2}' \
    "${OUTPUT_DIR}/jobs.tsv")"
  [[ "${source_job}" =~ ^[1-9][0-9]*$ ]]
  raw="$(sbatch --parsable --clusters="${TARGET_CLUSTER}" \
    --dependency="afterok:${source_job}" "${sbatch_file}")"
  job_id="${raw%%;*}"
  [[ "${job_id}" =~ ^[1-9][0-9]*$ ]]
  printf '%s\t%s\tafterok:%s\t%s\n' \
    "${cost_id}" "${job_id}" "${source_job}" "${sbatch_file}" \
    >> "${OUTPUT_DIR}/jobs.tsv"
done

all_dependencies=()
while IFS=$'\t' read -r role job_id dependency sbatch_file; do
  [[ "${role}" == role || "${role}" == temporalmaxer_one_step ]] && continue
  all_dependencies+=("${job_id}")
done < "${OUTPUT_DIR}/jobs.tsv"
aggregate_dependency="afterok:$(IFS=:; echo "${all_dependencies[*]}")"
aggregate_raw="$(sbatch --parsable --clusters="${TARGET_CLUSTER}" \
  --dependency="${aggregate_dependency}" "${OUTPUT_DIR}/jobs/aggregate.sbatch")"
aggregate_job="${aggregate_raw%%;*}"
[[ "${aggregate_job}" =~ ^[1-9][0-9]*$ ]]
printf 'aggregate\t%s\t%s\t%s\n' \
  "${aggregate_job}" "${aggregate_dependency}" \
  "${OUTPUT_DIR}/jobs/aggregate.sbatch" >> "${OUTPUT_DIR}/jobs.tsv"
sha256sum "${OUTPUT_DIR}/jobs.tsv" | awk '{print $1}' > \
  "${OUTPUT_DIR}/jobs.tsv.sha256"

echo "Submitted R5 gate, 24 train+terminal-eval jobs, 4 cost jobs, and aggregate."
cat "${OUTPUT_DIR}/jobs.tsv"
