#!/usr/bin/env bash
set -euo pipefail

BASE=/data/run01/sczc063/yuzibo
REPO="${BASE}/projects/opentad_duca_boundary_d9fb398_20260722"
RUN_ROOT="${BASE}/projects/c3_lowres_action_probe/duca_boundary_d9fb398_r0_formal_20260722_112357"
EXPECTED_COMMIT=d9fb398578716d278e818745677a92976bcedf2c
R0_JOB=1179517
DEPENDENCY="afterok:${R0_JOB}"
JOURNAL="${RUN_ROOT}/jobs.tsv"
SEAL="${RUN_ROOT}/jobs.complete.json"
TARGET_CLUSTER=n16r4

set +u
source /etc/profile
set -u
module load cuda/11.8
module load miniforge3/24.11
source "${BASE}/conda_envs/opentad/bin/activate"
cd "${REPO}"

[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
[[ -f "${RUN_ROOT}/submission/p0.sbatch" ]]

if awk -F '\t' '$1 == "p0" && $2 ~ /^[0-9]+$/ { found=1 } END { exit !found }' "${JOURNAL}"; then
  echo "P0_ALREADY_RECORDED"
  cat "${JOURNAL}"
  exit 0
fi

python -m tools.bata.duca_boundary_burst_submission_journal \
  --journal "${JOURNAL}" --seal "${SEAL}" \
  --expected-commit "${EXPECTED_COMMIT}" --target-cluster "${TARGET_CLUSTER}" \
  reserve --role p0 --dependency "${DEPENDENCY}"

job_id="$(sbatch --parsable --clusters="${TARGET_CLUSTER}" \
  --dependency="${DEPENDENCY}" "${RUN_ROOT}/submission/p0.sbatch")"
job_id="${job_id%%;*}"
[[ "${job_id}" =~ ^[0-9]+$ ]]

python -m tools.bata.duca_boundary_burst_submission_journal \
  --journal "${JOURNAL}" --seal "${SEAL}" \
  --expected-commit "${EXPECTED_COMMIT}" --target-cluster "${TARGET_CLUSTER}" \
  record --role p0 --job-id "${job_id}" --dependency "${DEPENDENCY}"

echo "P0_JOB=${job_id}"
sha256sum "${JOURNAL}"
cat "${JOURNAL}"
squeue -j "${R0_JOB},${job_id}" -o '%i|%j|%T|%M|%R' || true
