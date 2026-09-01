source /etc/profile
set -euo pipefail

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

BASE=/data/run01/sczc063/yuzibo
SNAPSHOT=${BASE}/projects/opentad_duca_boundary_d9fb398_20260722
EXPECTED=d9fb398578716d278e818745677a92976bcedf2c
CHECKPOINT=${BASE}/projects/c3_lowres_action_probe/duca_cellcf_1642f26_formal_seed0_20260717_0200/work_dirs/transition_beta0/gpu1_id0/checkpoint/epoch_131.pth
CHECKPOINT_SHA256=f4ac9891b7cfffd1ab482f28a43086a6e862112f6ffbcb79c7b86c3d2ed935ac
STAMP=$(date '+%Y%m%d_%H%M%S')
RUN_ROOT=${BASE}/projects/c3_lowres_action_probe/duca_boundary_d9fb398_r0_formal_${STAMP}

cd "${SNAPSHOT}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED}" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
[[ -f "${CHECKPOINT}" ]]
[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${CHECKPOINT_SHA256}" ]]
[[ ! -e "${RUN_ROOT}" ]]

export BASE RUN_ROOT
export DUCA_EXPECTED_COMMIT=${EXPECTED}
export DUCA_TARGET_CLUSTER=n16r4
export DUCA_R0_CHECKPOINT=${CHECKPOINT}
export DUCA_R0_CHECKPOINT_EPOCH=131
PRECHECK_ONLY=1 bash scripts/submit_duca_boundary_burst_official60_suite.sh

JOURNAL=${RUN_ROOT}/jobs.tsv
SEAL=${RUN_ROOT}/jobs.complete.json
python -m tools.bata.duca_boundary_burst_submission_journal \
  --journal "${JOURNAL}" --seal "${SEAL}" \
  --expected-commit "${EXPECTED}" --target-cluster n16r4 initialize
python -m tools.bata.duca_boundary_burst_submission_journal \
  --journal "${JOURNAL}" --seal "${SEAL}" \
  --expected-commit "${EXPECTED}" --target-cluster n16r4 \
  reserve --role r0_holdout_map --dependency none

RAW=$(sbatch --parsable --clusters=n16r4 "${RUN_ROOT}/submission/r0.sbatch")
JOB_ID=${RAW%%;*}
[[ "${JOB_ID}" =~ ^[1-9][0-9]*$ ]]
python -m tools.bata.duca_boundary_burst_submission_journal \
  --journal "${JOURNAL}" --seal "${SEAL}" \
  --expected-commit "${EXPECTED}" --target-cluster n16r4 \
  record --role r0_holdout_map --job-id "${JOB_ID}" --dependency none

echo "R0_ONLY_RUN_ROOT=${RUN_ROOT}"
echo "R0_ONLY_JOB_ID=${JOB_ID}"
echo 'R0_ONLY_HASHES_BEGIN'
sha256sum \
  "${RUN_ROOT}/submission_manifest.json" \
  "${RUN_ROOT}/submission_manifest.sha256" \
  "${RUN_ROOT}/submission/r0.sbatch" \
  "${RUN_ROOT}/jobs.tsv" \
  "${RUN_ROOT}/frontend_split/frontend_split_manifest.json"
echo 'R0_ONLY_HASHES_END'
cat "${RUN_ROOT}/jobs.tsv"
squeue -j "${JOB_ID}" -o '%i|%j|%T|%M|%R'
