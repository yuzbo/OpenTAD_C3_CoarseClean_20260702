#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_TWO_STAGE_SERIAL_SUBMIT][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export DUCA_CELLCF_TRAINING_PROFILE=official60
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

RUN_ROOT="${RUN_ROOT:-}"
TARGET_CLUSTER="${DUCA_TARGET_CLUSTER:-n16r4}"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
[[ -n "${RUN_ROOT}" ]] || fail "RUN_ROOT is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/submission"
exec 9>"${RUN_ROOT}/submission/submit.lock"
flock -n 9 || fail "another serial submission process holds the lock"
if [[ -f "${RUN_ROOT}/jobs.tsv" ]]; then
  cat "${RUN_ROOT}/jobs.tsv"
  exit 0
fi

SPLIT_ROOT="${RUN_ROOT}/frontend_split"
if [[ ! -f "${SPLIT_ROOT}/frontend_split_manifest.json" ]]; then
  "${PYTHON}" tools/bata/create_duca_frontend_split.py \
    --annotation "${THUMOS14_ANNOTATION_PATH}" \
    --output-dir "${SPLIT_ROOT}" \
    --seed 3407 \
    --holdout-fraction 0.20 \
    > "${RUN_ROOT}/frontend_split.out"
fi
SPLIT_MANIFEST="${SPLIT_ROOT}/frontend_split_manifest.json"
SPLIT_SHA256="$(sha256sum "${SPLIT_MANIFEST}" | awk '{print $1}')"
JOB_FILE="${RUN_ROOT}/serial.sbatch"
if [[ ! -f "${JOB_FILE}" ]]; then
  cat > "${JOB_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=d2_serial_${EXPECTED_COMMIT:0:7}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=${RUN_ROOT}/logs/serial-%j.out
#SBATCH --error=${RUN_ROOT}/logs/serial-%j.err
source /etc/profile
set -euo pipefail
module load cuda/11.8
module load miniforge3/24.11
source '${BASE}/conda_envs/opentad/bin/activate'
cd '${REPO_ROOT}'
export BASE='${BASE}'
export DUCA_EXPECTED_COMMIT='${EXPECTED_COMMIT}'
export RUN_ROOT='${RUN_ROOT}'
export DUCA_FRONTEND_SPLIT_MANIFEST='${SPLIT_MANIFEST}'
export DUCA_FRONTEND_SPLIT_MANIFEST_SHA256='${SPLIT_SHA256}'
bash scripts/run_duca_two_stage_curriculum_serial_gpu1.sh
EOF
  chmod 0755 "${JOB_FILE}"
fi
bash -n "${JOB_FILE}"
if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  echo "[DUCA_TWO_STAGE_SERIAL_SUBMIT] PRECHECK PASS ${RUN_ROOT}"
  exit 0
fi

TOKEN="duca-two-stage-${EXPECTED_COMMIT:0:12}-seed3407-serial"
printf '%s\n' "${TOKEN}" > "${RUN_ROOT}/submission/intent.txt"
raw="$(sbatch --parsable --clusters="${TARGET_CLUSTER}" --comment="${TOKEN}" "${JOB_FILE}")"
raw="${raw%%$'\n'*}"
job_id="${raw%%;*}"
[[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || fail "invalid sbatch response: ${raw}"
{
  printf 'key\tjob_id\tdependency\n'
  printf 'two_stage_serial\t%s\tnone\n' "${job_id}"
} > "${RUN_ROOT}/jobs.tsv"
cat > "${RUN_ROOT}/submission/receipt.json" <<EOF
{
  "schema": "duca_two_stage_serial_submission_v1",
  "git_commit": "${EXPECTED_COMMIT}",
  "job_id": ${job_id},
  "cluster": "${TARGET_CLUSTER}",
  "submission_token": "${TOKEN}",
  "split_manifest_sha256": "${SPLIT_SHA256}"
}
EOF
cat "${RUN_ROOT}/jobs.tsv"
