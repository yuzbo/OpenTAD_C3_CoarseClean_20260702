#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_HOMOTOPY_OFFICIAL60_SUBMIT][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${REPO_ROOT}/scripts/duca_protected_physical_env.sh"

RUN_ROOT="${RUN_ROOT:-}"
TARGET_CLUSTER="${DUCA_TARGET_CLUSTER:-n16r4}"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
PROTOCOL_JSON="${DUCA_PROTECTED_PROTOCOL_MANIFEST_JSON:-}"
PROTOCOL_SHA256="${DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256:-}"
AUTHORIZATION_JSON="${DUCA_PROTECTED_AUTHORIZATION_JSON:-}"
AUTHORIZATION_SHA256="${DUCA_PROTECTED_AUTHORIZATION_SHA256:-}"
VARIANT="protected_e2e_homotopy025"

[[ -n "${RUN_ROOT}" && -d "${RUN_ROOT}" ]] || fail "prepared RUN_ROOT is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -f "${PROTOCOL_JSON}" ]] || fail "P0 manifest is missing"
[[ "$(sha256sum "${PROTOCOL_JSON}" | awk '{print $1}')" == "${PROTOCOL_SHA256}" ]] \
  || fail "P0 hash drift"
[[ -f "${AUTHORIZATION_JSON}" ]] || fail "P0-P3 authorization is missing"
[[ "$(sha256sum "${AUTHORIZATION_JSON}" | awk '{print $1}')" == "${AUTHORIZATION_SHA256}" ]] \
  || fail "authorization hash drift"
[[ ! -e "${RUN_ROOT}/jobs.tsv" ]] || fail "RUN_ROOT was already submitted"

"${PYTHON}" - "${PROTOCOL_JSON}" "${AUTHORIZATION_JSON}" \
  "${EXPECTED_COMMIT}" "${PROTOCOL_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

protocol = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
authorization = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if (
    protocol.get("schema") != "duca_protected_physical_protocol_manifest_v1"
    or protocol.get("ok") is not True
    or protocol.get("git_commit") != sys.argv[3]
    or "protected_e2e_homotopy025" not in protocol.get("configs", {}).get("arms", {})
):
    raise SystemExit("P0 homotopy contract mismatch")
if (
    authorization.get("schema") != "duca_protected_physical_authorization_v1"
    or authorization.get("ok") is not True
    or authorization.get("git_commit") != sys.argv[3]
    or authorization.get("protocol_manifest_sha256") != sys.argv[4]
    or authorization.get("authorized_scope", {}).get(
        "official60_homotopy_training"
    )
    is not True
    or authorization.get("paper_claim_allowed") is not False
):
    raise SystemExit("homotopy authorization contract mismatch")
PY

mkdir -p "${RUN_ROOT}/jobs" "${RUN_ROOT}/logs" "${RUN_ROOT}/arm"
JOB_FILE="${RUN_ROOT}/jobs/${VARIANT}.sbatch"
RUN_DIR="${RUN_ROOT}/arm/run"
WORK_DIR="${RUN_ROOT}/arm/work"
SHORT_COMMIT="${EXPECTED_COMMIT:0:7}"
[[ ! -e "${JOB_FILE}" && ! -e "${RUN_DIR}" && ! -e "${WORK_DIR}" ]] \
  || fail "homotopy output already exists"

cat > "${JOB_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=duhom_${SHORT_COMMIT}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=${RUN_ROOT}/logs/homotopy-%j.out
#SBATCH --error=${RUN_ROOT}/logs/homotopy-%j.err
source /etc/profile
set -euo pipefail
module load cuda/11.8
module load miniforge3/24.11
source '${BASE}/conda_envs/opentad/bin/activate'
cd '${REPO_ROOT}'
export PYTHONNOUSERSITE=1
export BASE='${BASE}'
export DUCA_EXPECTED_COMMIT='${EXPECTED_COMMIT}'
export DUCA_PROTECTED_PROTOCOL_MANIFEST_JSON='${PROTOCOL_JSON}'
export DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256='${PROTOCOL_SHA256}'
export DUCA_PROTECTED_AUTHORIZATION_JSON='${AUTHORIZATION_JSON}'
export DUCA_PROTECTED_AUTHORIZATION_SHA256='${AUTHORIZATION_SHA256}'
export DUCA_PROTECTED_VARIANT='${VARIANT}'
export RUN_DIR='${RUN_DIR}'
export WORK_DIR='${WORK_DIR}'
bash scripts/run_duca_protected_physical_official60_variant_gpu1.sh
EOF
chmod 0755 "${JOB_FILE}"

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  bash -n "${JOB_FILE}"
  echo "[DUCA_HOMOTOPY_OFFICIAL60_SUBMIT] PRECHECK PASS ${RUN_ROOT}"
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || fail "sbatch is unavailable"
command -v squeue >/dev/null 2>&1 || fail "squeue is unavailable"
command -v scontrol >/dev/null 2>&1 || fail "scontrol is unavailable"
command -v scancel >/dev/null 2>&1 || fail "scancel is unavailable"

job_id=""
transaction_committed=0
cleanup_new_job() {
  [[ "${transaction_committed}" == "0" && -n "${job_id}" ]] || return 0
  scancel --clusters="${TARGET_CLUSTER}" "${job_id}" >/dev/null 2>&1 || true
}
trap cleanup_new_job EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

response="$(sbatch --hold --parsable --clusters="${TARGET_CLUSTER}" "${JOB_FILE}")"
response="${response%%$'\n'*}"
job_id="${response%%;*}"
[[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || fail "invalid sbatch response: ${response}"
squeue --clusters="${TARGET_CLUSTER}" -h -j "${job_id}" -o '%A|%T|%j' \
  | grep -q "^${job_id}|PENDING|" || fail "held job ${job_id} is not pending"
cat > "${RUN_ROOT}/jobs.tsv" <<EOF
key	job_id	dependency
${VARIANT}	${job_id}	none
EOF
scontrol --clusters="${TARGET_CLUSTER}" release "${job_id}"
transaction_committed=1
trap - EXIT INT TERM
echo "[DUCA_HOMOTOPY_OFFICIAL60_SUBMIT] submitted ${RUN_ROOT}"
cat "${RUN_ROOT}/jobs.tsv"
