#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_PROTECTED_PHYSICAL_SUBMIT_SINGLE_GATE][FAIL] $*" >&2
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
[[ -n "${RUN_ROOT}" && -d "${RUN_ROOT}" ]] || fail "prepared RUN_ROOT is required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ -f "${PROTOCOL_JSON}" ]] || fail "P0 manifest is missing"
[[ "$(sha256sum "${PROTOCOL_JSON}" | awk '{print $1}')" == "${PROTOCOL_SHA256}" ]] \
  || fail "P0 hash drift"
[[ ! -e "${RUN_ROOT}/jobs.tsv" ]] || fail "RUN_ROOT was already submitted"

"${PYTHON}" - "${PROTOCOL_JSON}" "${EXPECTED_COMMIT}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    payload.get("schema") != "duca_protected_physical_protocol_manifest_v1"
    or payload.get("ok") is not True
    or payload.get("git_commit") != sys.argv[2]
):
    raise SystemExit("P0 manifest schema/status/commit mismatch")
PY

mkdir -p "${RUN_ROOT}/jobs" "${RUN_ROOT}/logs" "${RUN_ROOT}/gates" \
  "${RUN_ROOT}/p3"
MAIN_GATE="${RUN_ROOT}/gates/protected_e2e.json"
BRIDGE025_GATE="${RUN_ROOT}/gates/protected_e2e_bridge025.json"
UNI_COMPANION_GATE="${RUN_ROOT}/gates/protected_e2e_uni_companion.json"
RHO_GATE="${RUN_ROOT}/gates/protected_e2e_rho001.json"
SHORT_SHARD="${RUN_ROOT}/p3/short.json"
MEDIUM_SHARD="${RUN_ROOT}/p3/medium.json"
LONG_SHARD="${RUN_ROOT}/p3/long.json"
P3_AGGREGATE="${RUN_ROOT}/p3/aggregate.json"
AUTHORIZATION="${RUN_ROOT}/authorization.json"
SHORT_COMMIT="${EXPECTED_COMMIT:0:7}"
JOB_FILE="${RUN_ROOT}/jobs/all_gates.sbatch"

cat > "${JOB_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=dp_all_${SHORT_COMMIT}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --output=${RUN_ROOT}/logs/all_gates-%j.out
#SBATCH --error=${RUN_ROOT}/logs/all_gates-%j.err
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

export DUCA_PROTECTED_GATE_ARM='protected_e2e'
export DUCA_PROTECTED_GATE_OUTPUT_JSON='${MAIN_GATE}'
bash scripts/run_duca_protected_physical_full_model_gate_gpu1.sh

export DUCA_PROTECTED_GATE_ARM='protected_e2e_bridge025'
export DUCA_PROTECTED_GATE_OUTPUT_JSON='${BRIDGE025_GATE}'
bash scripts/run_duca_protected_physical_full_model_gate_gpu1.sh

export DUCA_PROTECTED_GATE_ARM='protected_e2e_uni_companion'
export DUCA_PROTECTED_GATE_OUTPUT_JSON='${UNI_COMPANION_GATE}'
bash scripts/run_duca_protected_physical_full_model_gate_gpu1.sh

export DUCA_PROTECTED_GATE_ARM='protected_e2e_rho001'
export DUCA_PROTECTED_GATE_OUTPUT_JSON='${RHO_GATE}'
bash scripts/run_duca_protected_physical_full_model_gate_gpu1.sh

export DUCA_PROTECTED_P3_STRATUM='short'
export DUCA_PROTECTED_P3_OUTPUT_JSON='${SHORT_SHARD}'
bash scripts/run_duca_protected_physical_p3_shard_gpu1.sh

export DUCA_PROTECTED_P3_STRATUM='medium'
export DUCA_PROTECTED_P3_OUTPUT_JSON='${MEDIUM_SHARD}'
bash scripts/run_duca_protected_physical_p3_shard_gpu1.sh

export DUCA_PROTECTED_P3_STRATUM='long'
export DUCA_PROTECTED_P3_OUTPUT_JSON='${LONG_SHARD}'
bash scripts/run_duca_protected_physical_p3_shard_gpu1.sh

export DUCA_PROTECTED_MAIN_GATE_JSON='${MAIN_GATE}'
export DUCA_PROTECTED_BRIDGE025_GATE_JSON='${BRIDGE025_GATE}'
export DUCA_PROTECTED_UNI_COMPANION_GATE_JSON='${UNI_COMPANION_GATE}'
export DUCA_PROTECTED_RHO_GATE_JSON='${RHO_GATE}'
export DUCA_PROTECTED_P3_SHORT_JSON='${SHORT_SHARD}'
export DUCA_PROTECTED_P3_MEDIUM_JSON='${MEDIUM_SHARD}'
export DUCA_PROTECTED_P3_LONG_JSON='${LONG_SHARD}'
export DUCA_PROTECTED_P3_AGGREGATE_JSON='${P3_AGGREGATE}'
export DUCA_PROTECTED_AUTHORIZATION_JSON='${AUTHORIZATION}'
bash scripts/complete_duca_protected_physical_gate_suite.sh
EOF
chmod 0755 "${JOB_FILE}"

cat > "${RUN_ROOT}/execution_plan.tsv" <<'EOF'
order	component
1	gate_main
2	gate_bridge025
3	gate_uni_companion
4	gate_rho
5	p3_short
6	p3_medium
7	p3_long
8	complete
EOF

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  bash -n "${JOB_FILE}"
  echo "[DUCA_PROTECTED_PHYSICAL_SUBMIT_SINGLE_GATE] PRECHECK PASS ${RUN_ROOT}"
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
all_gates	${job_id}	none
EOF
scontrol --clusters="${TARGET_CLUSTER}" release "${job_id}"
transaction_committed=1
trap - EXIT INT TERM
echo "[DUCA_PROTECTED_PHYSICAL_SUBMIT_SINGLE_GATE] submitted ${RUN_ROOT}"
cat "${RUN_ROOT}/jobs.tsv"
