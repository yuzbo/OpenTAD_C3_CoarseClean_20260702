#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_PROTECTED_PHYSICAL_SUBMIT_GATES][FAIL] $*" >&2
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
[[ "$(sha256sum "${PROTOCOL_JSON}" | awk '{print $1}')" == "${PROTOCOL_SHA256}" ]] || fail "P0 hash drift"
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
HOMOTOPY_GATE="${RUN_ROOT}/gates/protected_e2e_homotopy025.json"
UNI_COMPANION_GATE="${RUN_ROOT}/gates/protected_e2e_uni_companion.json"
RHO_GATE="${RUN_ROOT}/gates/protected_e2e_rho001.json"
SHORT_SHARD="${RUN_ROOT}/p3/short.json"
MEDIUM_SHARD="${RUN_ROOT}/p3/medium.json"
LONG_SHARD="${RUN_ROOT}/p3/long.json"
P3_AGGREGATE="${RUN_ROOT}/p3/aggregate.json"
AUTHORIZATION="${RUN_ROOT}/authorization.json"
SHORT_COMMIT="${EXPECTED_COMMIT:0:7}"

write_gpu_job() {
  local key="$1"
  local body="$2"
  local hours="$3"
  local job_file="${RUN_ROOT}/jobs/${key}.sbatch"
  [[ ! -e "${job_file}" ]] || fail "job file already exists: ${job_file}"
  cat > "${job_file}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=dp_${key}_${SHORT_COMMIT}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=${hours}:00:00
#SBATCH --output=${RUN_ROOT}/logs/${key}-%j.out
#SBATCH --error=${RUN_ROOT}/logs/${key}-%j.err
source /etc/profile
set -euo pipefail
module load cuda/11.8
module load miniforge3/24.11
source '${BASE}/conda_envs/opentad/bin/activate'
cd '${REPO_ROOT}'
export BASE='${BASE}'
export DUCA_EXPECTED_COMMIT='${EXPECTED_COMMIT}'
export DUCA_PROTECTED_PROTOCOL_MANIFEST_JSON='${PROTOCOL_JSON}'
export DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256='${PROTOCOL_SHA256}'
${body}
EOF
  chmod 0755 "${job_file}"
}

write_gpu_job \
  "gate_main" \
  "export DUCA_PROTECTED_GATE_ARM='protected_e2e'
export DUCA_PROTECTED_GATE_OUTPUT_JSON='${MAIN_GATE}'
bash scripts/run_duca_protected_physical_full_model_gate_gpu1.sh" \
  2
write_gpu_job \
  "gate_bridge025" \
  "export DUCA_PROTECTED_GATE_ARM='protected_e2e_bridge025'
export DUCA_PROTECTED_GATE_OUTPUT_JSON='${BRIDGE025_GATE}'
bash scripts/run_duca_protected_physical_full_model_gate_gpu1.sh" \
  2
write_gpu_job \
  "gate_homotopy025" \
  "export DUCA_PROTECTED_GATE_ARM='protected_e2e_homotopy025'
export DUCA_PROTECTED_GATE_OUTPUT_JSON='${HOMOTOPY_GATE}'
bash scripts/run_duca_protected_physical_full_model_gate_gpu1.sh" \
  2
write_gpu_job \
  "gate_uni_companion" \
  "export DUCA_PROTECTED_GATE_ARM='protected_e2e_uni_companion'
export DUCA_PROTECTED_GATE_OUTPUT_JSON='${UNI_COMPANION_GATE}'
bash scripts/run_duca_protected_physical_full_model_gate_gpu1.sh" \
  2
write_gpu_job \
  "gate_rho" \
  "export DUCA_PROTECTED_GATE_ARM='protected_e2e_rho001'
export DUCA_PROTECTED_GATE_OUTPUT_JSON='${RHO_GATE}'
bash scripts/run_duca_protected_physical_full_model_gate_gpu1.sh" \
  2
for stratum in short medium long; do
  case "${stratum}" in
    short) output="${SHORT_SHARD}" ;;
    medium) output="${MEDIUM_SHARD}" ;;
    long) output="${LONG_SHARD}" ;;
  esac
  write_gpu_job \
    "p3_${stratum}" \
    "export DUCA_PROTECTED_P3_STRATUM='${stratum}'
export DUCA_PROTECTED_P3_OUTPUT_JSON='${output}'
bash scripts/run_duca_protected_physical_p3_shard_gpu1.sh" \
    12
done

COMPLETE_JOB="${RUN_ROOT}/jobs/complete.sbatch"
cat > "${COMPLETE_JOB}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=dp_complete_${SHORT_COMMIT}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=${RUN_ROOT}/logs/complete-%j.out
#SBATCH --error=${RUN_ROOT}/logs/complete-%j.err
source /etc/profile
set -euo pipefail
module load miniforge3/24.11
source '${BASE}/conda_envs/opentad/bin/activate'
cd '${REPO_ROOT}'
export BASE='${BASE}'
export DUCA_EXPECTED_COMMIT='${EXPECTED_COMMIT}'
export DUCA_PROTECTED_PROTOCOL_MANIFEST_JSON='${PROTOCOL_JSON}'
export DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256='${PROTOCOL_SHA256}'
export DUCA_PROTECTED_MAIN_GATE_JSON='${MAIN_GATE}'
export DUCA_PROTECTED_BRIDGE025_GATE_JSON='${BRIDGE025_GATE}'
export DUCA_PROTECTED_HOMOTOPY_GATE_JSON='${HOMOTOPY_GATE}'
export DUCA_PROTECTED_UNI_COMPANION_GATE_JSON='${UNI_COMPANION_GATE}'
export DUCA_PROTECTED_RHO_GATE_JSON='${RHO_GATE}'
export DUCA_PROTECTED_P3_SHORT_JSON='${SHORT_SHARD}'
export DUCA_PROTECTED_P3_MEDIUM_JSON='${MEDIUM_SHARD}'
export DUCA_PROTECTED_P3_LONG_JSON='${LONG_SHARD}'
export DUCA_PROTECTED_P3_AGGREGATE_JSON='${P3_AGGREGATE}'
export DUCA_PROTECTED_AUTHORIZATION_JSON='${AUTHORIZATION}'
bash scripts/complete_duca_protected_physical_gate_suite.sh
EOF
chmod 0755 "${COMPLETE_JOB}"

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  bash -n "${RUN_ROOT}"/jobs/*.sbatch
  echo "[DUCA_PROTECTED_PHYSICAL_SUBMIT_GATES] PRECHECK PASS ${RUN_ROOT}"
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || fail "sbatch is unavailable"
command -v squeue >/dev/null 2>&1 || fail "squeue is unavailable"
command -v scontrol >/dev/null 2>&1 || fail "scontrol is unavailable"
command -v scancel >/dev/null 2>&1 || fail "scancel is unavailable"
submitted=()
transaction_committed=0
cleanup_new_jobs() {
  [[ "${transaction_committed}" == "0" ]] || return 0
  local job
  for job in "${submitted[@]}"; do
    scancel --clusters="${TARGET_CLUSTER}" "${job}" >/dev/null 2>&1 || true
  done
}
trap cleanup_new_jobs EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

LAST_SUBMITTED_JOB_ID=""
submit_held() {
  local job_file="$1"
  shift
  local response job_id
  response="$(sbatch --hold --parsable --clusters="${TARGET_CLUSTER}" "$@" "${job_file}")"
  response="${response%%$'\n'*}"
  job_id="${response%%;*}"
  [[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || fail "invalid sbatch response: ${response}"
  submitted+=("${job_id}")
  LAST_SUBMITTED_JOB_ID="${job_id}"
}

submit_held "${RUN_ROOT}/jobs/gate_main.sbatch"
gate_main_id="${LAST_SUBMITTED_JOB_ID}"
submit_held "${RUN_ROOT}/jobs/gate_bridge025.sbatch"
gate_bridge025_id="${LAST_SUBMITTED_JOB_ID}"
submit_held "${RUN_ROOT}/jobs/gate_homotopy025.sbatch"
gate_homotopy025_id="${LAST_SUBMITTED_JOB_ID}"
submit_held "${RUN_ROOT}/jobs/gate_uni_companion.sbatch"
gate_uni_companion_id="${LAST_SUBMITTED_JOB_ID}"
submit_held "${RUN_ROOT}/jobs/gate_rho.sbatch"
gate_rho_id="${LAST_SUBMITTED_JOB_ID}"
submit_held "${RUN_ROOT}/jobs/p3_short.sbatch"
p3_short_id="${LAST_SUBMITTED_JOB_ID}"
submit_held "${RUN_ROOT}/jobs/p3_medium.sbatch"
p3_medium_id="${LAST_SUBMITTED_JOB_ID}"
submit_held "${RUN_ROOT}/jobs/p3_long.sbatch"
p3_long_id="${LAST_SUBMITTED_JOB_ID}"
dependency="afterok:${gate_main_id}:${gate_bridge025_id}:${gate_homotopy025_id}:${gate_uni_companion_id}:${gate_rho_id}:${p3_short_id}:${p3_medium_id}:${p3_long_id}"
submit_held "${COMPLETE_JOB}" --dependency="${dependency}"
complete_id="${LAST_SUBMITTED_JOB_ID}"

cat > "${RUN_ROOT}/jobs.tsv" <<EOF
key	job_id	dependency
gate_main	${gate_main_id}	none
gate_bridge025	${gate_bridge025_id}	none
gate_homotopy025	${gate_homotopy025_id}	none
gate_uni_companion	${gate_uni_companion_id}	none
gate_rho	${gate_rho_id}	none
p3_short	${p3_short_id}	none
p3_medium	${p3_medium_id}	none
p3_long	${p3_long_id}	none
complete	${complete_id}	${dependency}
EOF
for job in "${submitted[@]}"; do
  squeue --clusters="${TARGET_CLUSTER}" -h -j "${job}" -o '%A|%T|%j' \
    | grep -q "^${job}|PENDING|" || fail "held job ${job} is not pending"
done
for job in "${submitted[@]}"; do
  scontrol --clusters="${TARGET_CLUSTER}" release "${job}"
done
transaction_committed=1
trap - EXIT INT TERM
echo "[DUCA_PROTECTED_PHYSICAL_SUBMIT_GATES] submitted ${RUN_ROOT}"
cat "${RUN_ROOT}/jobs.tsv"
