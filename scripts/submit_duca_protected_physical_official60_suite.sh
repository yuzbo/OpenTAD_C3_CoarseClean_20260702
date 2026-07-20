#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_PROTECTED_PHYSICAL_SUBMIT_OFFICIAL60][FAIL] $*" >&2
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
SUITE_KIND="${DUCA_PROTECTED_SUITE_KIND:-original_protected}"
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
  "${EXPECTED_COMMIT}" "${PROTOCOL_SHA256}" "${SUITE_KIND}" <<'PY'
import json
import sys
from pathlib import Path

protocol = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
authorization = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if (
    protocol.get("schema") != "duca_protected_physical_protocol_manifest_v1"
    or protocol.get("ok") is not True
    or protocol.get("git_commit") != sys.argv[3]
):
    raise SystemExit("P0 manifest schema/status/commit mismatch")
if (
    authorization.get("schema")
    != "duca_protected_physical_authorization_v1"
    or authorization.get("ok") is not True
    or authorization.get("git_commit") != sys.argv[3]
    or authorization.get("protocol_manifest_sha256") != sys.argv[4]
    or authorization.get("authorized_scope", {}).get(
        "official60_four_arm_training"
    )
    is not True
    or authorization.get("paper_claim_allowed") is not False
):
    raise SystemExit("P0-P3 authorization contract mismatch")
if (
    sys.argv[5] == "uni_companion_optimization"
    and authorization.get("authorized_scope", {}).get(
        "official60_uni_companion_training"
    )
    is not True
):
    raise SystemExit("authorization does not include Uni companion training")
PY

mkdir -p "${RUN_ROOT}/jobs" "${RUN_ROOT}/logs" "${RUN_ROOT}/arms"
SHORT_COMMIT="${EXPECTED_COMMIT:0:7}"
case "${SUITE_KIND}" in
  original_protected)
    VARIANTS=(
      exact_uniform
      transition_no_bridge
      protected_e2e
      protected_e2e_rho001
    )
    ;;
  uni_companion_optimization)
    VARIANTS=(
      exact_uniform
      protected_e2e
      protected_e2e_bridge025
      protected_e2e_uni_companion
    )
    ;;
  *)
    fail "unknown DUCA_PROTECTED_SUITE_KIND: ${SUITE_KIND}"
    ;;
esac

for variant in "${VARIANTS[@]}"; do
  job_file="${RUN_ROOT}/jobs/${variant}.sbatch"
  run_dir="${RUN_ROOT}/arms/${variant}/run"
  work_dir="${RUN_ROOT}/arms/${variant}/work"
  [[ ! -e "${job_file}" && ! -e "${run_dir}" && ! -e "${work_dir}" ]] \
    || fail "variant output already exists: ${variant}"
  cat > "${job_file}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=dp60_${variant}_${SHORT_COMMIT}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=${RUN_ROOT}/logs/${variant}-%j.out
#SBATCH --error=${RUN_ROOT}/logs/${variant}-%j.err
set -euo pipefail
module load cuda/11.8
module load miniforge3/24.11
source '${BASE}/conda_envs/opentad/bin/activate'
cd '${REPO_ROOT}'
export BASE='${BASE}'
export DUCA_EXPECTED_COMMIT='${EXPECTED_COMMIT}'
export DUCA_PROTECTED_PROTOCOL_MANIFEST_JSON='${PROTOCOL_JSON}'
export DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256='${PROTOCOL_SHA256}'
export DUCA_PROTECTED_AUTHORIZATION_JSON='${AUTHORIZATION_JSON}'
export DUCA_PROTECTED_AUTHORIZATION_SHA256='${AUTHORIZATION_SHA256}'
export DUCA_PROTECTED_VARIANT='${variant}'
export RUN_DIR='${run_dir}'
export WORK_DIR='${work_dir}'
bash scripts/run_duca_protected_physical_official60_variant_gpu1.sh
EOF
  chmod 0755 "${job_file}"
done

SUITE_JSON="${RUN_ROOT}/official60_suite.json"
COMPLETE_JOB="${RUN_ROOT}/jobs/complete.sbatch"
cat > "${COMPLETE_JOB}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=dp60_complete_${SHORT_COMMIT}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=${RUN_ROOT}/logs/complete-%j.out
#SBATCH --error=${RUN_ROOT}/logs/complete-%j.err
set -euo pipefail
module load miniforge3/24.11
source '${BASE}/conda_envs/opentad/bin/activate'
cd '${REPO_ROOT}'
'${PYTHON}' -m tools.bata.aggregate_duca_protected_physical_official60 \
  --expected-commit '${EXPECTED_COMMIT}' \
  --protocol-manifest-sha256 '${PROTOCOL_SHA256}' \
  --authorization-sha256 '${AUTHORIZATION_SHA256}' \
EOF
for variant in "${VARIANTS[@]}"; do
  printf "  --evidence '%s' \\\\\n" \
    "${RUN_ROOT}/arms/${variant}/run/post_run_evidence.json" \
    >> "${COMPLETE_JOB}"
done
printf "  --output-json '%s'\n" "${SUITE_JSON}" >> "${COMPLETE_JOB}"
chmod 0755 "${COMPLETE_JOB}"

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  bash -n "${RUN_ROOT}"/jobs/*.sbatch
  echo "[DUCA_PROTECTED_PHYSICAL_SUBMIT_OFFICIAL60] PRECHECK PASS ${RUN_ROOT}"
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

arm_ids=()
for variant in "${VARIANTS[@]}"; do
  submit_held "${RUN_ROOT}/jobs/${variant}.sbatch"
  arm_ids+=("${LAST_SUBMITTED_JOB_ID}")
done
dependency="afterok:$(IFS=:; echo "${arm_ids[*]}")"
submit_held "${COMPLETE_JOB}" --dependency="${dependency}"
complete_id="${LAST_SUBMITTED_JOB_ID}"

{
  printf 'key\tjob_id\tdependency\n'
  for index in "${!VARIANTS[@]}"; do
    printf '%s\t%s\tnone\n' "${VARIANTS[${index}]}" "${arm_ids[${index}]}"
  done
  printf 'complete\t%s\t%s\n' "${complete_id}" "${dependency}"
} > "${RUN_ROOT}/jobs.tsv"

for job in "${submitted[@]}"; do
  squeue --clusters="${TARGET_CLUSTER}" -h -j "${job}" -o '%A|%T|%j' \
    | grep -q "^${job}|PENDING|" || fail "held job ${job} is not pending"
done
for job in "${submitted[@]}"; do
  scontrol --clusters="${TARGET_CLUSTER}" release "${job}"
done
transaction_committed=1
trap - EXIT INT TERM
echo "[DUCA_PROTECTED_PHYSICAL_SUBMIT_OFFICIAL60] submitted ${RUN_ROOT}"
cat "${RUN_ROOT}/jobs.tsv"
