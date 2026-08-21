#!/usr/bin/env bash
set -euo pipefail

fail() { echo "[DUCA_TRUETIME_PAIR_SUBMIT][FAIL] $*" >&2; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${ROOT}/scripts/duca_protected_physical_env.sh"
RUN_ROOT="${RUN_ROOT:-}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-}"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
TARGET_CLUSTER="${DUCA_TARGET_CLUSTER:-n16r4}"

[[ -n "${RUN_ROOT}" && -n "${EVIDENCE_ROOT}" ]] || fail "RUN_ROOT/EVIDENCE_ROOT required"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "exact commit required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "clean tree required"
[[ ! -e "${RUN_ROOT}/jobs.tsv" ]] || fail "run root already submitted"
mkdir -p "${RUN_ROOT}/jobs" "${RUN_ROOT}/logs"

declare -a ARMS=(RANKPACK_K384 TRUETIME_K384)
declare -a IDS=()
for ARM in "${ARMS[@]}"; do
  KEY="$(printf '%s' "${ARM}" | tr '[:upper:]' '[:lower:]')"
  PROTOCOL="${EVIDENCE_ROOT}/${KEY}/protocol.json"
  AUTH="${EVIDENCE_ROOT}/${KEY}/authorization.json"
  [[ -f "${PROTOCOL}" && -f "${AUTH}" ]] || fail "missing PRE_RUN evidence for ${ARM}"
  PROTOCOL_SHA="$(sha256sum "${PROTOCOL}" | awk '{print $1}')"
  AUTH_SHA="$(sha256sum "${AUTH}" | awk '{print $1}')"
  JOB_FILE="${RUN_ROOT}/jobs/${KEY}.sbatch"
  cat > "${JOB_FILE}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=ducat_${KEY}_${EXPECTED_COMMIT:0:7}
#SBATCH --clusters=${TARGET_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=${RUN_ROOT}/logs/${KEY}-%j.out
#SBATCH --error=${RUN_ROOT}/logs/${KEY}-%j.err
source /etc/profile
set -euo pipefail
module load cuda/11.8
module load miniforge3/24.11
source '${BASE}/conda_envs/opentad/bin/activate'
cd '${ROOT}'
export PYTHONNOUSERSITE=1
export BASE='${BASE}'
export DUCA_EXPECTED_COMMIT='${EXPECTED_COMMIT}'
export DUCA_TRUETIME_ROUTE_ARM='${ARM}'
export DUCA_PROTECTED_PROTOCOL_MANIFEST_JSON='${PROTOCOL}'
export DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256='${PROTOCOL_SHA}'
export DUCA_PROTECTED_AUTHORIZATION_JSON='${AUTH}'
export DUCA_PROTECTED_AUTHORIZATION_SHA256='${AUTH_SHA}'
export RUN_DIR='${RUN_ROOT}/${KEY}/run'
export WORK_DIR='${RUN_ROOT}/${KEY}/work'
bash scripts/run_duca_truetime_curriculum_official60_gpu1.sh
EOF
  chmod 0755 "${JOB_FILE}"
  bash -n "${JOB_FILE}"
  if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
    continue
  fi
  RESPONSE="$(sbatch --hold --parsable --clusters="${TARGET_CLUSTER}" "${JOB_FILE}")"
  JOB_ID="${RESPONSE%%;*}"
  [[ "${JOB_ID}" =~ ^[1-9][0-9]*$ ]] || fail "invalid sbatch response ${RESPONSE}"
  IDS+=("${JOB_ID}")
done

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  echo "DUCA_TRUETIME_PAIR_SUBMIT_PRECHECK_PASS"
  exit 0
fi

{
  echo -e "route_arm\tjob_id\tdependency"
  for IDX in "${!ARMS[@]}"; do
    echo -e "${ARMS[$IDX]}\t${IDS[$IDX]}\tnone"
  done
} > "${RUN_ROOT}/jobs.tsv"
for JOB_ID in "${IDS[@]}"; do
  scontrol --clusters="${TARGET_CLUSTER}" release "${JOB_ID}"
done
cat "${RUN_ROOT}/jobs.tsv"
