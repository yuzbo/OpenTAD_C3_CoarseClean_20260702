#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_CELLCF_PREPARE][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"
SEED="${SEED:-0}"
CURRENT_HEAD="$(git rev-parse HEAD)"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-${CURRENT_HEAD}}"
RUN_ROOT="${RUN_ROOT:-${BASE}/projects/c3_lowres_action_probe/duca_cellcf_${CURRENT_HEAD:0:7}_seed${SEED}}"
GATE_JSON="${DUCA_CELLCF_GATE_JSON:-}"
PILOT_JSON="${DUCA_CELLCF_DDP_PILOT_JSON:-}"

[[ "${CURRENT_HEAD}" == "${EXPECTED_COMMIT}" ]] || fail "HEAD differs from expected commit"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "suite preparation requires a clean tree"
[[ -x "${PYTHON}" ]] || fail "Python is missing: ${PYTHON}"
[[ -f "${GATE_JSON}" ]] || fail "real-loader gate JSON is missing"
[[ -f "${PILOT_JSON}" ]] || fail "DDP pilot JSON is missing"
[[ ! -e "${RUN_ROOT}" ]] || fail "RUN_ROOT already exists: ${RUN_ROOT}"

mkdir -p "${RUN_ROOT}/jobs" "${RUN_ROOT}/logs" "${RUN_ROOT}/work_dirs"
CANONICAL_ENV_FILE="${RUN_ROOT}/canonical_env.tsv"
duca_cellcf_canonical_env_payload > "${CANONICAL_ENV_FILE}"
CANONICAL_ENV_SHA256="$(sha256sum "${CANONICAL_ENV_FILE}" | awk '{print $1}')"
MANIFEST="${RUN_ROOT}/suite_manifest.json"
"${PYTHON}" -m tools.bata.validate_duca_cellcf_suite \
  --repo-root "${REPO_ROOT}" --seed "${SEED}" \
  --expected-commit "${EXPECTED_COMMIT}" --require-clean \
  --gate-json "${GATE_JSON}" --pilot-json "${PILOT_JSON}" \
  --output-json "${MANIFEST}"
GATE_SHA256="$(sha256sum "${GATE_JSON}" | awk '{print $1}')"
PILOT_SHA256="$(sha256sum "${PILOT_JSON}" | awk '{print $1}')"

variants=(uniform transition_beta0 cellcf)
for variant in "${variants[@]}"; do
  readarray -t binding < <("${PYTHON}" - "${MANIFEST}" "${variant}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
variant = next(item for item in payload["variants"] if item["name"] == sys.argv[2])
data = payload["reference_data_artifacts"]
for value in (
    variant["resolved_config_sha256"],
    payload["shared_protocol_sha256"],
    payload["ordered_exposure_sha256"],
    data["evaluation_annotation_path"],
    data["evaluation_annotation_sha256"],
    data["evaluation_class_map_path"],
    data["evaluation_class_map_sha256"],
    data["evaluation_config_sha256"],
):
    print(value)
PY
)
  job_file="${RUN_ROOT}/jobs/${variant}.sbatch"
  cat > "${job_file}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=cellcf-${variant}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=${RUN_ROOT}/logs/${variant}-%j.out
#SBATCH --error=${RUN_ROOT}/logs/${variant}-%j.err
set -euo pipefail
cd '${REPO_ROOT}'
export BASE='${BASE}'
export DUCA_CELLCF_VARIANT='${variant}'
export DUCA_EXPECTED_COMMIT='${EXPECTED_COMMIT}'
export DUCA_CELLCF_GATE_JSON='${GATE_JSON}'
export DUCA_CELLCF_DDP_PILOT_JSON='${PILOT_JSON}'
export DUCA_CELLCF_GATE_SHA256='${GATE_SHA256}'
export DUCA_CELLCF_DDP_PILOT_SHA256='${PILOT_SHA256}'
export DUCA_CELLCF_RESOLVED_CONFIG_SHA256='${binding[0]}'
export DUCA_CELLCF_PROTOCOL_SHA256='${binding[1]}'
export DUCA_CELLCF_ORDER_SHA256='${binding[2]}'
export DUCA_CELLCF_ANNOTATION_PATH='${binding[3]}'
export DUCA_CELLCF_ANNOTATION_SHA256='${binding[4]}'
export DUCA_CELLCF_CLASS_MAP_PATH='${binding[5]}'
export DUCA_CELLCF_CLASS_MAP_SHA256='${binding[6]}'
export DUCA_CELLCF_EVALUATION_CONFIG_SHA256='${binding[7]}'
export DUCA_CELLCF_CANONICAL_ENV_FILE='${CANONICAL_ENV_FILE}'
export DUCA_CELLCF_CANONICAL_ENV_SHA256='${CANONICAL_ENV_SHA256}'
export SEED='${SEED}'
export RUN_DIR='${RUN_ROOT}/logs/${variant}'
export WORK_DIR='${RUN_ROOT}/work_dirs/${variant}'
bash scripts/run_duca_cellcf_variant.sh
EOF
  chmod 0755 "${job_file}"
  bash -n "${job_file}"
done

printf 'variant\tseed\tcommit\tsbatch_file\tstatus\n' > "${RUN_ROOT}/jobs.tsv"
for variant in "${variants[@]}"; do
  printf '%s\t%s\t%s\t%s\tPREPARED_NOT_SUBMITTED\n' \
    "${variant}" "${SEED}" "${EXPECTED_COMMIT}" "${RUN_ROOT}/jobs/${variant}.sbatch" \
    >> "${RUN_ROOT}/jobs.tsv"
done
echo "[DUCA_CELLCF_PREPARE] prepared ${RUN_ROOT}; no job submitted"
