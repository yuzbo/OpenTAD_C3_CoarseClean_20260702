#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_P0_PREPARE][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${REPO_ROOT}/scripts/duca_transition_only_p0_canonical_env.sh"
SEED="${SEED:-0}"
CURRENT_HEAD="$(git rev-parse HEAD 2>/dev/null)" || fail "cannot resolve current HEAD"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-${CURRENT_HEAD}}"
RUN_ROOT="${RUN_ROOT:-${BASE}/projects/c3_lowres_action_probe/duca_p0_matched_${CURRENT_HEAD:0:7}_seed${SEED}}"
CORE_GATE_JSON="${DUCA_CORE_GATE_JSON:-}"
DDP_PILOT_JSON="${DUCA_DDP_PILOT_JSON:-}"

[[ -x "${PYTHON}" ]] || fail "Python environment missing: ${PYTHON}"
[[ "${CURRENT_HEAD}" == "${EXPECTED_COMMIT}" ]] || fail "HEAD differs from DUCA_EXPECTED_COMMIT"
[[ -n "${CORE_GATE_JSON}" && -f "${CORE_GATE_JSON}" ]] || fail "DUCA_CORE_GATE_JSON must name an existing formal gate"
[[ -n "${DDP_PILOT_JSON}" && -f "${DDP_PILOT_JSON}" ]] || fail "DUCA_DDP_PILOT_JSON must name an existing four-arm pilot"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "formal suite preparation requires a clean git tree"
[[ ! -e "${RUN_ROOT}" ]] || fail "RUN_ROOT already exists: ${RUN_ROOT}"

mkdir -p "${RUN_ROOT}/jobs" "${RUN_ROOT}/logs"
CANONICAL_ENV_FILE="${RUN_ROOT}/canonical_env.tsv"
duca_p0_canonical_env_payload > "${CANONICAL_ENV_FILE}"
CANONICAL_ENV_SHA256="$(sha256sum "${CANONICAL_ENV_FILE}" | awk '{print $1}')"
MANIFEST="${RUN_ROOT}/suite_manifest.json"
"${PYTHON}" -m tools.bata.validate_duca_transition_only_p0_suite \
  --repo-root "${REPO_ROOT}" \
  --seed "${SEED}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --require-clean \
  --core-gate-json "${CORE_GATE_JSON}" \
  --ddp-pilot-json "${DDP_PILOT_JSON}" \
  --require-ddp-pilot \
  --output-json "${MANIFEST}"

variants=(uniform direct transition_beta0 transition_counterfactual)
readarray -t data_binding < <("${PYTHON}" -c "import json; p=json.load(open('${MANIFEST}', encoding='utf-8'))['reference_data_artifacts']; print(p['evaluation_annotation_path']); print(p['evaluation_annotation_sha256']); print(p['evaluation_class_map_path']); print(p['evaluation_class_map_sha256']); print(p['evaluation_config_sha256'])")
evaluation_annotation_path="${data_binding[0]}"
evaluation_annotation_sha256="${data_binding[1]}"
evaluation_class_map_path="${data_binding[2]}"
evaluation_class_map_sha256="${data_binding[3]}"
evaluation_config_sha256="${data_binding[4]}"
for index in "${!variants[@]}"; do
  variant="${variants[$index]}"
  readarray -t binding < <("${PYTHON}" -c "import json; p=json.load(open('${MANIFEST}', encoding='utf-8')); v=next(x for x in p['variants'] if x['name']=='${variant}'); print(v['resolved_config_sha256']); print(v['variant_contract_sha256']); print(p['shared_protocol_sha256'])")
  resolved_config_sha256="${binding[0]}"
  variant_contract_sha256="${binding[1]}"
  shared_protocol_sha256="${binding[2]}"
  job_file="${RUN_ROOT}/jobs/${variant}.sbatch"
  cat > "${job_file}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=duca-p0-${variant}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=${RUN_ROOT}/logs/${variant}-%j.out
#SBATCH --error=${RUN_ROOT}/logs/${variant}-%j.err
set -euo pipefail
cd '${REPO_ROOT}'
export BASE='${BASE}'
export PYTHON='${PYTHON}'
export ADATAD_PRETRAIN_PATH='${ADATAD_PRETRAIN_PATH}'
export DUCA_P0_VARIANT='${variant}'
export DUCA_EXPECTED_COMMIT='${EXPECTED_COMMIT}'
export DUCA_CORE_GATE_JSON='${CORE_GATE_JSON}'
export DUCA_DDP_PILOT_JSON='${DDP_PILOT_JSON}'
export DUCA_CANONICAL_ENV_FILE='${CANONICAL_ENV_FILE}'
export DUCA_CANONICAL_ENV_SHA256='${CANONICAL_ENV_SHA256}'
export DUCA_RESOLVED_CONFIG_SHA256='${resolved_config_sha256}'
export DUCA_VARIANT_CONTRACT_SHA256='${variant_contract_sha256}'
export DUCA_SHARED_PROTOCOL_SHA256='${shared_protocol_sha256}'
export DUCA_EVALUATION_ANNOTATION_PATH='${evaluation_annotation_path}'
export DUCA_EVALUATION_ANNOTATION_SHA256='${evaluation_annotation_sha256}'
export DUCA_EVALUATION_CLASS_MAP_PATH='${evaluation_class_map_path}'
export DUCA_EVALUATION_CLASS_MAP_SHA256='${evaluation_class_map_sha256}'
export DUCA_EVALUATION_CONFIG_SHA256='${evaluation_config_sha256}'
export FULLTRAIN_CANDIDATE=1
export SEED='${SEED}'
export RUN_ID='${index}'
export RUN_DIR='${RUN_ROOT}/logs/${variant}'
export WORK_DIR='${RUN_ROOT}/work_dirs/${variant}'
bash scripts/run_duca_transition_only_p0_variant_gpu1.sh
EOF
  chmod 0755 "${job_file}"
done

printf 'variant\tseed\tcommit\tsbatch_file\tstatus\n' > "${RUN_ROOT}/jobs.tsv"
for variant in "${variants[@]}"; do
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${variant}" "${SEED}" "${EXPECTED_COMMIT}" \
    "${RUN_ROOT}/jobs/${variant}.sbatch" "PREPARED_NOT_SUBMITTED" >> "${RUN_ROOT}/jobs.tsv"
done

for job_file in "${RUN_ROOT}"/jobs/*.sbatch; do
  bash -n "${job_file}" || fail "generated job file has invalid syntax: ${job_file}"
done

echo "[DUCA_P0_PREPARE] prepared four matched jobs under ${RUN_ROOT}"
echo "[DUCA_P0_PREPARE] no Slurm jobs were submitted"
