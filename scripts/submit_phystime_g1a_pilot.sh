#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PhysTime G1a submit] ERROR: $*" >&2
  exit 1
}

WORK_DIR="${PHYSTIME_WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${WORK_DIR}"
BASE="${PHYSTIME_BASE:-/data/run01/sczc063/yuzibo}"
COMMIT="$(git rev-parse HEAD)"
TREE="$(git rev-parse 'HEAD^{tree}')"
RUN_TAG="${PHYSTIME_RUN_TAG:-phystime_g1a_${COMMIT:0:7}_$(date +%Y%m%d_%H%M%S_%z)}"
RUN_ROOT="${PHYSTIME_RUN_ROOT:-${BASE}/projects/phystime_tad/runs/${RUN_TAG}}"
SBATCH_ROOT="${RUN_ROOT}/sbatch"
LOG_ROOT="${RUN_ROOT}/slurm_logs"
G0_JSON="${RUN_ROOT}/gate/g0_static_precheck.json"
CONTRACT_JSON="${RUN_ROOT}/gate/g1a_contract.json"
GATE_JSON="${RUN_ROOT}/gate/g1a_real_gate.json"
PARTITION="${PHYSTIME_SLURM_PARTITION:-gpu}"

: "${OPENTAD_THUMOS14_ANNOTATION:?OPENTAD_THUMOS14_ANNOTATION is required}"
: "${OPENTAD_THUMOS14_CLASS_MAP:?OPENTAD_THUMOS14_CLASS_MAP is required}"
: "${OPENTAD_THUMOS14_TRAIN_VIDEOS:?OPENTAD_THUMOS14_TRAIN_VIDEOS is required}"
: "${OPENTAD_THUMOS14_TEST_VIDEOS:?OPENTAD_THUMOS14_TEST_VIDEOS is required}"
: "${PHYSTIME_VIDEOMAE_CHECKPOINT:?PHYSTIME_VIDEOMAE_CHECKPOINT is required}"
[[ -z "$(git status --porcelain)" ]] || fail "snapshot must be completely clean, including untracked files"
[[ -f "${PHYSTIME_VIDEOMAE_CHECKPOINT}" ]] || fail "checkpoint not found"
[[ "${PHYSTIME_G1A_PILOT_EPOCHS:-6}" == "6" ]] || fail "formal G1a pilots require exactly six epochs"
[[ ! -e "${RUN_ROOT}" ]] || fail "run root already exists: ${RUN_ROOT}"
mkdir -p "${SBATCH_ROOT}" "${LOG_ROOT}" "${RUN_ROOT}/gate"

submit() {
  local output attempt
  for attempt in $(seq 1 "${PHYSTIME_SUBMIT_RETRIES:-12}"); do
    if output="$(sbatch --parsable "$@" 2>&1)"; then
      printf '%s\n' "${output%%;*}"
      return 0
    fi
    echo "[PhysTime G1a submit] sbatch attempt ${attempt} failed: ${output}" >&2
    sleep "${PHYSTIME_SUBMIT_RETRY_DELAY_SEC:-20}"
  done
  return 1
}

write_header() {
  local path="$1" name="$2" time_limit="$3"
  {
    echo '#!/usr/bin/env bash'
    echo "#SBATCH --job-name=${name}"
    echo "#SBATCH --partition=${PARTITION}"
    echo '#SBATCH --gres=gpu:1'
    echo '#SBATCH --cpus-per-task=6'
    echo "#SBATCH --time=${time_limit}"
    echo "#SBATCH --output=${LOG_ROOT}/${name}_%j.out"
    echo "#SBATCH --error=${LOG_ROOT}/${name}_%j.err"
    echo 'set -euo pipefail'
    printf 'cd %q\n' "${WORK_DIR}"
    printf 'export PHYSTIME_BASE=%q\n' "${BASE}"
    printf 'export PHYSTIME_WORK_DIR=%q\n' "${WORK_DIR}"
    printf 'export OPENTAD_THUMOS14_ANNOTATION=%q\n' "${OPENTAD_THUMOS14_ANNOTATION}"
    printf 'export OPENTAD_THUMOS14_CLASS_MAP=%q\n' "${OPENTAD_THUMOS14_CLASS_MAP}"
    printf 'export OPENTAD_THUMOS14_TRAIN_VIDEOS=%q\n' "${OPENTAD_THUMOS14_TRAIN_VIDEOS}"
    printf 'export OPENTAD_THUMOS14_TEST_VIDEOS=%q\n' "${OPENTAD_THUMOS14_TEST_VIDEOS}"
    printf 'export PHYSTIME_VIDEOMAE_CHECKPOINT=%q\n' "${PHYSTIME_VIDEOMAE_CHECKPOINT}"
    printf 'export PHYSTIME_EXPECTED_COMMIT=%q\n' "${COMMIT}"
    printf 'export PHYSTIME_EXPECTED_TREE=%q\n' "${TREE}"
    printf 'export HOME=%q\n' "${BASE}/tmp/home"
    printf 'export XDG_CACHE_HOME=%q\n' "${BASE}/tmp/xdg_cache"
    printf 'export XDG_CONFIG_HOME=%q\n' "${BASE}/tmp/xdg_config"
    printf 'export HF_HOME=%q\n' "${BASE}/hf_cache"
  } > "${path}"
}

gate_sbatch="${SBATCH_ROOT}/g1a_gate.sbatch"
write_header "${gate_sbatch}" pt_g1a_gate "${PHYSTIME_G1A_GATE_TIME:-02:00:00}"
{
  printf 'export PHYSTIME_G1A_CONTRACT_JSON=%q\n' "${CONTRACT_JSON}"
  printf 'export PHYSTIME_G0_OUTPUT=%q\n' "${G0_JSON}"
  printf 'export PHYSTIME_G1A_GATE_OUTPUT=%q\n' "${GATE_JSON}"
  echo "export PHYSTIME_SEED='42'"
  echo 'bash scripts/run_phystime_g1a_gate_slurm.sh'
} >> "${gate_sbatch}"
gate_job="$(submit "${gate_sbatch}")"

printf 'variant\tjob_id\tdependency\tconfig\tK\tJ\tQ0\tQ_total\tstatus\n' > "${RUN_ROOT}/jobs.tsv"
printf 'g1a_gate\t%s\tnone\treal_thumos_gate\t384\t192\t192\t378\tsubmitted\n' "${gate_job}" >> "${RUN_ROOT}/jobs.tsv"

declare -A jobs
for spec in \
  "selected_axis|${WORK_DIR}/configs/adatad/thumos/phystime_g1a_selected_axis_native_j192.py" \
  "physical_metric|${WORK_DIR}/configs/adatad/thumos/phystime_g1a_physical_metric_native_j192.py"; do
  IFS='|' read -r variant config <<< "${spec}"
  sbatch_path="${SBATCH_ROOT}/${variant}.sbatch"
  run_dir="${RUN_ROOT}/${variant}"
  write_header "${sbatch_path}" "pt_g1a_${variant}" "${PHYSTIME_G1A_PILOT_TIME:-12:00:00}"
  {
    printf 'export PHYSTIME_G1A_CONFIG=%q\n' "${config}"
    printf 'export PHYSTIME_G1A_RUN_DIR=%q\n' "${run_dir}"
    printf 'export PHYSTIME_G1A_GATE_JSON=%q\n' "${GATE_JSON}"
    printf 'export PHYSTIME_G1A_CONTRACT_JSON=%q\n' "${CONTRACT_JSON}"
    printf 'export PHYSTIME_G0_OUTPUT=%q\n' "${G0_JSON}"
    printf 'export PHYSTIME_G1A_PILOT_EPOCHS=%q\n' "${PHYSTIME_G1A_PILOT_EPOCHS:-6}"
    echo "export PHYSTIME_SEED='42'"
    echo 'bash scripts/run_phystime_g1a_pilot_slurm.sh'
  } >> "${sbatch_path}"
  jobs["${variant}"]="$(submit --dependency="afterok:${gate_job}" "${sbatch_path}")"
  printf '%s\t%s\tafterok:%s\t%s\t384\t192\t192\t378\tsubmitted\n' \
    "${variant}" "${jobs[${variant}]}" "${gate_job}" "${config}" >> "${RUN_ROOT}/jobs.tsv"
done

cat > "${RUN_ROOT}/deployment_summary.json" <<EOF
{
  "schema_version": "phystime_g1a_pilot_deployment_v2",
  "track": "g0_native_provenance_then_g1a_matched_metric",
  "commit": "${COMMIT}",
  "git_tree": "${TREE}",
  "run_root": "${RUN_ROOT}",
  "gate_job": "${gate_job}",
  "contract_json": "${CONTRACT_JSON}",
  "g0_json": "${G0_JSON}",
  "gate_json": "${GATE_JSON}",
  "pilot_jobs": {
    "selected_axis": "${jobs[selected_axis]}",
    "physical_metric": "${jobs[physical_metric]}"
  },
  "K_raw_observations": 384,
  "J_native_tubelet_tokens": 192,
  "Q0_base_candidates": 192,
  "Q_total_candidates": 378,
  "Q0_base_candidate_tensor_slots": 192,
  "Q_total_candidate_tensor_slots": 378,
  "effective_candidate_count_policy": "semantic_anchor_prefix_reported_per_sample",
  "feature_interpolation": false,
  "g1b_g2_status": "held_until_g1a_interpretation"
}
EOF

echo "[PhysTime G1a submit] RUN_ROOT=${RUN_ROOT}"
echo "[PhysTime G1a submit] GATE_JOB=${gate_job}"
cat "${RUN_ROOT}/jobs.tsv"
