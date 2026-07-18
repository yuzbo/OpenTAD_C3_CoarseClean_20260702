#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PhysTime G1 matched full60 submit] ERROR: $*" >&2
  exit 1
}

WORK_DIR="${PHYSTIME_WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${WORK_DIR}"
BASE="${PHYSTIME_BASE:-/data/run01/sczc063/yuzibo}"
COMMIT="$(git rev-parse HEAD)"
TREE="$(git rev-parse 'HEAD^{tree}')"
RUN_TAG="${PHYSTIME_RUN_TAG:-phystime_g1_matched_full60_${COMMIT:0:7}_$(date +%Y%m%d_%H%M%S_%z)}"
RUN_ROOT="${PHYSTIME_RUN_ROOT:-${BASE}/projects/phystime_tad/runs/${RUN_TAG}}"
SBATCH_ROOT="${RUN_ROOT}/sbatch"
LOG_ROOT="${RUN_ROOT}/slurm_logs"
GATE_ROOT="${RUN_ROOT}/gate"
G0_JSON="${GATE_ROOT}/g0_static_precheck.json"
CONTRACT_JSON="${GATE_ROOT}/g1a_contract.json"
G1A_GATE_JSON="${GATE_ROOT}/g1a_real_gate.json"
PARTITION="${PHYSTIME_SLURM_PARTITION:-gpu}"
EPOCHS="${PHYSTIME_FULL_EPOCHS:-60}"
SEED="${PHYSTIME_SEED:-42}"
MIN_FREE_KB="${PHYSTIME_MIN_FREE_KB:-8388608}"

: "${OPENTAD_THUMOS14_ANNOTATION:?OPENTAD_THUMOS14_ANNOTATION is required}"
: "${OPENTAD_THUMOS14_CLASS_MAP:?OPENTAD_THUMOS14_CLASS_MAP is required}"
: "${OPENTAD_THUMOS14_TRAIN_VIDEOS:?OPENTAD_THUMOS14_TRAIN_VIDEOS is required}"
: "${OPENTAD_THUMOS14_TEST_VIDEOS:?OPENTAD_THUMOS14_TEST_VIDEOS is required}"
: "${PHYSTIME_VIDEOMAE_CHECKPOINT:?PHYSTIME_VIDEOMAE_CHECKPOINT is required}"
[[ "${EPOCHS}" == "60" ]] || fail "matched full60 suite requires exactly 60 epochs"
[[ "${SEED}" == "42" ]] || fail "matched full60 suite requires seed 42"
[[ "${MIN_FREE_KB}" =~ ^[0-9]+$ ]] || fail "PHYSTIME_MIN_FREE_KB must be an integer"
[[ -z "$(git status --porcelain)" ]] || fail "snapshot must be clean before submission"
[[ -f "${PHYSTIME_VIDEOMAE_CHECKPOINT}" ]] || fail "checkpoint not found"
[[ ! -e "${RUN_ROOT}" ]] || fail "run root already exists: ${RUN_ROOT}"
FREE_KB="$(df -Pk "${BASE}" | awk 'END {print $4}')"
[[ "${FREE_KB}" =~ ^[0-9]+$ ]] || fail "cannot determine free space under ${BASE}"
(( FREE_KB >= MIN_FREE_KB )) \
  || fail "insufficient free space: ${FREE_KB} KiB available, ${MIN_FREE_KB} KiB required"
mkdir -p "${SBATCH_ROOT}" "${LOG_ROOT}" "${GATE_ROOT}"

submit() {
  local output attempt
  for attempt in $(seq 1 "${PHYSTIME_SUBMIT_RETRIES:-12}"); do
    if output="$(sbatch --parsable "$@" 2>&1)"; then
      printf '%s\n' "${output%%;*}"
      return 0
    fi
    echo "[PhysTime G1 matched full60 submit] sbatch attempt ${attempt} failed: ${output}" >&2
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
  chmod +x "${path}"
}

gate_sbatch="${SBATCH_ROOT}/matched_full60_gate.sbatch"
write_header "${gate_sbatch}" pt_g1_full_gate "${PHYSTIME_MATCHED_GATE_TIME:-04:00:00}"
{
  printf 'export PHYSTIME_G1A_CONTRACT_JSON=%q\n' "${CONTRACT_JSON}"
  printf 'export PHYSTIME_G0_OUTPUT=%q\n' "${G0_JSON}"
  printf 'export PHYSTIME_G1A_GATE_OUTPUT=%q\n' "${G1A_GATE_JSON}"
  printf 'export PHYSTIME_SEED=%q\n' "${SEED}"
  echo 'bash scripts/run_phystime_g1a_gate_slurm.sh'
} >> "${gate_sbatch}"
gate_job="$(submit "${gate_sbatch}")"

printf 'variant\tjob_id\tdependency\tconfig\tK\tJ\tepochs\tstatus\n' > "${RUN_ROOT}/jobs.tsv"
printf 'matched_gate\t%s\tnone\tg1a_real_gate\t384\t192\tNA\tsubmitted\n' "${gate_job}" \
  >> "${RUN_ROOT}/jobs.tsv"

declare -A jobs
for spec in \
  "selected_axis|configs/adatad/thumos/phystime_g1a_selected_axis_native_j192.py" \
  "physical_metric|configs/adatad/thumos/phystime_g1a_physical_metric_native_j192.py"; do
  IFS='|' read -r variant config <<< "${spec}"
  sbatch_path="${SBATCH_ROOT}/${variant}.sbatch"
  run_dir="${RUN_ROOT}/${variant}"
  write_header "${sbatch_path}" "pt_g1_${variant}_f60" "${PHYSTIME_FULL_TIME:-36:00:00}"
  {
    printf 'export PHYSTIME_FULL_VARIANT=%q\n' "${variant}"
    printf 'export PHYSTIME_FULL_CONFIG=%q\n' "${WORK_DIR}/${config}"
    printf 'export PHYSTIME_FULL_RUN_DIR=%q\n' "${run_dir}"
    printf 'export PHYSTIME_G1A_GATE_OUTPUT=%q\n' "${G1A_GATE_JSON}"
    printf 'export PHYSTIME_FULL_EPOCHS=%q\n' "${EPOCHS}"
    printf 'export PHYSTIME_SEED=%q\n' "${SEED}"
    echo 'bash scripts/run_phystime_g1_matched_full60_slurm.sh'
  } >> "${sbatch_path}"
  jobs["${variant}"]="$(submit --dependency="afterok:${gate_job}" "${sbatch_path}")"
  printf '%s\t%s\tafterok:%s\t%s\t384\t192\t%s\tsubmitted\n' \
    "${variant}" "${jobs[${variant}]}" "${gate_job}" "${config}" "${EPOCHS}" \
    >> "${RUN_ROOT}/jobs.tsv"
done

cat > "${RUN_ROOT}/deployment_summary.json" <<EOF
{
  "schema_version": "phystime_g1_matched_full60_deployment_v1",
  "track": "native_j192_matched_full60_two_arm",
  "commit": "${COMMIT}",
  "git_tree": "${TREE}",
  "run_root": "${RUN_ROOT}",
  "gate_job": "${gate_job}",
  "gate_artifacts": {
    "g1a_contract": "${CONTRACT_JSON}",
    "g0_static_precheck": "${G0_JSON}",
    "g1a_real_gate": "${G1A_GATE_JSON}"
  },
  "jobs": {
    "selected_axis": "${jobs[selected_axis]}",
    "physical_metric": "${jobs[physical_metric]}"
  },
  "K_raw_observations": 384,
  "J_native_tubelet_tokens": 192,
  "epochs": ${EPOCHS},
  "seed": ${SEED},
  "feature_interpolation": false,
  "checkpoint_save_mode": "lightweight_final_only_with_ema",
  "submission_free_space_kib": ${FREE_KB},
  "minimum_free_space_kib": ${MIN_FREE_KB},
  "validation_start_epoch": 40,
  "validation_interval": 2,
  "scheduler_max_epoch": ${EPOCHS},
  "full_train_status": "submitted_after_matched_medium_survivor"
}
EOF

echo "[PhysTime G1 matched full60 submit] RUN_ROOT=${RUN_ROOT}"
cat "${RUN_ROOT}/jobs.tsv"
