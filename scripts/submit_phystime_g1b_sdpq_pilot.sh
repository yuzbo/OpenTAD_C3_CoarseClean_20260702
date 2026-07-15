#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PhysTime G1b SDPQ submit] ERROR: $*" >&2
  exit 1
}

WORK_DIR="${PHYSTIME_WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${WORK_DIR}"
BASE="${PHYSTIME_BASE:-/data/run01/sczc063/yuzibo}"
COMMIT="$(git rev-parse HEAD)"
TREE="$(git rev-parse 'HEAD^{tree}')"
RUN_TAG="${PHYSTIME_RUN_TAG:-phystime_g1b_sdpq_${COMMIT:0:7}_$(date +%Y%m%d_%H%M%S_%z)}"
RUN_ROOT="${PHYSTIME_RUN_ROOT:-${BASE}/projects/phystime_tad/runs/${RUN_TAG}}"
SBATCH_ROOT="${RUN_ROOT}/sbatch"
LOG_ROOT="${RUN_ROOT}/slurm_logs"
GATE_JSON="${RUN_ROOT}/gate/g1b_sdpq_real_gate.json"
PARTITION="${PHYSTIME_SLURM_PARTITION:-gpu}"

: "${OPENTAD_THUMOS14_ANNOTATION:?OPENTAD_THUMOS14_ANNOTATION is required}"
: "${OPENTAD_THUMOS14_CLASS_MAP:?OPENTAD_THUMOS14_CLASS_MAP is required}"
: "${OPENTAD_THUMOS14_TRAIN_VIDEOS:?OPENTAD_THUMOS14_TRAIN_VIDEOS is required}"
: "${OPENTAD_THUMOS14_TEST_VIDEOS:?OPENTAD_THUMOS14_TEST_VIDEOS is required}"
: "${PHYSTIME_VIDEOMAE_CHECKPOINT:?PHYSTIME_VIDEOMAE_CHECKPOINT is required}"
[[ -z "$(git status --porcelain)" ]] || fail "snapshot must be clean before submission"
[[ -f "${PHYSTIME_VIDEOMAE_CHECKPOINT}" ]] || fail "checkpoint not found"
[[ ! -e "${RUN_ROOT}" ]] || fail "run root already exists: ${RUN_ROOT}"
mkdir -p "${SBATCH_ROOT}" "${LOG_ROOT}" "${RUN_ROOT}/gate"

submit() {
  local output attempt
  for attempt in $(seq 1 "${PHYSTIME_SUBMIT_RETRIES:-12}"); do
    if output="$(sbatch --parsable "$@" 2>&1)"; then
      printf '%s\n' "${output%%;*}"
      return 0
    fi
    echo "[PhysTime G1b SDPQ submit] sbatch attempt ${attempt} failed: ${output}" >&2
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

gate_sbatch="${SBATCH_ROOT}/g1b_sdpq_gate.sbatch"
write_header "${gate_sbatch}" pt_g1b_sdpq_gate "${PHYSTIME_G1B_GATE_TIME:-02:00:00}"
{
  printf 'export PHYSTIME_G1B_GATE_OUTPUT=%q\n' "${GATE_JSON}"
  echo "export PHYSTIME_SEED='42'"
  echo 'bash scripts/run_phystime_g1b_sdpq_gate_slurm.sh'
} >> "${gate_sbatch}"
gate_job="$(submit "${gate_sbatch}")"

pilot_sbatch="${SBATCH_ROOT}/g1b_sdpq_pilot.sbatch"
pilot_run_dir="${RUN_ROOT}/sdpq_pilot"
write_header "${pilot_sbatch}" pt_g1b_sdpq_pilot "${PHYSTIME_G1B_PILOT_TIME:-12:00:00}"
{
  printf 'export PHYSTIME_G1B_GATE_OUTPUT=%q\n' "${GATE_JSON}"
  printf 'export PHYSTIME_G1B_RUN_DIR=%q\n' "${pilot_run_dir}"
  printf 'export PHYSTIME_G1B_PILOT_EPOCHS=%q\n' "${PHYSTIME_G1B_PILOT_EPOCHS:-6}"
  echo "export PHYSTIME_SEED='42'"
  echo 'bash scripts/run_phystime_g1b_sdpq_pilot_slurm.sh'
} >> "${pilot_sbatch}"
pilot_job="$(submit --dependency="afterok:${gate_job}" "${pilot_sbatch}")"

printf 'variant\tjob_id\tdependency\tconfig\tK\tJ\thead\tstatus\n' > "${RUN_ROOT}/jobs.tsv"
printf 'g1b_sdpq_gate\t%s\tnone\tphystime_g1b_sdpq_pool_native_j192.py\t384\t192\tSDPQ\tsubmitted\n' "${gate_job}" >> "${RUN_ROOT}/jobs.tsv"
printf 'g1b_sdpq_pilot\t%s\tafterok:%s\tphystime_g1b_sdpq_pool_native_j192.py\t384\t192\tSDPQ\tsubmitted\n' "${pilot_job}" "${gate_job}" >> "${RUN_ROOT}/jobs.tsv"

cat > "${RUN_ROOT}/deployment_summary.json" <<EOF
{
  "schema_version": "phystime_g1b_sdpq_deployment_v1",
  "track": "support_decoupled_physical_query_sparse_head",
  "commit": "${COMMIT}",
  "git_tree": "${TREE}",
  "run_root": "${RUN_ROOT}",
  "gate_job": "${gate_job}",
  "pilot_job": "${pilot_job}",
  "gate_json": "${GATE_JSON}",
  "K_raw_observations": 384,
  "J_native_tubelet_tokens": 192,
  "feature_interpolation": false,
  "head": "SupportDecoupledPhysicalQueryHead",
  "full_train_status": "held_until_gate_and_pilot"
}
EOF

echo "[PhysTime G1b SDPQ submit] RUN_ROOT=${RUN_ROOT}"
cat "${RUN_ROOT}/jobs.tsv"
