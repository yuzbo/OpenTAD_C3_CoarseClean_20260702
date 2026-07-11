#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PhysTime-AdaTAD submit] ERROR: $*" >&2
  exit 1
}

WORK_DIR="${PHYSTIME_WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${WORK_DIR}"
BASE="${PHYSTIME_BASE:-/data/run01/sczc063/yuzibo}"
COMMIT="$(git rev-parse HEAD)"
RUN_TAG="${PHYSTIME_RUN_TAG:-phystime_adatad_${COMMIT:0:7}_$(date +%Y%m%d_%H%M%S_%z)}"
RUN_ROOT="${PHYSTIME_RUN_ROOT:-${BASE}/projects/phystime_tad/runs/${RUN_TAG}}"
SBATCH_ROOT="${RUN_ROOT}/sbatch"
LOG_ROOT="${RUN_ROOT}/slurm_logs"
GATE_JSON="${RUN_ROOT}/real_gate/real_gate.json"
PARTITION="${PHYSTIME_SLURM_PARTITION:-gpu}"

: "${OPENTAD_THUMOS14_ANNOTATION:?OPENTAD_THUMOS14_ANNOTATION is required}"
: "${OPENTAD_THUMOS14_CLASS_MAP:?OPENTAD_THUMOS14_CLASS_MAP is required}"
: "${OPENTAD_THUMOS14_TRAIN_VIDEOS:?OPENTAD_THUMOS14_TRAIN_VIDEOS is required}"
: "${OPENTAD_THUMOS14_TEST_VIDEOS:?OPENTAD_THUMOS14_TEST_VIDEOS is required}"
: "${PHYSTIME_VIDEOMAE_CHECKPOINT:?PHYSTIME_VIDEOMAE_CHECKPOINT is required}"

git diff --quiet || fail "snapshot has unstaged tracked changes"
git diff --cached --quiet || fail "snapshot has staged changes"
[[ -f "${OPENTAD_THUMOS14_ANNOTATION}" ]] || fail "annotation file not found"
[[ -f "${OPENTAD_THUMOS14_CLASS_MAP}" ]] || fail "class map not found"
[[ -d "${OPENTAD_THUMOS14_TRAIN_VIDEOS}" ]] || fail "training videos not found"
[[ -d "${OPENTAD_THUMOS14_TEST_VIDEOS}" ]] || fail "test videos not found"
[[ -f "${PHYSTIME_VIDEOMAE_CHECKPOINT}" ]] || fail "VideoMAE-S checkpoint not found"
mkdir -p "${SBATCH_ROOT}" "${LOG_ROOT}" "${RUN_ROOT}/real_gate"

submit() {
  local output attempt
  for attempt in $(seq 1 "${PHYSTIME_SUBMIT_RETRIES:-12}"); do
    if output="$(sbatch --parsable "$@" 2>&1)"; then
      printf '%s\n' "${output%%;*}"
      return 0
    fi
    echo "[PhysTime-AdaTAD submit] sbatch attempt ${attempt} failed: ${output}" >&2
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
    printf 'export HOME=%q\n' "${BASE}/tmp/home"
    printf 'export XDG_CACHE_HOME=%q\n' "${BASE}/tmp/xdg_cache"
    printf 'export XDG_CONFIG_HOME=%q\n' "${BASE}/tmp/xdg_config"
    printf 'export HF_HOME=%q\n' "${BASE}/hf_cache"
  } > "${path}"
}

variants=(
  'selected_axis|configs/adatad/thumos/selected_axis_adatad_sparse_k384.py'
  'physical_grid|configs/adatad/thumos/physical_grid_adatad_sparse_k384.py'
  'phystime|configs/adatad/thumos/phystime_adatad_sparse_k384.py'
)

gate_sbatch="${SBATCH_ROOT}/real_gate.sbatch"
write_header "${gate_sbatch}" phystime_raw_gate "${PHYSTIME_GATE_TIME:-02:00:00}"
{
  printf 'export PHYSTIME_GATE_OUTPUT=%q\n' "${GATE_JSON}"
  echo "export PHYSTIME_SEED='42'"
  echo 'bash scripts/run_phystime_adatad_gate_gpu1.sh'
} >> "${gate_sbatch}"
gate_job="$(submit "${gate_sbatch}")"

printf 'variant\tjob_id\tdependency\tconfig\tlogical_window\tdecoded_frames\tseed\tstatus\n' > "${RUN_ROOT}/jobs.tsv"
printf 'real_gate\t%s\tnone\treal_raw_video_gate\t768\t384\t42\tsubmitted\n' "${gate_job}" >> "${RUN_ROOT}/jobs.tsv"

stability_config="${variants[2]#*|}"
stability_sbatch="${SBATCH_ROOT}/phystime_stability.sbatch"
stability_run_dir="${RUN_ROOT}/phystime_stability_gate"
write_header "${stability_sbatch}" phystime_stability "${PHYSTIME_STABILITY_TIME:-02:00:00}"
{
  printf 'export PHYSTIME_CONFIG=%q\n' "${stability_config}"
  printf 'export PHYSTIME_RUN_DIR=%q\n' "${stability_run_dir}"
  printf 'export PHYSTIME_REAL_GATE_JSON=%q\n' "${GATE_JSON}"
  echo "export PHYSTIME_SEED='42'"
  echo "export PHYSTIME_STABILITY_GATE='1'"
  echo 'bash scripts/run_phystime_adatad_full_train_gpu1.sh'
} >> "${stability_sbatch}"
stability_job="$(submit --dependency="afterok:${gate_job}" "${stability_sbatch}")"
printf 'phystime_stability\t%s\tafterok:%s\t%s\t768\t384\t42\tsubmitted\n' \
  "${stability_job}" "${gate_job}" "${stability_config}" >> "${RUN_ROOT}/jobs.tsv"

declare -A formal_jobs
for spec in "${variants[@]}"; do
  IFS='|' read -r variant config <<< "${spec}"
  sbatch_path="${SBATCH_ROOT}/${variant}.sbatch"
  run_dir="${RUN_ROOT}/${variant}"
  write_header "${sbatch_path}" "pt_${variant}" "${PHYSTIME_TRAIN_TIME:-72:00:00}"
  {
    printf 'export PHYSTIME_CONFIG=%q\n' "${config}"
    printf 'export PHYSTIME_RUN_DIR=%q\n' "${run_dir}"
    printf 'export PHYSTIME_REAL_GATE_JSON=%q\n' "${GATE_JSON}"
    echo "export PHYSTIME_SEED='42'"
    echo 'bash scripts/run_phystime_adatad_full_train_gpu1.sh'
  } >> "${sbatch_path}"
  job_id="$(submit --dependency="afterok:${stability_job}" "${sbatch_path}")"
  formal_jobs["${variant}"]="${job_id}"
  printf '%s\t%s\tafterok:%s\t%s\t768\t384\t42\tsubmitted\n' \
    "${variant}" "${job_id}" "${stability_job}" "${config}" >> "${RUN_ROOT}/jobs.tsv"
done

CHECKPOINT_SHA256="$(sha256sum "${PHYSTIME_VIDEOMAE_CHECKPOINT}" | awk '{print $1}')"
ANNOTATION_SHA256="$(sha256sum "${OPENTAD_THUMOS14_ANNOTATION}" | awk '{print $1}')"
cat > "${RUN_ROOT}/deployment_summary.json" <<EOF
{
  "schema_version": "phystime_adatad_deployment_v2",
  "track": "raw_video_k384_matched_head_comparison",
  "commit": "${COMMIT}",
  "run_root": "${RUN_ROOT}",
  "gate_job": "${gate_job}",
  "gate_json": "${GATE_JSON}",
  "stability_job": "${stability_job}",
  "stability_epochs": 2,
  "formal_job_count": 3,
  "formal_jobs": {
    "selected_axis": "${formal_jobs[selected_axis]}",
    "physical_grid": "${formal_jobs[physical_grid]}",
    "phystime": "${formal_jobs[phystime]}"
  },
  "logical_window": 768,
  "decoded_frame_budget": 384,
  "sampling": "deterministic_random_fixed_subsample",
  "seed": 42,
  "checkpoint_sha256": "${CHECKPOINT_SHA256}",
  "annotation_sha256": "${ANNOTATION_SHA256}",
  "phase2_status": "held"
}
EOF

echo "[PhysTime-AdaTAD submit] RUN_ROOT=${RUN_ROOT}"
echo "[PhysTime-AdaTAD submit] GATE_JOB=${gate_job}"
echo "[PhysTime-AdaTAD submit] STABILITY_JOB=${stability_job}"
cat "${RUN_ROOT}/jobs.tsv"
