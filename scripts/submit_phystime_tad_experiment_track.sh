#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PHYSTIME_TRACK][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
PYTHON="${PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
COMMIT="$(git rev-parse HEAD)"
RUN_TAG="${PHYSTIME_RUN_TAG:-phystime_${COMMIT:0:7}_$(date +%Y%m%d_%H%M%S_%z)}"
RUN_ROOT="${PHYSTIME_RUN_ROOT:-${BASE}/projects/phystime_tad/runs/${RUN_TAG}}"
LOG_ROOT="${RUN_ROOT}/slurm_logs"
SBATCH_ROOT="${RUN_ROOT}/sbatch"
DATA_ROOT="${PHYSTIME_THUMOS_ROOT:-${BASE}/datasets/phystime_thumos_i3d}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${SBATCH_ROOT}"

write_job() {
  local path="$1"
  local name="$2"
  local gpu="$3"
  local body="$4"
  {
    echo '#!/usr/bin/env bash'
    echo "#SBATCH --job-name=${name}"
    echo '#SBATCH --partition=gpu'
    [[ "${gpu}" == "1" ]] && echo '#SBATCH --gres=gpu:1'
    echo '#SBATCH --cpus-per-task=4'
    echo "#SBATCH --output=${LOG_ROOT}/${name}_%j.out"
    echo "#SBATCH --error=${LOG_ROOT}/${name}_%j.err"
    echo 'set -euo pipefail'
    echo "cd '${REPO_ROOT}'"
    echo 'module load cuda/11.8 >/dev/null 2>&1 || true'
    echo 'module load miniforge3/24.11 >/dev/null 2>&1 || true'
    echo "export BASE='${BASE}'"
    echo "export PYTHON='${PYTHON}'"
    echo "export PHYSTIME_REPO_ROOT='${REPO_ROOT}'"
    echo "export PHYSTIME_THUMOS_ROOT='${DATA_ROOT}'"
    if [[ -n "${PHYSTIME_DOWNLOAD_PROXY:-}" ]]; then
      printf 'export PHYSTIME_DOWNLOAD_PROXY=%q\n' "${PHYSTIME_DOWNLOAD_PROXY}"
    fi
    printf '%s\n' "${body}"
  } > "${path}"
  chmod +x "${path}"
}

submit() {
  local output attempt
  local retries="${PHYSTIME_SUBMIT_RETRIES:-12}"
  local delay="${PHYSTIME_SUBMIT_RETRY_DELAY_SEC:-30}"
  for ((attempt = 1; attempt <= retries; attempt++)); do
    if output="$(sbatch --parsable "$@" 2>&1)"; then
      echo "${output%%;*}"
      return 0
    fi
    echo "[PHYSTIME_TRACK] sbatch attempt ${attempt}/${retries} failed: ${output}" >&2
    if (( attempt < retries )); then
      sleep "${delay}"
    fi
  done
  fail "sbatch failed after ${retries} attempts: $*"
}

data_script="${SBATCH_ROOT}/data.sbatch"
# The N16R4 public partition rejects jobs without an explicit GPU request,
# even though data preparation itself does not execute CUDA kernels.
write_job "${data_script}" phystime_data 1 "bash scripts/prepare_phystime_thumos_i3d_n16r4.sh"
data_job="$(submit "${data_script}")"

gate_script="${SBATCH_ROOT}/real_gate.sbatch"
write_job "${gate_script}" phystime_gate 1 "
export PHYSTIME_OBSERVATION_COUNT=384
export PHYSTIME_PAIRED_TRAIN=1
export PHYSTIME_WORK_DIR='${RUN_ROOT}/real_gate/work_dir'
'${PYTHON}' tools/bata/run_phystime_real_data_gate.py \\
  --config configs/adatad/thumos/phystime_tad_i3d_feature_gate0b.py \\
  --device cuda:0 \\
  --output '${RUN_ROOT}/real_gate/real_data_gate.json'
"
gate_job="$(submit --dependency="afterok:${data_job}" "${gate_script}")"

printf 'id\tjob_id\tdependency\tconfig\tk\tseed\tstatus\n' > "${RUN_ROOT}/jobs.tsv"
printf 'data\t%s\tnone\tdata_prepare\t-\t-\tsubmitted\n' "${data_job}" >> "${RUN_ROOT}/jobs.tsv"
printf 'real_gate\t%s\tafterok:%s\tphystime\t384\t42\tsubmitted\n' "${gate_job}" "${data_job}" >> "${RUN_ROOT}/jobs.tsv"

pilots=(
  'phys_support_k384_s42|configs/adatad/thumos/phystime_tad_i3d_feature_gate0b.py|384|42|support_overlap|1|0.1|0'
  'phys_point_k384_s42|configs/adatad/thumos/phystime_tad_i3d_feature_gate0b.py|384|42|point_gaussian|0|0|0'
  'phys_nodisc_k384_s42|configs/adatad/thumos/phystime_tad_i3d_feature_gate0b.py|384|42|support_overlap|0|0|0'
  'selected_k384_s42|configs/adatad/thumos/selected_axis_actionformer_i3d_k384.py|384|42|selected_axis|0|0|0'
  'timestamp_k384_s42|configs/adatad/thumos/timestamp_selected_axis_actionformer_i3d_k384.py|384|42|timestamp_selected_axis|0|0|1'
  'phys_support_k192_s42|configs/adatad/thumos/phystime_tad_i3d_feature_gate0b.py|192|42|support_overlap|1|0.1|0'
  'phys_support_k768_s42|configs/adatad/thumos/phystime_tad_i3d_feature_gate0b.py|768|42|support_overlap|1|0.1|0'
)

for spec in "${pilots[@]}"; do
  IFS='|' read -r id config k seed measure paired disc append_time <<< "${spec}"
  script="${SBATCH_ROOT}/${id}.sbatch"
  write_job "${script}" "${id:0:28}" 1 "
export PHYSTIME_CONFIG='${config}'
export PHYSTIME_EXPERIMENT_ID='${id}'
export PHYSTIME_RUN_DIR='${RUN_ROOT}/${id}'
export PHYSTIME_SEED='${seed}'
export PHYSTIME_OBSERVATION_COUNT='${k}'
export PHYSTIME_OBSERVATION_MEASURE='${measure}'
export PHYSTIME_PAIRED_TRAIN='${paired}'
export PHYSTIME_DISCRETIZATION_WEIGHT='${disc}'
export PHYSTIME_APPEND_TIMESTAMP_CHANNELS='${append_time}'
bash scripts/run_phystime_feature_full_train_gpu1.sh
"
  job_id="$(submit --dependency="afterok:${gate_job}" "${script}")"
  printf '%s\t%s\tafterok:%s\t%s\t%s\t%s\tsubmitted\n' \
    "${id}" "${job_id}" "${gate_job}" "${config}" "${k}" "${seed}" >> "${RUN_ROOT}/jobs.tsv"
done

cat > "${RUN_ROOT}/deployment_summary.json" <<EOF
{
  "track": "phystime-tad-feature-token",
  "commit": "${COMMIT}",
  "run_root": "${RUN_ROOT}",
  "data_job": "${data_job}",
  "real_gate_job": "${gate_job}",
  "pilot_count": 7,
  "phase2_auto_submit": false
}
EOF

echo "[PHYSTIME_TRACK] RUN_ROOT=${RUN_ROOT}"
echo "[PHYSTIME_TRACK] DATA_JOB=${data_job} GATE_JOB=${gate_job}"
cat "${RUN_ROOT}/jobs.tsv"
