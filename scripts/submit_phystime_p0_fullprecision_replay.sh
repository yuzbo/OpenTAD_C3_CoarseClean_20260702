#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PhysTime P0 full-precision submit] ERROR: $*" >&2
  exit 1
}

WORK_DIR="${PHYSTIME_WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${WORK_DIR}"
BASE="${PHYSTIME_BASE:-/data/run01/sczc063/yuzibo}"
COMMIT="$(git rev-parse HEAD)"
TREE="$(git rev-parse 'HEAD^{tree}')"
SOURCE_COMMIT="${PHYSTIME_SOURCE_COMMIT:-0dc5851a8feb12b97d16bdb5ea8fc60e9273d132}"
SOURCE_TREE="${PHYSTIME_SOURCE_TREE:-bddc9b9386604d00d213275a47ce7997b35d3f4c}"
SOURCE_ROOT="${PHYSTIME_SOURCE_ROOT:-${BASE}/projects/phystime_tad/runs/phystime_g1_matched_full60_0dc5851_20260718_112053_+0800}"
SELECTED_SOURCE_DIR="${PHYSTIME_SELECTED_SOURCE_DIR:-${SOURCE_ROOT}/selected_axis}"
PHYSICAL_SOURCE_DIR="${PHYSTIME_PHYSICAL_SOURCE_DIR:-${SOURCE_ROOT}/physical_metric}"
SELECTED_CHECKPOINT="${PHYSTIME_SELECTED_CHECKPOINT:-${SELECTED_SOURCE_DIR}/work_dir/gpu1_id0/checkpoint/epoch_59.pth}"
PHYSICAL_CHECKPOINT="${PHYSTIME_PHYSICAL_CHECKPOINT:-${PHYSICAL_SOURCE_DIR}/work_dir/gpu1_id0/checkpoint/epoch_59.pth}"
PYTHON="${PHYSTIME_PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
[[ -x "${PYTHON}" ]] || fail "fixed OpenTAD Python is missing: ${PYTHON}"
[[ -f "${SELECTED_SOURCE_DIR}/run_manifest.json" \
    && -f "${PHYSICAL_SOURCE_DIR}/run_manifest.json" ]] \
  || fail "source full60 manifests are missing"
mapfile -t SOURCE_DATASET_PATHS < <(
  "${PYTHON}" - \
    "${SELECTED_SOURCE_DIR}/run_manifest.json" \
    "${PHYSICAL_SOURCE_DIR}/run_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest_paths = [Path(value).resolve() for value in sys.argv[1:]]
manifests = [
    json.loads(path.read_text(encoding="utf-8"))
    for path in manifest_paths
]
gate_paths = [Path(manifest["g1a_gate"]).resolve() for manifest in manifests]
if gate_paths[0] != gate_paths[1]:
    raise SystemExit("source arms do not bind the same full60 real gate")
gate = json.loads(gate_paths[0].read_text(encoding="utf-8"))
dataset = gate["dataset_manifest"]
for key in ("annotation", "class_map", "train_videos", "test_videos"):
    value = dataset[key]
    if isinstance(value, dict):
        value = value["path"]
    print(Path(value).resolve())
PY
)
[[ "${#SOURCE_DATASET_PATHS[@]}" == "4" ]] \
  || fail "could not recover the frozen full60 dataset paths"
SOURCE_ANNOTATION="${SOURCE_DATASET_PATHS[0]}"
SOURCE_CLASS_MAP="${SOURCE_DATASET_PATHS[1]}"
SOURCE_TRAIN_VIDEOS="${SOURCE_DATASET_PATHS[2]}"
SOURCE_TEST_VIDEOS="${SOURCE_DATASET_PATHS[3]}"
ANNOTATION="${OPENTAD_THUMOS14_ANNOTATION:-${SOURCE_ANNOTATION}}"
CLASS_MAP="${OPENTAD_THUMOS14_CLASS_MAP:-${SOURCE_CLASS_MAP}}"
TRAIN_VIDEOS="${OPENTAD_THUMOS14_TRAIN_VIDEOS:-${SOURCE_TRAIN_VIDEOS}}"
TEST_VIDEOS="${OPENTAD_THUMOS14_TEST_VIDEOS:-${SOURCE_TEST_VIDEOS}}"
if [[ -n "${PHYSTIME_VIDEOMAE_CHECKPOINT:-}" ]]; then
  VIDEOMAE_CHECKPOINT="${PHYSTIME_VIDEOMAE_CHECKPOINT}"
else
  VIDEOMAE_CHECKPOINT="$(
    "${PYTHON}" - "${SELECTED_SOURCE_DIR}/run_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(manifest["pretrained_checkpoint"])
PY
  )"
fi
RUN_TAG="${PHYSTIME_RUN_TAG:-phystime_p0_fullprecision_${COMMIT:0:7}_$(date +%Y%m%d_%H%M%S_%z)}"
RUN_ROOT="${PHYSTIME_RUN_ROOT:-${BASE}/projects/phystime_tad/runs/${RUN_TAG}}"
SBATCH_ROOT="${RUN_ROOT}/sbatch"
LOG_ROOT="${RUN_ROOT}/slurm_logs"
GATE_ROOT="${RUN_ROOT}/gate"
GATE_OUTPUT="${GATE_ROOT}/p0_fullprecision_gate.json"
TEST_LOG="${GATE_ROOT}/focused_tests.log"
PARTITION="${PHYSTIME_SLURM_PARTITION:-gpu}"
MIN_FREE_KB="${PHYSTIME_MIN_FREE_KB:-12582912}"

[[ "${SOURCE_COMMIT}" == "0dc5851a8feb12b97d16bdb5ea8fc60e9273d132" ]] \
  || fail "P0 source commit is not the reviewed full60 snapshot"
[[ "${SOURCE_TREE}" == "bddc9b9386604d00d213275a47ce7997b35d3f4c" ]] \
  || fail "P0 source tree is not the reviewed full60 snapshot"
[[ -z "$(git status --porcelain)" ]] || fail "runtime snapshot must be clean"
[[ -d "${SELECTED_SOURCE_DIR}" && -d "${PHYSICAL_SOURCE_DIR}" ]] \
  || fail "source full60 run directories are missing"
[[ -f "${SELECTED_CHECKPOINT}" && -f "${PHYSICAL_CHECKPOINT}" ]] \
  || fail "source epoch-59 checkpoints are missing"
[[ -f "${VIDEOMAE_CHECKPOINT}" ]] || fail "VideoMAE initialization checkpoint is missing"
[[ -f "${ANNOTATION}" && -f "${CLASS_MAP}" ]] \
  || fail "THUMOS annotations/class map are missing"
[[ -d "${TRAIN_VIDEOS}" && -d "${TEST_VIDEOS}" ]] \
  || fail "THUMOS raw-video roots are missing"
[[ ! -e "${RUN_ROOT}" ]] || fail "run root already exists: ${RUN_ROOT}"
[[ "${MIN_FREE_KB}" =~ ^[0-9]+$ ]] || fail "minimum free space must be an integer"
FREE_KB="$(df -Pk "${BASE}" | awk 'END {print $4}')"
[[ "${FREE_KB}" =~ ^[0-9]+$ ]] || fail "cannot determine remote free space"
(( FREE_KB >= MIN_FREE_KB )) \
  || fail "insufficient free space: ${FREE_KB} KiB < ${MIN_FREE_KB} KiB"

mkdir -p "${SBATCH_ROOT}" "${LOG_ROOT}" "${GATE_ROOT}"

submit() {
  local output attempt
  for attempt in $(seq 1 "${PHYSTIME_SUBMIT_RETRIES:-12}"); do
    if output="$(sbatch --parsable "$@" 2>&1)"; then
      printf '%s\n' "${output%%;*}"
      return 0
    fi
    echo "[PhysTime P0 submit] sbatch attempt ${attempt} failed: ${output}" >&2
    sleep "${PHYSTIME_SUBMIT_RETRY_DELAY_SEC:-20}"
  done
  return 1
}

write_header() {
  local path="$1" name="$2" time_limit="$3" needs_gpu="$4"
  {
    echo '#!/usr/bin/env bash'
    echo "#SBATCH --job-name=${name}"
    echo "#SBATCH --partition=${PARTITION}"
    if [[ "${needs_gpu}" == "true" ]]; then
      echo '#SBATCH --gres=gpu:1'
    fi
    echo '#SBATCH --cpus-per-task=6'
    echo "#SBATCH --time=${time_limit}"
    echo "#SBATCH --output=${LOG_ROOT}/${name}_%j.out"
    echo "#SBATCH --error=${LOG_ROOT}/${name}_%j.err"
    echo 'set -euo pipefail'
    printf 'cd %q\n' "${WORK_DIR}"
    printf 'export PHYSTIME_BASE=%q\n' "${BASE}"
    printf 'export PHYSTIME_WORK_DIR=%q\n' "${WORK_DIR}"
    printf 'export PHYSTIME_EXPECTED_COMMIT=%q\n' "${COMMIT}"
    printf 'export PHYSTIME_EXPECTED_TREE=%q\n' "${TREE}"
    printf 'export PHYSTIME_SOURCE_COMMIT=%q\n' "${SOURCE_COMMIT}"
    printf 'export PHYSTIME_SOURCE_TREE=%q\n' "${SOURCE_TREE}"
    printf 'export PHYSTIME_P0_GATE_OUTPUT=%q\n' "${GATE_OUTPUT}"
    printf 'export PHYSTIME_SELECTED_SOURCE_DIR=%q\n' "${SELECTED_SOURCE_DIR}"
    printf 'export PHYSTIME_PHYSICAL_SOURCE_DIR=%q\n' "${PHYSICAL_SOURCE_DIR}"
    printf 'export PHYSTIME_SELECTED_CHECKPOINT=%q\n' "${SELECTED_CHECKPOINT}"
    printf 'export PHYSTIME_PHYSICAL_CHECKPOINT=%q\n' "${PHYSICAL_CHECKPOINT}"
    printf 'export PHYSTIME_VIDEOMAE_CHECKPOINT=%q\n' "${VIDEOMAE_CHECKPOINT}"
    printf 'export OPENTAD_THUMOS14_ANNOTATION=%q\n' "${ANNOTATION}"
    printf 'export OPENTAD_THUMOS14_CLASS_MAP=%q\n' "${CLASS_MAP}"
    printf 'export OPENTAD_THUMOS14_TRAIN_VIDEOS=%q\n' "${TRAIN_VIDEOS}"
    printf 'export OPENTAD_THUMOS14_TEST_VIDEOS=%q\n' "${TEST_VIDEOS}"
    printf 'export HOME=%q\n' "${BASE}/tmp/home"
    printf 'export XDG_CACHE_HOME=%q\n' "${BASE}/tmp/xdg_cache"
    printf 'export XDG_CONFIG_HOME=%q\n' "${BASE}/tmp/xdg_config"
    printf 'export HF_HOME=%q\n' "${BASE}/hf_cache"
  } > "${path}"
  chmod +x "${path}"
}

gate_sbatch="${SBATCH_ROOT}/p0_gate.sbatch"
write_header \
  "${gate_sbatch}" pt_p0_gate "${PHYSTIME_GATE_TIME:-02:00:00}" true
{
  printf 'export PHYSTIME_P0_TEST_LOG=%q\n' "${TEST_LOG}"
  echo 'bash scripts/run_phystime_p0_fullprecision_gate_slurm.sh'
} >> "${gate_sbatch}"
gate_job="$(submit "${gate_sbatch}")"

printf 'variant\tjob_id\tdependency\tarm\tweights_source\tstatus\n' \
  > "${RUN_ROOT}/jobs.tsv"
printf 'p0_gate\t%s\tnone\tshared\tNA\tsubmitted\n' "${gate_job}" \
  >> "${RUN_ROOT}/jobs.tsv"

declare -A jobs
for spec in \
  "selected_online|selected_axis|online|configs/adatad/thumos/phystime_g1a_selected_axis_native_j192_p0_replay.py|${SELECTED_SOURCE_DIR}|${SELECTED_CHECKPOINT}" \
  "selected_ema|selected_axis|ema|configs/adatad/thumos/phystime_g1a_selected_axis_native_j192_p0_replay.py|${SELECTED_SOURCE_DIR}|${SELECTED_CHECKPOINT}" \
  "physical_online|physical_metric|online|configs/adatad/thumos/phystime_g1a_physical_metric_native_j192_p0_replay.py|${PHYSICAL_SOURCE_DIR}|${PHYSICAL_CHECKPOINT}" \
  "physical_ema|physical_metric|ema|configs/adatad/thumos/phystime_g1a_physical_metric_native_j192_p0_replay.py|${PHYSICAL_SOURCE_DIR}|${PHYSICAL_CHECKPOINT}"; do
  IFS='|' read -r variant arm weights config source_dir checkpoint <<< "${spec}"
  sbatch_path="${SBATCH_ROOT}/${variant}.sbatch"
  run_dir="${RUN_ROOT}/${variant}"
  write_header \
    "${sbatch_path}" "pt_p0_${variant}" \
    "${PHYSTIME_REPLAY_TIME:-20:00:00}" true
  {
    printf 'export PHYSTIME_P0_ARM=%q\n' "${arm}"
    printf 'export PHYSTIME_P0_WEIGHTS_SOURCE=%q\n' "${weights}"
    printf 'export PHYSTIME_P0_CONFIG=%q\n' "${WORK_DIR}/${config}"
    printf 'export PHYSTIME_P0_RUN_DIR=%q\n' "${run_dir}"
    printf 'export PHYSTIME_P0_SOURCE_DIR=%q\n' "${source_dir}"
    printf 'export PHYSTIME_P0_CHECKPOINT=%q\n' "${checkpoint}"
    printf 'export PHYSTIME_SEED=%q\n' "42"
    printf 'export PHYSTIME_EVALUATION_EPOCH=%q\n' "59"
    echo 'bash scripts/run_phystime_p0_fullprecision_replay_slurm.sh'
  } >> "${sbatch_path}"
  jobs["${variant}"]="$(submit --dependency="afterok:${gate_job}" "${sbatch_path}")"
  printf '%s\t%s\tafterok:%s\t%s\t%s\tsubmitted\n' \
    "${variant}" "${jobs[${variant}]}" "${gate_job}" "${arm}" "${weights}" \
    >> "${RUN_ROOT}/jobs.tsv"
done

suite_sbatch="${SBATCH_ROOT}/p0_suite.sbatch"
write_header \
  "${suite_sbatch}" pt_p0_suite "${PHYSTIME_SUITE_TIME:-02:00:00}" false
{
  printf 'export PHYSTIME_P0_RUN_ROOT=%q\n' "${RUN_ROOT}"
  echo 'bash scripts/run_phystime_p0_fullprecision_suite_slurm.sh'
} >> "${suite_sbatch}"
suite_dependency="afterok:${jobs[selected_online]}:${jobs[selected_ema]}:${jobs[physical_online]}:${jobs[physical_ema]}"
suite_job="$(submit --dependency="${suite_dependency}" "${suite_sbatch}")"
printf 'p0_suite\t%s\t%s\tshared\tall\t%s\n' \
  "${suite_job}" "${suite_dependency}" "submitted" \
  >> "${RUN_ROOT}/jobs.tsv"

cat > "${RUN_ROOT}/deployment_summary.json" <<EOF
{
  "schema_version": "phystime_p0_fullprecision_deployment_v1",
  "track": "p0_fullprecision_nms_frozen_epoch59",
  "runtime_commit": "${COMMIT}",
  "runtime_tree": "${TREE}",
  "source_commit": "${SOURCE_COMMIT}",
  "source_tree": "${SOURCE_TREE}",
  "source_root": "${SOURCE_ROOT}",
  "videomae_checkpoint": "${VIDEOMAE_CHECKPOINT}",
  "run_root": "${RUN_ROOT}",
  "gate_job": "${gate_job}",
  "gate_output": "${GATE_OUTPUT}",
  "jobs": {
    "selected_online": "${jobs[selected_online]}",
    "selected_ema": "${jobs[selected_ema]}",
    "physical_online": "${jobs[physical_online]}",
    "physical_ema": "${jobs[physical_ema]}",
    "p0_suite": "${suite_job}"
  },
  "suite_job": "${suite_job}",
  "suite_dependency": "${suite_dependency}",
  "suite_output": "${RUN_ROOT}/P0_SUITE_COMPLETE.json",
  "new_training": false,
  "frozen_epoch": 59,
  "arms": ["selected_axis", "physical_metric"],
  "weights_sources": ["online", "ema"],
  "replay_modes": [
    "legacy_unfiltered",
    "legacy_filtered",
    "fullprecision_unfiltered",
    "fullprecision_filtered"
  ],
  "submission_free_space_kib": ${FREE_KB},
  "minimum_free_space_kib": ${MIN_FREE_KB}
}
EOF

echo "[PhysTime P0 full-precision submit] RUN_ROOT=${RUN_ROOT}"
cat "${RUN_ROOT}/jobs.tsv"
