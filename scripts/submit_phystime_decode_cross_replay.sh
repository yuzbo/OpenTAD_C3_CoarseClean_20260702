#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PhysTime decode cross submit] ERROR: $*" >&2
  exit 1
}

WORK_DIR="${PHYSTIME_WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
BASE="${PHYSTIME_BASE:-/data/run01/sczc063/yuzibo}"
COMMIT="$(git rev-parse HEAD)"
TREE="$(git rev-parse 'HEAD^{tree}')"
SOURCE_COMMIT="${PHYSTIME_SOURCE_COMMIT:-0dc5851a8feb12b97d16bdb5ea8fc60e9273d132}"
SOURCE_TREE="${PHYSTIME_SOURCE_TREE:-bddc9b9386604d00d213275a47ce7997b35d3f4c}"
SOURCE_ROOT="${PHYSTIME_SOURCE_ROOT:-${BASE}/projects/phystime_tad/runs/phystime_g1_matched_full60_0dc5851_20260718_112053_+0800}"
P0_RUN_ROOT="${PHYSTIME_P0_RUN_ROOT:-${BASE}/projects/phystime_tad/runs/phystime_p0_fullprecision_c2cfcfa_20260720_025843_+0800}"
SELECTED_SOURCE_DIR="${PHYSTIME_SELECTED_SOURCE_DIR:-${SOURCE_ROOT}/selected_axis}"
PHYSICAL_SOURCE_DIR="${PHYSTIME_PHYSICAL_SOURCE_DIR:-${SOURCE_ROOT}/physical_metric}"
SELECTED_CHECKPOINT="${PHYSTIME_SELECTED_CHECKPOINT:-${SELECTED_SOURCE_DIR}/work_dir/gpu1_id0/checkpoint/epoch_59.pth}"
PHYSICAL_CHECKPOINT="${PHYSTIME_PHYSICAL_CHECKPOINT:-${PHYSICAL_SOURCE_DIR}/work_dir/gpu1_id0/checkpoint/epoch_59.pth}"
SELECTED_CONFIG="${WORK_DIR}/configs/adatad/thumos/phystime_g1a_selected_axis_native_j192_decode_replay.py"
PHYSICAL_CONFIG="${WORK_DIR}/configs/adatad/thumos/phystime_g1a_physical_metric_native_j192_decode_replay.py"
PYTHON="${PHYSTIME_PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
P0_SUITE="${P0_RUN_ROOT}/P0_SUITE_COMPLETE.json"

[[ -x "${PYTHON}" ]] || fail "fixed OpenTAD Python is missing: ${PYTHON}"
for command_name in sbatch scancel squeue sacct scontrol flock; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || fail "required Slurm command is missing: ${command_name}"
done
[[ "${SOURCE_COMMIT}" == "0dc5851a8feb12b97d16bdb5ea8fc60e9273d132" ]] \
  || fail "source commit is not the reviewed full60 snapshot"
[[ "${SOURCE_TREE}" == "bddc9b9386604d00d213275a47ce7997b35d3f4c" ]] \
  || fail "source tree is not the reviewed full60 snapshot"
[[ -z "$(git status --porcelain)" ]] || fail "runtime snapshot must be clean"
for path in \
  "${SELECTED_CONFIG}" \
  "${PHYSICAL_CONFIG}" \
  "${SELECTED_SOURCE_DIR}/FULL_COMPLETE.json" \
  "${SELECTED_SOURCE_DIR}/run_manifest.json" \
  "${PHYSICAL_SOURCE_DIR}/FULL_COMPLETE.json" \
  "${PHYSICAL_SOURCE_DIR}/run_manifest.json" \
  "${SELECTED_CHECKPOINT}" \
  "${PHYSICAL_CHECKPOINT}" \
  "${P0_SUITE}" \
  "${P0_RUN_ROOT}/selected_online/P0_COMPLETE.json" \
  "${P0_RUN_ROOT}/selected_ema/P0_COMPLETE.json" \
  "${P0_RUN_ROOT}/physical_online/P0_COMPLETE.json" \
  "${P0_RUN_ROOT}/physical_ema/P0_COMPLETE.json"; do
  [[ -f "${path}" ]] || fail "required frozen artifact is missing: ${path}"
done

mapfile -t SOURCE_BINDINGS < <(
  "${PYTHON}" - \
    "${SELECTED_SOURCE_DIR}/run_manifest.json" \
    "${PHYSICAL_SOURCE_DIR}/run_manifest.json" \
    "${P0_SUITE}" \
    "${SOURCE_COMMIT}" "${SOURCE_TREE}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

P0_RUNTIME_COMMIT = "c2cfcfa2470f9f1e0b9d10e397480f6c66aeaf2c"
P0_RUNTIME_TREE = "0b78dd402e8997239ef9d1b4b4cd8bfa4f7a6338"
P0_SUITE_SHA256 = "afb3e300424a57eb590a21129217e040677dc875fdede3be344352dc2bd268e7"
P0_GATE_SHA256 = "1ca0efcdeb9f6343da076a00660675759358ac467074919a34d01c0d7c7250d9"
DATASET_SHA256 = "1da0bca28f14ca2f1e4b2baf0f199dce18f4dd925e0f097a70a3fcc1c13eb1b2"
VIDEOMAE_SHA256 = "4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251"

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

selected_path = Path(sys.argv[1]).resolve()
physical_path = Path(sys.argv[2]).resolve()
p0_path = Path(sys.argv[3]).resolve()
source_commit = sys.argv[4]
source_tree = sys.argv[5]
selected = json.loads(selected_path.read_text(encoding="utf-8"))
physical = json.loads(physical_path.read_text(encoding="utf-8"))
p0 = json.loads(p0_path.read_text(encoding="utf-8"))
if selected["commit"] != source_commit:
    raise SystemExit("selected source commit mismatch")
if physical["commit"] != source_commit:
    raise SystemExit("physical source commit mismatch")
if selected["git_tree"] != source_tree:
    raise SystemExit("selected source tree mismatch")
if physical["git_tree"] != source_tree:
    raise SystemExit("physical source tree mismatch")
if (
    p0.get("schema_version")
    != "phystime_p0_fullprecision_suite_completion_v1"
    or p0.get("validation_pass") is not True
    or p0.get("runtime_commit") != P0_RUNTIME_COMMIT
    or p0.get("runtime_tree") != P0_RUNTIME_TREE
    or sha256_file(p0_path) != P0_SUITE_SHA256
):
    raise SystemExit("P0 suite identity/hash did not pass")
if p0["source_commit"] != source_commit:
    raise SystemExit("P0 source commit mismatch")
if p0["source_tree"] != source_tree:
    raise SystemExit("P0 source tree mismatch")
p0_gate_path = Path(p0["gate"]["path"]).resolve()
if (
    p0["gate"]["sha256"] != P0_GATE_SHA256
    or sha256_file(p0_gate_path) != P0_GATE_SHA256
):
    raise SystemExit("P0 gate hash mismatch")
p0_gate = json.loads(p0_gate_path.read_text(encoding="utf-8"))
if (
    p0_gate["runtime"]["dataset_manifest_sha256"] != DATASET_SHA256
    or p0_gate["runtime"]["videomae_checkpoint_sha256"] != VIDEOMAE_SHA256
):
    raise SystemExit("P0 data/VideoMAE binding mismatch")
for name, manifest in (("selected", selected), ("physical", physical)):
    if manifest.get("dataset_manifest_sha256") != DATASET_SHA256:
        raise SystemExit(f"{name} source dataset hash mismatch")
    if manifest.get("pretrained_checkpoint_sha256") != VIDEOMAE_SHA256:
        raise SystemExit(f"{name} source VideoMAE hash mismatch")
for variant, arm, weights in (
    ("selected_online", "selected_axis", "online"),
    ("selected_ema", "selected_axis", "ema"),
    ("physical_online", "physical_metric", "online"),
    ("physical_ema", "physical_metric", "ema"),
):
    completion_path = p0_path.parent / variant / "P0_COMPLETE.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    suite_artifact = p0["completion_artifacts"][variant]
    if (
        completion.get("schema_version")
        != "phystime_p0_fullprecision_completion_v2"
        or completion.get("validation_pass") is not True
        or completion.get("arm") != arm
        or completion.get("weights_source") != weights
        or completion.get("runtime_commit") != P0_RUNTIME_COMMIT
        or completion.get("runtime_tree") != P0_RUNTIME_TREE
        or completion["artifacts"]["gate"]["sha256"] != P0_GATE_SHA256
        or Path(suite_artifact["path"]).resolve()
        != completion_path.resolve()
        or suite_artifact["sha256"] != sha256_file(completion_path)
    ):
        raise SystemExit(f"{variant} P0 completion provenance mismatch")
gate_paths = [
    Path(selected["g1a_gate"]).resolve(),
    Path(physical["g1a_gate"]).resolve(),
]
if gate_paths[0] != gate_paths[1]:
    raise SystemExit("source arms do not bind the same full60 real gate")
gate = json.loads(gate_paths[0].read_text(encoding="utf-8"))
dataset = gate["dataset_manifest"]
for key in ("annotation", "class_map", "train_videos", "test_videos"):
    value = dataset[key]
    if isinstance(value, dict):
        value = value["path"]
    print(Path(value).resolve())
if Path(selected["pretrained_checkpoint"]).resolve() != Path(
    physical["pretrained_checkpoint"]
).resolve():
    raise SystemExit("source arms do not bind the same VideoMAE checkpoint")
print(Path(selected["pretrained_checkpoint"]).resolve())
PY
)
[[ "${#SOURCE_BINDINGS[@]}" == "5" ]] \
  || fail "could not recover the reviewed dataset and VideoMAE bindings"
SOURCE_ANNOTATION="${SOURCE_BINDINGS[0]}"
SOURCE_CLASS_MAP="${SOURCE_BINDINGS[1]}"
SOURCE_TRAIN_VIDEOS="${SOURCE_BINDINGS[2]}"
SOURCE_TEST_VIDEOS="${SOURCE_BINDINGS[3]}"
SOURCE_VIDEOMAE="${SOURCE_BINDINGS[4]}"
ANNOTATION="${SOURCE_ANNOTATION}"
CLASS_MAP="${SOURCE_CLASS_MAP}"
TRAIN_VIDEOS="${SOURCE_TRAIN_VIDEOS}"
TEST_VIDEOS="${SOURCE_TEST_VIDEOS}"
VIDEOMAE_CHECKPOINT="${SOURCE_VIDEOMAE}"
export OPENTAD_THUMOS14_ANNOTATION="${ANNOTATION}"
export OPENTAD_THUMOS14_CLASS_MAP="${CLASS_MAP}"
export OPENTAD_THUMOS14_TRAIN_VIDEOS="${TRAIN_VIDEOS}"
export OPENTAD_THUMOS14_TEST_VIDEOS="${TEST_VIDEOS}"

for path in "${VIDEOMAE_CHECKPOINT}" "${ANNOTATION}" "${CLASS_MAP}"; do
  [[ -f "${path}" ]] || fail "bound input file is missing: ${path}"
done
for path in "${TRAIN_VIDEOS}" "${TEST_VIDEOS}"; do
  [[ -d "${path}" ]] || fail "bound raw-video directory is missing: ${path}"
done

sha256_file() {
  "${PYTHON}" - "$1" <<'PY'
import hashlib
import sys
from pathlib import Path

digest = hashlib.sha256()
with Path(sys.argv[1]).open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
}

RUN_UUID="${PHYSTIME_RUN_UUID:-$("${PYTHON}" -c 'import uuid; print(uuid.uuid4().hex)')}"
[[ "${RUN_UUID}" =~ ^[0-9a-f]{32}$ ]] \
  || fail "run UUID must contain exactly 32 lowercase hexadecimal characters"
RUN_TAG="${PHYSTIME_RUN_TAG:-phystime_decode_cross_${COMMIT:0:7}_$(date +%Y%m%d_%H%M%S_%z)_${RUN_UUID}}"
RUN_ROOT="${PHYSTIME_RUN_ROOT:-${BASE}/projects/phystime_tad/runs/${RUN_TAG}}"
OWNER_MANIFEST="${RUN_ROOT}/submission_owner.json"
EXISTING_DAG_TOKEN=""
if [[ -f "${OWNER_MANIFEST}" && -z "${PHYSTIME_DAG_TOKEN:-}" ]]; then
  EXISTING_DAG_TOKEN="$("${PYTHON}" - "${OWNER_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("dag_token", ""))
PY
)"
fi
DAG_TOKEN="${PHYSTIME_DAG_TOKEN:-${EXISTING_DAG_TOKEN:-${RUN_TAG}_${COMMIT:0:12}}}"
[[ "${DAG_TOKEN}" =~ ^[A-Za-z0-9_.:+-]+$ ]] \
  || fail "DAG token contains unsupported characters: ${DAG_TOKEN}"
SBATCH_ROOT="${RUN_ROOT}/sbatch"
LOG_ROOT="${RUN_ROOT}/slurm_logs"
GATE_ROOT="${RUN_ROOT}/gate"
GATE_OUTPUT="${GATE_ROOT}/decode_cross_gate.json"
TEST_LOG="${GATE_ROOT}/focused_tests.log"
PARTITION="${PHYSTIME_SLURM_PARTITION:-gpu}"
MIN_FREE_KB="${PHYSTIME_MIN_FREE_KB:-12582912}"

[[ "${MIN_FREE_KB}" =~ ^[0-9]+$ ]] || fail "minimum free space must be an integer"
FREE_KB="$(df -Pk "${BASE}" | awk 'END {print $4}')"
[[ "${FREE_KB}" =~ ^[0-9]+$ ]] || fail "cannot determine remote free space"
(( FREE_KB >= MIN_FREE_KB )) \
  || fail "insufficient free space: ${FREE_KB} KiB < ${MIN_FREE_KB} KiB"

mkdir -p "${RUN_ROOT}" || fail "cannot create run root: ${RUN_ROOT}"
[[ -d "${RUN_ROOT}" ]] || fail "run root is not a directory: ${RUN_ROOT}"
SUBMISSION_LOCK_ROOT="${BASE}/projects/phystime_tad/submission_locks"
mkdir -p "${SUBMISSION_LOCK_ROOT}"
exec 9>"${SUBMISSION_LOCK_ROOT}/${DAG_TOKEN}.lock"
flock -n 9 || fail "another submitter owns DAG token ${DAG_TOKEN}"
GLOBAL_OWNER_MANIFEST="${SUBMISSION_LOCK_ROOT}/${DAG_TOKEN}.owner.json"
RECOVERY_MODE="$(
  "${PYTHON}" tools/bata/claim_phystime_decode_cross_owner.py \
    --global-owner "${GLOBAL_OWNER_MANIFEST}" \
    --local-owner "${OWNER_MANIFEST}" \
    --run-root "${RUN_ROOT}" \
    --dag-token "${DAG_TOKEN}" \
    --runtime-commit "${COMMIT}" \
    --runtime-tree "${TREE}" \
    --run-uuid "${RUN_UUID}"
)"
[[ "${RECOVERY_MODE}" == "0" || "${RECOVERY_MODE}" == "1" ]] \
  || fail "submission owner claim returned an invalid recovery mode"
OWNER_MANIFEST_SHA256="$(sha256_file "${OWNER_MANIFEST}")"
GLOBAL_OWNER_MANIFEST_SHA256="$(sha256_file "${GLOBAL_OWNER_MANIFEST}")"
mkdir -p "${SBATCH_ROOT}" "${LOG_ROOT}" "${GATE_ROOT}"
AMBIGUOUS_SUBMISSION_ROOT="${RUN_ROOT}/submission_attempts"
mkdir -p "${AMBIGUOUS_SUBMISSION_ROOT}"

PREFLIGHT_MANIFEST="${RUN_ROOT}/preflight_manifest.json"
"${PYTHON}" tools/bata/preflight_phystime_decode_cross.py \
  --selected-config "${SELECTED_CONFIG}" \
  --physical-config "${PHYSICAL_CONFIG}" \
  --selected-checkpoint "${SELECTED_CHECKPOINT}" \
  --physical-checkpoint "${PHYSICAL_CHECKPOINT}" \
  --videomae-checkpoint "${VIDEOMAE_CHECKPOINT}" \
  --selected-source-dir "${SELECTED_SOURCE_DIR}" \
  --physical-source-dir "${PHYSICAL_SOURCE_DIR}" \
  --p0-run-root "${P0_RUN_ROOT}" \
  --expected-runtime-commit "${COMMIT}" \
  --expected-runtime-tree "${TREE}" \
  --output "${PREFLIGHT_MANIFEST}"
PREFLIGHT_SHA256="$(sha256_file "${PREFLIGHT_MANIFEST}")"
[[ "${PREFLIGHT_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || fail "preflight manifest SHA256 is invalid"

submitted_jobs=()
submission_complete=0
lookup_jobs_by_comment() {
  local expected_comment="$1"
  {
    squeue --noheader --user="${USER}" --format='%i|%k' 2>/dev/null || true
    sacct -nX --user="${USER}" \
      --starttime "$(date -d '2 days ago' +%F)" \
      --format=JobIDRaw,Comment%256 -P 2>/dev/null || true
  } | awk -F'|' -v expected="${expected_comment}" '
    {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
      if ($1 ~ /^[0-9]+$/ && $2 == expected) print $1
    }
  ' | sort -u
}

lookup_dag_jobs() {
  {
    squeue --noheader --user="${USER}" --format='%i|%k' 2>/dev/null || true
    sacct -nX --user="${USER}" \
      --starttime "$(date -d '2 days ago' +%F)" \
      --format=JobIDRaw,Comment%256 -P 2>/dev/null || true
  } | awk -F'|' -v prefix="${DAG_TOKEN}:" '
    {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
      if ($1 ~ /^[0-9]+$/ && index($2, prefix) == 1) print $1
    }
  ' | sort -u
}

cancel_partial_submission() {
  local status=$?
  if (( status != 0 && submission_complete == 0 )); then
    if compgen -G "${AMBIGUOUS_SUBMISSION_ROOT}/*.fatal.json" >/dev/null; then
      echo "[PhysTime decode cross submit] fatal submission state requires token-scoped cleanup for ${DAG_TOKEN}" >&2
    elif compgen -G "${AMBIGUOUS_SUBMISSION_ROOT}/*.ambiguous.json" >/dev/null \
      || compgen -G "${AMBIGUOUS_SUBMISSION_ROOT}/*.resolved.json" >/dev/null; then
      echo "[PhysTime decode cross submit] preserving token ${DAG_TOKEN}: persistent submission state requires query-only recovery" >&2
      exit "${status}"
    fi
    mapfile -t token_jobs < <(lookup_dag_jobs)
    if (( ${#token_jobs[@]} > 0 )); then
      echo "[PhysTime decode cross submit] cancelling only token ${DAG_TOKEN}: ${token_jobs[*]}" >&2
      scancel "${token_jobs[@]}" || true
    fi
  fi
  exit "${status}"
}
trap cancel_partial_submission EXIT

record_ambiguous_submission() {
  local variant="$1" comment="$2" phase="$3"
  local expected_job_id="$4" sbatch_output="$5"
  local ambiguous_path="${AMBIGUOUS_SUBMISSION_ROOT}/${variant}.ambiguous.json"
  "${PYTHON}" tools/bata/manage_phystime_decode_cross_submission_state.py \
    record \
    --output "${ambiguous_path}" \
    --run-root "${RUN_ROOT}" \
    --dag-token "${DAG_TOKEN}" \
    --variant "${variant}" \
    --comment "${comment}" \
    --runtime-commit "${COMMIT}" \
    --runtime-tree "${TREE}" \
    --phase "${phase}" \
    --expected-job-id "${expected_job_id}" \
    --sbatch-output "${sbatch_output}"
}

resolve_ambiguous_submission() {
  local variant="$1" job_id="$2"
  local ambiguous_path="${AMBIGUOUS_SUBMISSION_ROOT}/${variant}.ambiguous.json"
  local resolved_path="${AMBIGUOUS_SUBMISSION_ROOT}/${variant}.resolved.json"
  if [[ -f "${ambiguous_path}" ]]; then
    "${PYTHON}" tools/bata/manage_phystime_decode_cross_submission_state.py \
      resolve \
      --ambiguous "${ambiguous_path}" \
      --resolved "${resolved_path}" \
      --job-id "${job_id}"
  fi
}

abort_ambiguous_submission() {
  local variant="$1" reason="$2"
  local ambiguous_path="${AMBIGUOUS_SUBMISSION_ROOT}/${variant}.ambiguous.json"
  local fatal_path="${AMBIGUOUS_SUBMISSION_ROOT}/${variant}.fatal.json"
  if [[ ! -f "${ambiguous_path}" ]]; then
    record_ambiguous_submission \
      "${variant}" "${DAG_TOKEN}:${variant}" "fatal_conflict" "" "${reason}"
  fi
  "${PYTHON}" tools/bata/manage_phystime_decode_cross_submission_state.py \
    abort \
    --ambiguous "${ambiguous_path}" \
    --fatal "${fatal_path}" \
    --reason "${reason}"
}

inspect_resolved_submission() {
  local variant="$1" comment="$2"
  local resolved_path="${AMBIGUOUS_SUBMISSION_ROOT}/${variant}.resolved.json"
  "${PYTHON}" tools/bata/manage_phystime_decode_cross_submission_state.py \
    inspect-resolved \
    --resolved "${resolved_path}" \
    --run-root "${RUN_ROOT}" \
    --dag-token "${DAG_TOKEN}" \
    --variant "${variant}" \
    --comment "${comment}" \
    --runtime-commit "${COMMIT}" \
    --runtime-tree "${TREE}"
}

fatalize_submission_state() {
  local source_path="$1" variant="$2" reason="$3"
  local fatal_path="${AMBIGUOUS_SUBMISSION_ROOT}/${variant}.fatal.json"
  "${PYTHON}" tools/bata/manage_phystime_decode_cross_submission_state.py \
    fatalize \
    --source "${source_path}" \
    --fatal "${fatal_path}" \
    --reason "${reason}"
}

submit() {
  local variant="$1"
  shift
  local comment="${DAG_TOKEN}:${variant}"
  local output job_id visibility_attempt response_phase
  local ambiguous_path="${AMBIGUOUS_SUBMISSION_ROOT}/${variant}.ambiguous.json"
  local resolved_path="${AMBIGUOUS_SUBMISSION_ROOT}/${variant}.resolved.json"
  local resolved_job_id=""
  local -a matches
  if compgen -G "${AMBIGUOUS_SUBMISSION_ROOT}/*.fatal.json" >/dev/null; then
    fail "DAG token ${DAG_TOKEN} has a persistent fatal submission state"
  fi
  if [[ -f "${resolved_path}" ]]; then
    if ! resolved_job_id="$(
      inspect_resolved_submission "${variant}" "${comment}"
    )"; then
      fatalize_submission_state \
        "${resolved_path}" "${variant}" \
        "resolved marker failed contract validation"
      fail "resolved submission marker failed validation for ${comment}"
    fi
    mapfile -t matches < <(lookup_jobs_by_comment "${comment}")
    if (( ${#matches[@]} > 1 )); then
      fatalize_submission_state \
        "${resolved_path}" "${variant}" \
        "resolved comment has multiple scheduler jobs: ${matches[*]}"
      fail "multiple jobs found for resolved comment ${comment}: ${matches[*]}"
    fi
    if (( ${#matches[@]} == 0 )); then
      echo "[PhysTime decode cross submit] resolved job ${resolved_job_id} for ${comment} is temporarily invisible; refusing resubmission" >&2
      return 1
    fi
    if [[ "${matches[0]}" != "${resolved_job_id}" ]]; then
      fatalize_submission_state \
        "${resolved_path}" "${variant}" \
        "resolved job ${resolved_job_id} differs from scheduler job ${matches[0]}"
      fail "resolved job mismatch for ${comment}: marker=${resolved_job_id}, scheduler=${matches[0]}"
    fi
    printf '%s\n' "${resolved_job_id}"
    return 0
  fi
  mapfile -t matches < <(lookup_jobs_by_comment "${comment}")
  if (( ${#matches[@]} > 1 )); then
    abort_ambiguous_submission \
      "${variant}" "multiple existing jobs share comment: ${matches[*]}"
    fail "multiple existing jobs share comment ${comment}: ${matches[*]}"
  fi
  if (( ${#matches[@]} == 1 )); then
    if [[ ! -f "${ambiguous_path}" ]]; then
      record_ambiguous_submission \
        "${variant}" "${comment}" "scheduler_adoption" "${matches[0]}" ""
    fi
    if ! resolve_ambiguous_submission "${variant}" "${matches[0]}"; then
      abort_ambiguous_submission \
        "${variant}" \
        "visible scheduler job ${matches[0]} conflicts with recorded intent"
      fail "visible scheduler job conflicts with recorded intent for ${comment}"
    fi
    printf '%s\n' "${matches[0]}"
    return 0
  fi
  if [[ -f "${ambiguous_path}" ]]; then
    echo "[PhysTime decode cross submit] unresolved intent for ${comment}; recovery is query-only until reliable accounting or manual reconciliation" >&2
    return 1
  fi

  record_ambiguous_submission \
    "${variant}" "${comment}" "before_sbatch" "" ""
  if output="$(sbatch --parsable --comment="${comment}" "$@" 2>&1)"; then
    job_id="${output%%;*}"
    if [[ "${job_id}" =~ ^[0-9]+$ ]]; then
      record_ambiguous_submission \
        "${variant}" "${comment}" "numeric_response" "${job_id}" "${output}"
      response_phase="numeric_response"
    else
      job_id=""
      record_ambiguous_submission \
        "${variant}" "${comment}" "non_numeric_response" "" "${output}"
      response_phase="non_numeric_response"
      echo "[PhysTime decode cross submit] non-numeric sbatch response: ${output}" >&2
    fi
  else
    job_id=""
    record_ambiguous_submission \
      "${variant}" "${comment}" "failed_response" "" "${output}"
    response_phase="failed_response"
    echo "[PhysTime decode cross submit] sbatch response failed: ${output}" >&2
  fi

  for visibility_attempt in $(seq 1 "${PHYSTIME_SUBMIT_VISIBILITY_POLLS:-20}"); do
    mapfile -t matches < <(lookup_jobs_by_comment "${comment}")
    if (( ${#matches[@]} > 1 )); then
      abort_ambiguous_submission \
        "${variant}" "multiple accepted jobs became visible: ${matches[*]}"
      fail "ambiguous accepted jobs for ${comment}: ${matches[*]}"
    fi
    if (( ${#matches[@]} == 1 )); then
      if [[ -n "${job_id}" && "${matches[0]}" != "${job_id}" ]]; then
        abort_ambiguous_submission \
          "${variant}" "sbatch job ${job_id} differs from scheduler job ${matches[0]}"
        fail "accepted job mismatch for ${comment}: sbatch=${job_id}, scheduler=${matches[0]}"
      fi
      resolve_ambiguous_submission "${variant}" "${matches[0]}"
      printf '%s\n' "${matches[0]}"
      return 0
    fi
    sleep "${PHYSTIME_SUBMIT_VISIBILITY_DELAY_SEC:-1}"
  done
  echo "[PhysTime decode cross submit] ${response_phase} for ${comment} remains unresolved; refusing automatic resubmission" >&2
  return 1
}

write_header() {
  local path="$1" name="$2" time_limit="$3"
  {
    echo '#!/usr/bin/env bash'
    echo "#SBATCH --job-name=${name}"
    echo "#SBATCH --partition=${PARTITION}"
    echo '#SBATCH --gpus=1'
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
    printf 'export PHYSTIME_SELECTED_SOURCE_DIR=%q\n' "${SELECTED_SOURCE_DIR}"
    printf 'export PHYSTIME_PHYSICAL_SOURCE_DIR=%q\n' "${PHYSICAL_SOURCE_DIR}"
    printf 'export PHYSTIME_SELECTED_CHECKPOINT=%q\n' "${SELECTED_CHECKPOINT}"
    printf 'export PHYSTIME_PHYSICAL_CHECKPOINT=%q\n' "${PHYSICAL_CHECKPOINT}"
    printf 'export PHYSTIME_DECODE_SELECTED_CONFIG=%q\n' "${SELECTED_CONFIG}"
    printf 'export PHYSTIME_DECODE_PHYSICAL_CONFIG=%q\n' "${PHYSICAL_CONFIG}"
    printf 'export PHYSTIME_VIDEOMAE_CHECKPOINT=%q\n' "${VIDEOMAE_CHECKPOINT}"
    printf 'export PHYSTIME_P0_RUN_ROOT=%q\n' "${P0_RUN_ROOT}"
    printf 'export PHYSTIME_DECODE_GATE_ROOT=%q\n' "${GATE_ROOT}"
    printf 'export PHYSTIME_DECODE_GATE_OUTPUT=%q\n' "${GATE_OUTPUT}"
    printf 'export PHYSTIME_DECODE_RUN_ROOT=%q\n' "${RUN_ROOT}"
    printf 'export PHYSTIME_DECODE_JOBS_TSV=%q\n' "${RUN_ROOT}/jobs.tsv"
    printf 'export PHYSTIME_DECODE_DEPLOYMENT_SUMMARY=%q\n' "${RUN_ROOT}/deployment_summary.json"
    printf 'export PHYSTIME_DECODE_SLURM_LOG_ROOT=%q\n' "${LOG_ROOT}"
    printf 'export PHYSTIME_DECODE_PREFLIGHT=%q\n' "${PREFLIGHT_MANIFEST}"
    printf 'export PHYSTIME_DECODE_PREFLIGHT_SHA256=%q\n' "${PREFLIGHT_SHA256}"
    printf 'export PHYSTIME_DAG_TOKEN=%q\n' "${DAG_TOKEN}"
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

gate_sbatch="${SBATCH_ROOT}/decode_cross_gate.sbatch"
write_header \
  "${gate_sbatch}" pt_dc_gate "${PHYSTIME_GATE_TIME:-04:00:00}"
{
  printf 'export PHYSTIME_JOB_VARIANT=%q\n' "decode_cross_gate"
  printf 'export PHYSTIME_SBATCH_PATH=%q\n' "${gate_sbatch}"
  printf 'export PHYSTIME_EXPECTED_DEPENDENCY=%q\n' "none"
  printf 'export PHYSTIME_DECODE_SELECTED_CONFIG=%q\n' "${SELECTED_CONFIG}"
  printf 'export PHYSTIME_DECODE_PHYSICAL_CONFIG=%q\n' "${PHYSICAL_CONFIG}"
  printf 'export PHYSTIME_DECODE_TEST_LOG=%q\n' "${TEST_LOG}"
  printf 'export PHYSTIME_EXPECTED_JOB_COMMENT=%q\n' "${DAG_TOKEN}:decode_cross_gate"
  echo 'bash scripts/run_phystime_decode_cross_gate_slurm.sh'
} >> "${gate_sbatch}"
gate_sbatch_sha="$(sha256_file "${gate_sbatch}")"
gate_job="$(submit decode_cross_gate "${gate_sbatch}")"
submitted_jobs+=("${gate_job}")

jobs_tmp="${RUN_ROOT}/jobs.tsv.tmp.$$"
printf 'variant\tjob_id\tdependency\tarm\tweights_source\tjob_name\tdag_token\tcomment\tsbatch_path\tsbatch_sha256\tstdout\tstderr\tstatus\n' \
  > "${jobs_tmp}"
printf 'decode_cross_gate\t%s\tnone\tshared\tNA\tpt_dc_gate\t%s\t%s\t%s\t%s\t%s\t%s\tsubmitted\n' \
  "${gate_job}" "${DAG_TOKEN}" "${DAG_TOKEN}:decode_cross_gate" \
  "${gate_sbatch}" "${gate_sbatch_sha}" \
  "${LOG_ROOT}/pt_dc_gate_${gate_job}.out" \
  "${LOG_ROOT}/pt_dc_gate_${gate_job}.err" \
  >> "${jobs_tmp}"

declare -A jobs
for spec in \
  "selected_online|selected_axis|online|${SELECTED_CONFIG}|${SELECTED_SOURCE_DIR}|${SELECTED_CHECKPOINT}" \
  "selected_ema|selected_axis|ema|${SELECTED_CONFIG}|${SELECTED_SOURCE_DIR}|${SELECTED_CHECKPOINT}" \
  "physical_online|physical_metric|online|${PHYSICAL_CONFIG}|${PHYSICAL_SOURCE_DIR}|${PHYSICAL_CHECKPOINT}" \
  "physical_ema|physical_metric|ema|${PHYSICAL_CONFIG}|${PHYSICAL_SOURCE_DIR}|${PHYSICAL_CHECKPOINT}"; do
  IFS='|' read -r variant arm weights config source_dir checkpoint <<< "${spec}"
  p0_completion="${P0_RUN_ROOT}/${variant}/P0_COMPLETE.json"
  sbatch_path="${SBATCH_ROOT}/${variant}.sbatch"
  run_dir="${RUN_ROOT}/${variant}"
  write_header \
    "${sbatch_path}" "pt_dc_${variant}" \
    "${PHYSTIME_REPLAY_TIME:-20:00:00}"
  {
    printf 'export PHYSTIME_JOB_VARIANT=%q\n' "${variant}"
    printf 'export PHYSTIME_SBATCH_PATH=%q\n' "${sbatch_path}"
    printf 'export PHYSTIME_EXPECTED_DEPENDENCY=%q\n' "afterok:${gate_job}"
    printf 'export PHYSTIME_DECODE_ARM=%q\n' "${arm}"
    printf 'export PHYSTIME_DECODE_WEIGHTS_SOURCE=%q\n' "${weights}"
    printf 'export PHYSTIME_DECODE_CONFIG=%q\n' "${config}"
    printf 'export PHYSTIME_DECODE_RUN_DIR=%q\n' "${run_dir}"
    printf 'export PHYSTIME_DECODE_SOURCE_DIR=%q\n' "${source_dir}"
    printf 'export PHYSTIME_DECODE_CHECKPOINT=%q\n' "${checkpoint}"
    printf 'export PHYSTIME_DECODE_P0_COMPLETION=%q\n' "${p0_completion}"
    printf 'export PHYSTIME_EXPECTED_JOB_COMMENT=%q\n' "${DAG_TOKEN}:${variant}"
    printf 'export PHYSTIME_SEED=%q\n' "42"
    printf 'export PHYSTIME_EVALUATION_EPOCH=%q\n' "59"
    echo 'bash scripts/run_phystime_decode_cross_replay_slurm.sh'
  } >> "${sbatch_path}"
  sbatch_sha="$(sha256_file "${sbatch_path}")"
  jobs["${variant}"]="$(submit "${variant}" --dependency="afterok:${gate_job}" "${sbatch_path}")"
  submitted_jobs+=("${jobs[${variant}]}")
  printf '%s\t%s\tafterok:%s\t%s\t%s\tpt_dc_%s\t%s\t%s\t%s\t%s\t%s\t%s\tsubmitted\n' \
    "${variant}" "${jobs[${variant}]}" "${gate_job}" "${arm}" "${weights}" \
    "${variant}" "${DAG_TOKEN}" "${DAG_TOKEN}:${variant}" \
    "${sbatch_path}" "${sbatch_sha}" \
    "${LOG_ROOT}/pt_dc_${variant}_${jobs[${variant}]}.out" \
    "${LOG_ROOT}/pt_dc_${variant}_${jobs[${variant}]}.err" \
    >> "${jobs_tmp}"
done

suite_sbatch="${SBATCH_ROOT}/decode_cross_suite.sbatch"
suite_dependency="afterok:${jobs[selected_online]}:${jobs[selected_ema]}:${jobs[physical_online]}:${jobs[physical_ema]}"
write_header \
  "${suite_sbatch}" pt_dc_suite "${PHYSTIME_SUITE_TIME:-02:00:00}"
{
  printf 'export PHYSTIME_JOB_VARIANT=%q\n' "decode_cross_suite"
  printf 'export PHYSTIME_SBATCH_PATH=%q\n' "${suite_sbatch}"
  printf 'export PHYSTIME_EXPECTED_DEPENDENCY=%q\n' "${suite_dependency}"
  printf 'export PHYSTIME_DECODE_RUN_ROOT=%q\n' "${RUN_ROOT}"
  printf 'export PHYSTIME_EXPECTED_JOB_COMMENT=%q\n' "${DAG_TOKEN}:decode_cross_suite"
  echo 'bash scripts/run_phystime_decode_cross_suite_slurm.sh'
} >> "${suite_sbatch}"
suite_sbatch_sha="$(sha256_file "${suite_sbatch}")"
suite_job="$(submit decode_cross_suite --dependency="${suite_dependency}" "${suite_sbatch}")"
submitted_jobs+=("${suite_job}")
printf 'decode_cross_suite\t%s\t%s\tshared\tall\tpt_dc_suite\t%s\t%s\t%s\t%s\t%s\t%s\tsubmitted\n' \
  "${suite_job}" "${suite_dependency}" "${DAG_TOKEN}" \
  "${DAG_TOKEN}:decode_cross_suite" "${suite_sbatch}" \
  "${suite_sbatch_sha}" "${LOG_ROOT}/pt_dc_suite_${suite_job}.out" \
  "${LOG_ROOT}/pt_dc_suite_${suite_job}.err" >> "${jobs_tmp}"
mv "${jobs_tmp}" "${RUN_ROOT}/jobs.tsv"
jobs_tsv_sha="$(sha256_file "${RUN_ROOT}/jobs.tsv")"
SCHEDULER_SUBMISSION="${RUN_ROOT}/scheduler_submission.json"
"${PYTHON}" tools/bata/capture_phystime_decode_cross_scheduler.py \
  --jobs-tsv "${RUN_ROOT}/jobs.tsv" \
  --dag-token "${DAG_TOKEN}" \
  --mode submission \
  --output "${SCHEDULER_SUBMISSION}"
SCHEDULER_SUBMISSION_SHA256="$(sha256_file "${SCHEDULER_SUBMISSION}")"

"${PYTHON}" - \
  "${RUN_ROOT}/deployment_summary.json" \
  "${COMMIT}" "${TREE}" "${SOURCE_COMMIT}" "${SOURCE_TREE}" \
  "${SOURCE_ROOT}" "${P0_RUN_ROOT}" "${P0_SUITE}" \
  "${VIDEOMAE_CHECKPOINT}" "${RUN_ROOT}" "${RUN_ROOT}/jobs.tsv" \
  "${jobs_tsv_sha}" "${DAG_TOKEN}" "${OWNER_MANIFEST}" \
  "${OWNER_MANIFEST_SHA256}" "${GLOBAL_OWNER_MANIFEST}" \
  "${GLOBAL_OWNER_MANIFEST_SHA256}" "${RECOVERY_MODE}" "${PREFLIGHT_MANIFEST}" \
  "${PREFLIGHT_SHA256}" "${gate_job}" "${gate_sbatch}" \
  "${gate_sbatch_sha}" "${GATE_OUTPUT}" \
  "${jobs[selected_online]}" "${jobs[selected_ema]}" \
  "${jobs[physical_online]}" "${jobs[physical_ema]}" \
  "${suite_job}" "${suite_sbatch}" "${suite_sbatch_sha}" \
  "${suite_dependency}" "${LOG_ROOT}" "${FREE_KB}" "${MIN_FREE_KB}" \
  "${SCHEDULER_SUBMISSION}" "${SCHEDULER_SUBMISSION_SHA256}" <<'PY'
import json
import os
import sys
from pathlib import Path

(
    output,
    runtime_commit,
    runtime_tree,
    source_commit,
    source_tree,
    source_root,
    p0_run_root,
    p0_suite,
    videomae,
    run_root,
    jobs_tsv,
    jobs_tsv_sha,
    dag_token,
    owner_manifest,
    owner_manifest_sha,
    global_owner_manifest,
    global_owner_manifest_sha,
    recovery_mode,
    preflight,
    preflight_sha,
    gate_job,
    gate_sbatch,
    gate_sbatch_sha,
    gate_output,
    selected_online,
    selected_ema,
    physical_online,
    physical_ema,
    suite_job,
    suite_sbatch,
    suite_sbatch_sha,
    suite_dependency,
    log_root,
    free_kib,
    minimum_free_kib,
    scheduler_submission,
    scheduler_submission_sha,
) = sys.argv[1:]

payload = {
    "schema_version": "phystime_decode_cross_deployment_v1",
    "track": "frozen_epoch59_same_raw_tensor_dual_axis_decode",
    "runtime_commit": runtime_commit,
    "runtime_tree": runtime_tree,
    "source_commit": source_commit,
    "source_tree": source_tree,
    "source_root": source_root,
    "p0_run_root": p0_run_root,
    "p0_suite_completion": p0_suite,
    "videomae_checkpoint": videomae,
    "run_root": run_root,
    "dag_token": dag_token,
    "submission_owner_manifest": owner_manifest,
    "submission_owner_manifest_sha256": owner_manifest_sha,
    "global_submission_owner_manifest": global_owner_manifest,
    "global_submission_owner_manifest_sha256": global_owner_manifest_sha,
    "submission_recovery_mode": recovery_mode == "1",
    "preflight_manifest": preflight,
    "preflight_manifest_sha256": preflight_sha,
    "jobs_tsv": jobs_tsv,
    "jobs_tsv_sha256": jobs_tsv_sha,
    "scheduler_submission": scheduler_submission,
    "scheduler_submission_sha256": scheduler_submission_sha,
    "gate_job": gate_job,
    "gate_sbatch": gate_sbatch,
    "gate_sbatch_sha256": gate_sbatch_sha,
    "gate_stdout": str(Path(log_root, f"pt_dc_gate_{gate_job}.out")),
    "gate_stderr": str(Path(log_root, f"pt_dc_gate_{gate_job}.err")),
    "gate_output": gate_output,
    "jobs": {
        "selected_online": selected_online,
        "selected_ema": selected_ema,
        "physical_online": physical_online,
        "physical_ema": physical_ema,
        "decode_cross_suite": suite_job,
    },
    "suite_job": suite_job,
    "suite_sbatch": suite_sbatch,
    "suite_sbatch_sha256": suite_sbatch_sha,
    "suite_dependency": suite_dependency,
    "suite_output": str(Path(run_root, "DECODE_CROSS_SUITE_COMPLETE.json")),
    "new_training": False,
    "frozen_epoch": 59,
    "seed": 42,
    "arms": ["selected_axis", "physical_metric"],
    "weights_sources": ["online", "ema"],
    "decode_modes": ["uniform_rank_seconds", "physical_time_seconds"],
    "native_exact_equivalence_required": True,
    "submission_free_space_kib": int(free_kib),
    "minimum_free_space_kib": int(minimum_free_kib),
}
path = Path(output)
temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
with temporary.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
PY

echo "[PhysTime decode cross submit] RUN_ROOT=${RUN_ROOT}"
cat "${RUN_ROOT}/jobs.tsv"
submission_complete=1
trap - EXIT
