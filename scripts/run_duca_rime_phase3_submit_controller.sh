#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE3_CONTROLLER][FAIL] $*" >&2
  exit 1
}

required() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "${name} is required"
}

check_sha256() {
  local path="$1" expected="$2" label="$3"
  [[ -f "${path}" ]] || fail "${label} is missing: ${path}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] \
    || fail "${label} SHA-256 drift"
}

for name in \
  DUCA_RIME_REPO_ROOT \
  DUCA_RIME_EXPECTED_COMMIT \
  DUCA_RIME_PHASE3_CONTROLLER_ROOT \
  DUCA_RIME_PHASE2_PIPELINE_RECEIPT \
  DUCA_RIME_PHASE3_ASSET_ROOT \
  DUCA_RIME_PHASE3_BUNDLE_ROOT \
  DUCA_RIME_PHASE4_CELLS_ROOT \
  DUCA_RIME_PHASE4_SUBMISSION_ROOT \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_PHASE2_MIXED_K_CONFIG \
  DUCA_RIME_PRETRAIN_PATH \
  DUCA_RIME_PRETRAIN_SHA256 \
  DUCA_RIME_CANDIDATE_BUDGETS \
  DUCA_RIME_SHORT_MAX_SECONDS \
  DUCA_RIME_MEDIUM_MAX_SECONDS \
  DUCA_RIME_DENSE_CONFIG_ACTIONFORMER \
  DUCA_RIME_DENSE_CHECKPOINT_ACTIONFORMER \
  DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_ACTIONFORMER \
  DUCA_RIME_DENSE_TRAINED_COMMIT_ACTIONFORMER \
  DUCA_RIME_DENSE_CONFIG_TRIDET \
  DUCA_RIME_DENSE_CHECKPOINT_TRIDET \
  DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_TRIDET \
  DUCA_RIME_DENSE_TRAINED_COMMIT_TRIDET; do
  required "${name}"
  export "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] \
  || fail "the Phase-3 submit controller must run inside Slurm"
[[ ! -e "${DUCA_RIME_PHASE3_CONTROLLER_ROOT}" ]] \
  || fail "a fresh Phase-3 controller root is required"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"
[[ -f "${DUCA_RIME_PHASE2_PIPELINE_RECEIPT}" ]] \
  || fail "Phase-2 pipeline receipt is missing"
export DUCA_RIME_PHASE2_PIPELINE_RECEIPT_SHA256="$(
  sha256sum "${DUCA_RIME_PHASE2_PIPELINE_RECEIPT}" | awk '{print $1}'
)"
for backend in ACTIONFORMER TRIDET; do
  evidence_var="DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_${backend}"
  sha_var="${evidence_var}_SHA256"
  [[ -f "${!evidence_var}" ]] \
    || fail "${backend} dense checkpoint evidence is missing"
  export "${sha_var}=$(sha256sum "${!evidence_var}" | awk '{print $1}')"
done
check_sha256 \
  "${DUCA_RIME_PHASE2_PIPELINE_RECEIPT}" \
  "${DUCA_RIME_PHASE2_PIPELINE_RECEIPT_SHA256}" \
  "Phase-2 pipeline receipt"
check_sha256 \
  "${DUCA_RIME_SPLIT_MANIFEST}" \
  "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  "split manifest"

readarray -t phase2_values < <(
  python - \
    "${DUCA_RIME_PHASE2_PIPELINE_RECEIPT}" \
    "${DUCA_RIME_EXPECTED_COMMIT}" <<'PY'
import hashlib
import json
import pathlib
import sys

pipeline = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    pipeline.get("schema_version")
    != "duca_rime_phase2_evidence_pipeline_receipt_v1"
    or pipeline.get("status") != "passed"
    or pipeline.get("git_commit") != sys.argv[2]
    or pipeline.get("uses_official_final") is not False
):
    raise SystemExit("invalid Phase-2 pipeline receipt")
for key in ("phase2_receipt", "crossfit_summary"):
    binding = pipeline["artifacts"][key]
    path = pathlib.Path(binding["path"]).resolve()
    if (
        not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]
    ):
        raise SystemExit(f"Phase-2 pipeline binding drift: {key}")
    print(path)
    print(binding["sha256"])

phase2 = json.loads(
    pathlib.Path(pipeline["artifacts"]["phase2_receipt"]["path"]).read_text(
        encoding="utf-8"
    )
)
protocols = {
    int(float(row["target_mean_cost"])): row
    for row in phase2["formal_budget_protocols"]
}
protocol = protocols[384]
path = pathlib.Path(protocol["path"]).resolve()
if (
    not path.is_file()
    or hashlib.sha256(path.read_bytes()).hexdigest() != protocol["sha256"]
):
    raise SystemExit("K384 protocol binding drift")
print(path)
print(protocol["sha256"])
PY
)
[[ "${#phase2_values[@]}" == 6 ]] \
  || fail "failed to resolve Phase-2 terminal evidence"
export DUCA_RIME_PHASE2_RECEIPT="${phase2_values[0]}"
export DUCA_RIME_PHASE2_RECEIPT_SHA256="${phase2_values[1]}"
export DUCA_RIME_PHASE2_CROSSFIT_SUMMARY="${phase2_values[2]}"
export DUCA_RIME_PHASE2_CROSSFIT_SUMMARY_SHA256="${phase2_values[3]}"
export DUCA_RIME_BUDGET_PROTOCOL_JSON="${phase2_values[4]}"
export DUCA_RIME_BUDGET_PROTOCOL_SHA256="${phase2_values[5]}"
export DUCA_RIME_PHASE2_PROTOCOL_ROOT="$(dirname "${DUCA_RIME_BUDGET_PROTOCOL_JSON}")"

mkdir -p "${DUCA_RIME_PHASE3_CONTROLLER_ROOT}"
scripts/run_duca_rime_phase3_asset_producer.sh

export DUCA_RIME_TARGETS_JSONL="${DUCA_RIME_PHASE3_ASSET_ROOT}/training_targets.jsonl"
export DUCA_RIME_TARGETS_SHA256="$(
  sha256sum "${DUCA_RIME_TARGETS_JSONL}" | awk '{print $1}'
)"
export DUCA_RIME_DSHUFFLE_TRAIN_REPLAY_JSONL="${DUCA_RIME_PHASE3_ASSET_ROOT}/dshuffle_training_replay.jsonl"
export DUCA_RIME_DSHUFFLE_TRAIN_REPLAY_SHA256="$(
  sha256sum "${DUCA_RIME_DSHUFFLE_TRAIN_REPLAY_JSONL}" | awk '{print $1}'
)"
export DUCA_RIME_ADAPTOK_REPLAY_JSONL="${DUCA_RIME_PHASE3_ASSET_ROOT}/adaptok_replay.jsonl"
export DUCA_RIME_ADAPTOK_REPLAY_SHA256="$(
  sha256sum "${DUCA_RIME_ADAPTOK_REPLAY_JSONL}" | awk '{print $1}'
)"
export DUCA_RIME_PHASE3_TRAINING_EXPOSURE_JSON="${DUCA_RIME_PHASE3_ASSET_ROOT}/training_exposure.json"
export DUCA_RIME_PHASE3_TRAINING_EXPOSURE_SHA256="$(
  sha256sum "${DUCA_RIME_PHASE3_TRAINING_EXPOSURE_JSON}" | awk '{print $1}'
)"
export DUCA_RIME_PHASE3_ASSET_RECEIPT="${DUCA_RIME_PHASE3_ASSET_ROOT}/production_receipt.json"
export DUCA_RIME_PHASE3_ASSET_RECEIPT_SHA256="$(
  sha256sum "${DUCA_RIME_PHASE3_ASSET_RECEIPT}" | awk '{print $1}'
)"

export DUCA_RIME_SUBMIT_CONTROLLER=1
scripts/submit_duca_rime_phase3.sh
phase3_manifest="${DUCA_RIME_PHASE3_BUNDLE_ROOT}/submission_manifest.json"
phase3_seal_job="$(
  python - "${phase3_manifest}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["seal_job_id"])
PY
)"
export DUCA_RIME_PHASE3_SEAL_ROOT="${DUCA_RIME_PHASE3_BUNDLE_ROOT}/seal"

phase4_controller_job="$(
  sbatch \
    --parsable \
    --partition=gpu \
    --cpus-per-task=2 \
    --time=00:30:00 \
    --job-name=rime4-controller \
    --dependency="afterok:${phase3_seal_job}" \
    --output="${DUCA_RIME_PHASE3_CONTROLLER_ROOT}/phase4-controller-%j.out" \
    --export=ALL \
    scripts/run_duca_rime_phase4_submit_controller.sh
)"
phase4_controller_job="${phase4_controller_job%%;*}"

python - \
  "${DUCA_RIME_PHASE3_CONTROLLER_ROOT}/controller_receipt.json" \
  "${DUCA_RIME_EXPECTED_COMMIT}" \
  "${SLURM_JOB_ID}" \
  "${phase3_manifest}" \
  "${phase4_controller_job}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1]).resolve()
manifest = pathlib.Path(sys.argv[4]).resolve()
payload = {
    "schema_version": "duca_rime_phase3_controller_receipt_v1",
    "status": "submitted",
    "git_commit": sys.argv[2],
    "controller_slurm_job_id": sys.argv[3],
    "phase3_submission_manifest_path": str(manifest),
    "phase3_submission_manifest_sha256": hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest(),
    "phase4_controller_job_id": sys.argv[5],
}
temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
PY
echo \
  "[DUCA_RIME_PHASE3_CONTROLLER] SUBMITTED Phase-3 and Phase-4 controller ${phase4_controller_job}"
