#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE2_PIPELINE][FAIL] $*" >&2
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
  DUCA_RIME_PHASE2_PIPELINE_ROOT \
  DUCA_RIME_PHASE1_PIPELINE_RECEIPT \
  DUCA_RIME_PHASE2_MIXED_K_CONFIG \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_PRETRAIN_PATH \
  DUCA_RIME_PRETRAIN_SHA256 \
  DUCA_RIME_CANDIDATE_BUDGETS \
  DUCA_RIME_CANDIDATE_COSTS \
  DUCA_RIME_TARGET_MEAN_COST \
  DUCA_RIME_PHASE4_SECOND_TARGET_MEAN_COST \
  DUCA_RIME_DECODER_FAMILY \
  DUCA_RIME_RISK_WEIGHT \
  DUCA_RIME_RISK_THRESHOLD \
  DUCA_RIME_O4_MAX_BRIER \
  DUCA_RIME_O4_MAX_ECE \
  DUCA_RIME_O4_MIN_COVERAGE \
  DUCA_RIME_O4_MAX_LOW_RISK_FAILURE \
  DUCA_RIME_SHORT_MAX_SECONDS \
  DUCA_RIME_MEDIUM_MAX_SECONDS; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] \
  || fail "Phase-2 training/evidence must run inside Slurm"
[[ ! -e "${DUCA_RIME_PHASE2_PIPELINE_ROOT}" ]] \
  || fail "a fresh Phase-2 pipeline root is required"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"
check_sha256 \
  "${DUCA_RIME_SPLIT_MANIFEST}" \
  "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  "split manifest"
check_sha256 \
  "${DUCA_RIME_PRETRAIN_PATH}" \
  "${DUCA_RIME_PRETRAIN_SHA256}" \
  "VideoMAE pretrain"

readarray -t phase1_values < <(
  python - \
    "${DUCA_RIME_PHASE1_PIPELINE_RECEIPT}" \
    "${DUCA_RIME_EXPECTED_COMMIT}" <<'PY'
import hashlib
import json
import pathlib
import sys

pipeline = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    pipeline.get("schema_version")
    != "duca_rime_phase1_evidence_pipeline_receipt_v1"
    or pipeline.get("status") != "passed"
    or pipeline.get("git_commit") != sys.argv[2]
    or pipeline.get("official_final_subset_consumed") is not False
):
    raise SystemExit("invalid Phase-1 pipeline receipt")
path = pathlib.Path(pipeline["phase1_receipt_path"]).resolve()
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if digest != pipeline["phase1_receipt_sha256"]:
    raise SystemExit("Phase-1 stage receipt binding drift")
receipt = json.loads(path.read_text(encoding="utf-8"))
if (
    receipt.get("phase") != "phase1"
    or receipt.get("gate_pass") is not True
    or receipt.get("phase2_authorized") is not True
):
    raise SystemExit("Phase-1 did not authorize Phase-2")
print(path)
print(digest)
PY
)
[[ "${#phase1_values[@]}" == 2 ]] \
  || fail "failed to resolve the Phase-1 receipt"
export DUCA_RIME_PHASE1_RECEIPT="${phase1_values[0]}"
export DUCA_RIME_PHASE1_RECEIPT_SHA256="${phase1_values[1]}"

readarray -t split_values < <(
  python - "${DUCA_RIME_SPLIT_MANIFEST}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for role in ("detector_selector_train", "certification_development"):
    row = manifest["train_roles"][role]
    print(row["block_list_path"])
    print(row["block_list_sha256"])
PY
)
[[ "${#split_values[@]}" == 4 ]] \
  || fail "failed to resolve Phase-2 train/development roles"
check_sha256 "${split_values[0]}" "${split_values[1]}" "train block list"
check_sha256 "${split_values[2]}" "${split_values[3]}" "development block list"
export DUCA_RIME_TRAIN_BLOCK_LIST="${split_values[0]}"
export DUCA_RIME_DEVELOPMENT_BLOCK_LIST="${split_values[2]}"

mkdir -p "${DUCA_RIME_PHASE2_PIPELINE_ROOT}"
exposure="${DUCA_RIME_PHASE2_PIPELINE_ROOT}/training_exposure.json"
python tools/bata/create_duca_rime_training_exposure.py \
  --repo-root "${DUCA_RIME_REPO_ROOT}" \
  --expected-commit "${DUCA_RIME_EXPECTED_COMMIT}" \
  --config "${DUCA_RIME_PHASE2_MIXED_K_CONFIG}" \
  --split-manifest "${DUCA_RIME_SPLIT_MANIFEST}" \
  --split-manifest-sha256 "${DUCA_RIME_SPLIT_MANIFEST_SHA256}" \
  --output "${exposure}" \
  --research-phase 2 \
  --seed 3407 \
  --detector-backend ActionFormer \
  --target-mean-cost 384 \
  > "${DUCA_RIME_PHASE2_PIPELINE_ROOT}/training_exposure.receipt.json"

export DUCA_RIME_TRAINING_EXPOSURE_JSON="${exposure}"
export DUCA_RIME_TRAINING_EXPOSURE_SHA256="$(
  sha256sum "${exposure}" | awk '{print $1}'
)"
export DUCA_RIME_PHASE2_MIXED_K_ROOT="${DUCA_RIME_PHASE2_PIPELINE_ROOT}/mixed_k"
scripts/run_duca_rime_phase2_mixed_k_train.sh

export DUCA_RIME_TRAINING_RECEIPT="${DUCA_RIME_PHASE2_MIXED_K_ROOT}/training_receipt.json"
export DUCA_RIME_TRAINING_RECEIPT_SHA256="$(
  sha256sum "${DUCA_RIME_TRAINING_RECEIPT}" | awk '{print $1}'
)"
export DUCA_RIME_CHECKPOINT="${DUCA_RIME_PHASE2_MIXED_K_ROOT}/train/gpu1_id0/checkpoint/terminal_ema.pth"
export DUCA_RIME_CHECKPOINT_SHA256="$(
  sha256sum "${DUCA_RIME_CHECKPOINT}" | awk '{print $1}'
)"
export DUCA_RIME_PHASE2_EVIDENCE_ROOT="${DUCA_RIME_PHASE2_PIPELINE_ROOT}/evidence"
scripts/run_duca_rime_phase2_evidence_pipeline.sh

python - \
  "${DUCA_RIME_PHASE2_PIPELINE_ROOT}" \
  "${DUCA_RIME_EXPECTED_COMMIT}" \
  "${SLURM_JOB_ID}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
evidence = root / "evidence" / "pipeline_receipt.json"
training = root / "mixed_k" / "training_receipt.json"
for path in (evidence, training):
    if not path.is_file():
        raise SystemExit(f"missing Phase-2 terminal artifact: {path}")
payload = {
    "schema_version": "duca_rime_phase2_train_and_evidence_receipt_v1",
    "status": "passed",
    "git_commit": sys.argv[2],
    "slurm_job_id": sys.argv[3],
    "mixed_k_training_receipt": {
        "path": str(training),
        "sha256": hashlib.sha256(training.read_bytes()).hexdigest(),
    },
    "evidence_pipeline_receipt": {
        "path": str(evidence),
        "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
    },
    "uses_official_final": False,
}
target = root / "pipeline_receipt.json"
temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
PY
echo "[DUCA_RIME_PHASE2_PIPELINE] PASS ${DUCA_RIME_PHASE2_PIPELINE_ROOT}"
