#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_RIME_PHASE4_CONTROLLER][FAIL] $*" >&2
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
  DUCA_RIME_PHASE3_SEAL_ROOT \
  DUCA_RIME_PHASE4_CELLS_ROOT \
  DUCA_RIME_PHASE4_SUBMISSION_ROOT \
  DUCA_RIME_PHASE2_RECEIPT \
  DUCA_RIME_PHASE2_RECEIPT_SHA256 \
  DUCA_RIME_PHASE2_PROTOCOL_ROOT \
  DUCA_RIME_SPLIT_MANIFEST \
  DUCA_RIME_SPLIT_MANIFEST_SHA256 \
  DUCA_RIME_TARGETS_JSONL \
  DUCA_RIME_TARGETS_SHA256 \
  DUCA_RIME_PHASE3_ASSET_RECEIPT \
  DUCA_RIME_PHASE3_ASSET_RECEIPT_SHA256 \
  DUCA_RIME_PRETRAIN_PATH \
  DUCA_RIME_PRETRAIN_SHA256 \
  DUCA_RIME_CANDIDATE_BUDGETS \
  DUCA_RIME_SHORT_MAX_SECONDS \
  DUCA_RIME_MEDIUM_MAX_SECONDS \
  DUCA_RIME_DENSE_CONFIG_ACTIONFORMER \
  DUCA_RIME_DENSE_CHECKPOINT_ACTIONFORMER \
  DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_ACTIONFORMER \
  DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_SHA256_ACTIONFORMER \
  DUCA_RIME_DENSE_TRAINED_COMMIT_ACTIONFORMER \
  DUCA_RIME_DENSE_CONFIG_TRIDET \
  DUCA_RIME_DENSE_CHECKPOINT_TRIDET \
  DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_TRIDET \
  DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_SHA256_TRIDET \
  DUCA_RIME_DENSE_TRAINED_COMMIT_TRIDET; do
  required "${name}"
  export "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] \
  || fail "the Phase-4 submit controller must run inside Slurm"
cd "${DUCA_RIME_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_RIME_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"

authorization="${DUCA_RIME_PHASE3_SEAL_ROOT}/phase4_authorization.json"
[[ -f "${authorization}" ]] \
  || fail "Phase-3 did not authorize the formal matrix"
export DUCA_RIME_PHASE4_AUTHORIZATION="${authorization}"
export DUCA_RIME_PHASE4_AUTHORIZATION_SHA256="$(
  sha256sum "${authorization}" | awk '{print $1}'
)"
check_sha256 \
  "${DUCA_RIME_PHASE2_RECEIPT}" \
  "${DUCA_RIME_PHASE2_RECEIPT_SHA256}" \
  "Phase-2 receipt"
check_sha256 \
  "${DUCA_RIME_PHASE3_ASSET_RECEIPT}" \
  "${DUCA_RIME_PHASE3_ASSET_RECEIPT_SHA256}" \
  "Phase-3 asset receipt"

export DUCA_RIME_SUBMIT_CONTROLLER=1
scripts/submit_duca_rime_phase4_matrix.sh

python - \
  "${DUCA_RIME_PHASE4_SUBMISSION_ROOT}/submission_manifest.json" \
  "${DUCA_RIME_PHASE4_SUBMISSION_ROOT}/controller_receipt.json" \
  "${SLURM_JOB_ID}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1]).resolve()
payload = json.loads(manifest.read_text(encoding="utf-8"))
if payload.get("cell_count") != 12:
    raise SystemExit("Phase-4 controller did not submit the frozen 12-cell matrix")
receipt = {
    "schema_version": "duca_rime_phase4_controller_receipt_v1",
    "status": "submitted",
    "controller_slurm_job_id": sys.argv[3],
    "submission_manifest_path": str(manifest),
    "submission_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "cell_job_ids": [row["slurm_job_id"] for row in payload["cells"]],
    "seal_job_id": payload["seal_job_id"],
}
target = pathlib.Path(sys.argv[2]).resolve()
temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
PY
echo "[DUCA_RIME_PHASE4_CONTROLLER] SUBMITTED frozen 12-cell matrix"
