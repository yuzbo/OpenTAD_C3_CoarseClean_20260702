#!/usr/bin/env bash
set -Eeuo pipefail

fail() {
  echo "[DUCA_PAPER_RELEASE_GATES][FAIL] $*" >&2
  exit 1
}

required() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "${name} is required"
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/duca_cellcf_path_contract.sh"
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"

for name in \
  DUCA_PAPER_EXPECTED_COMMIT \
  DUCA_PAPER_RELEASE_GATE_ROOT \
  DUCA_PAPER_CODE_GATE_RECEIPT \
  DUCA_PAPER_CODE_GATE_RECEIPT_SHA256 \
  DUCA_PAPER_SHORT_WINDOW_GATE_JSON \
  DUCA_PAPER_SHORT_WINDOW_GATE_SHA256 \
  DUCA_PAPER_PRETRAIN_SHA256 \
  DUCA_PAPER_ANNOTATION_SHA256 \
  DUCA_PAPER_CLASS_MAP_SHA256; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "release gates must run inside Slurm"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not expose logical GPUs"
[[ "${DUCA_PAPER_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected commit is required"
[[ "$(git rev-parse HEAD)" == "${DUCA_PAPER_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "release gates require a clean tree"

DUCA_PAPER_RELEASE_GATE_ROOT="$(
  duca_cellcf_require_external_path \
    "DUCA_PAPER_RELEASE_GATE_ROOT" \
    "${REPO_ROOT}" \
    "${BASE}" \
    "${DUCA_PAPER_RELEASE_GATE_ROOT}"
)" || fail "release-gate root violates the formal path contract"
[[ ! -e "${DUCA_PAPER_RELEASE_GATE_ROOT}" ]] \
  || fail "a fresh release-gate root is required"
mkdir -p "${DUCA_PAPER_RELEASE_GATE_ROOT}"

"${PYTHON}" -m tools.bata.validate_duca_paper_code_gate \
  --receipt "${DUCA_PAPER_CODE_GATE_RECEIPT}" \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --expected-sha256 "${DUCA_PAPER_CODE_GATE_RECEIPT_SHA256}"
"${PYTHON}" -m tools.bata.validate_duca_paper_short_window_gate \
  --receipt "${DUCA_PAPER_SHORT_WINDOW_GATE_JSON}" \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --expected-sha256 "${DUCA_PAPER_SHORT_WINDOW_GATE_SHA256}"

numeric_root="${DUCA_PAPER_RELEASE_GATE_ROOT}/numeric"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
"${PYTHON}" -m torch.distributed.run \
  --rdzv-backend=c10d \
  --rdzv-endpoint=localhost:0 \
  --rdzv-id="${SLURM_JOB_ID}-duca-paper-numeric" \
  --nproc_per_node=2 \
  -m tools.bata.run_duca_paper_numeric_gate \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --code-gate-receipt "${DUCA_PAPER_CODE_GATE_RECEIPT}" \
  --code-gate-receipt-sha256 "${DUCA_PAPER_CODE_GATE_RECEIPT_SHA256}" \
  --short-window-gate "${DUCA_PAPER_SHORT_WINDOW_GATE_JSON}" \
  --short-window-gate-sha256 "${DUCA_PAPER_SHORT_WINDOW_GATE_SHA256}" \
  --pretrain "${ADATAD_PRETRAIN_PATH}" \
  --pretrain-sha256 "${DUCA_PAPER_PRETRAIN_SHA256}" \
  --annotation "${THUMOS14_ANNOTATION_PATH}" \
  --annotation-sha256 "${DUCA_PAPER_ANNOTATION_SHA256}" \
  --class-map "${THUMOS14_CLASS_MAP}" \
  --class-map-sha256 "${DUCA_PAPER_CLASS_MAP_SHA256}" \
  --train-data-path "${THUMOS14_TRAIN_DATA_PATH}" \
  --output-root "${numeric_root}"

numeric_receipt="${numeric_root}/numeric_gate.receipt.json"
numeric_sha256="$(sha256sum "${numeric_receipt}" | awk '{print $1}')"
"${PYTHON}" -m tools.bata.validate_duca_paper_numeric_gate \
  --receipt "${numeric_receipt}" \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --expected-sha256 "${numeric_sha256}"

exact211_receipt="${DUCA_PAPER_RELEASE_GATE_ROOT}/exact211_uid_gate.receipt.json"
"${PYTHON}" -m tools.bata.run_duca_paper_exact211_uid_gate \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --numeric-gate "${numeric_receipt}" \
  --numeric-gate-sha256 "${numeric_sha256}" \
  --annotation "${THUMOS14_ANNOTATION_PATH}" \
  --annotation-sha256 "${DUCA_PAPER_ANNOTATION_SHA256}" \
  --class-map "${THUMOS14_CLASS_MAP}" \
  --class-map-sha256 "${DUCA_PAPER_CLASS_MAP_SHA256}" \
  --test-data-path "${THUMOS14_TEST_DATA_PATH}" \
  --output-json "${exact211_receipt}"
exact211_sha256="$(sha256sum "${exact211_receipt}" | awk '{print $1}')"
"${PYTHON}" -m tools.bata.validate_duca_paper_exact211_uid_gate \
  --receipt "${exact211_receipt}" \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --expected-sha256 "${exact211_sha256}"

"${PYTHON}" - \
  "${DUCA_PAPER_RELEASE_GATE_ROOT}/release_gates.receipt.json" \
  "${DUCA_PAPER_EXPECTED_COMMIT}" \
  "${SLURM_JOB_ID}" \
  "${DUCA_PAPER_CODE_GATE_RECEIPT}" \
  "${DUCA_PAPER_CODE_GATE_RECEIPT_SHA256}" \
  "${DUCA_PAPER_SHORT_WINDOW_GATE_JSON}" \
  "${DUCA_PAPER_SHORT_WINDOW_GATE_SHA256}" \
  "${numeric_receipt}" \
  "${numeric_sha256}" \
  "${exact211_receipt}" \
  "${exact211_sha256}" <<'PY'
import json
import os
import pathlib
import sys

(
    output,
    commit,
    job_id,
    code_path,
    code_sha,
    short_path,
    short_sha,
    numeric_path,
    numeric_sha,
    uid_path,
    uid_sha,
) = sys.argv[1:]
payload = {
    "schema_version": "duca_paper_stage_a_release_gates_v1",
    "status": "passed",
    "git_commit": commit,
    "slurm_job_id": job_id,
    "code_gate_path": str(pathlib.Path(code_path).resolve()),
    "code_gate_sha256": code_sha,
    "short_window_gate_path": str(pathlib.Path(short_path).resolve()),
    "short_window_gate_sha256": short_sha,
    "numeric_gate_path": str(pathlib.Path(numeric_path).resolve()),
    "numeric_gate_sha256": numeric_sha,
    "exact211_uid_gate_path": str(pathlib.Path(uid_path).resolve()),
    "exact211_uid_gate_sha256": uid_sha,
    "paper_metric_claim_allowed": False,
    "paper_method_performance_evidence": False,
    "stage_a_release_prerequisites_satisfied": True,
    "stage_b_enabled": False,
    "official_final_consumed": False,
}
target = pathlib.Path(output)
with target.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
sha256sum "${DUCA_PAPER_RELEASE_GATE_ROOT}/release_gates.receipt.json" \
  > "${DUCA_PAPER_RELEASE_GATE_ROOT}/release_gates.receipt.sha256"

echo "[DUCA_PAPER_RELEASE_GATES] PASS ${DUCA_PAPER_RELEASE_GATE_ROOT}/release_gates.receipt.json"
