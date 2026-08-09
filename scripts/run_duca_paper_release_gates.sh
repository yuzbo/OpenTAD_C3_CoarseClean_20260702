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

release_gate_step="prerequisite_validation"
write_release_gate_failure() {
  local rc="$1"
  local line="$2"
  local command="$3"
  trap - ERR INT TERM
  "${PYTHON}" - \
    "${DUCA_PAPER_RELEASE_GATE_ROOT}" \
    "${DUCA_PAPER_EXPECTED_COMMIT}" \
    "${SLURM_JOB_ID:-}" \
    "${release_gate_step}" \
    "${rc}" \
    "${line}" \
    "${command}" <<'PY' || true
import hashlib
import json
import os
import pathlib
import sys

root, commit, job_id, step, rc, line, command = sys.argv[1:]
root = pathlib.Path(root)
failure_class = (
    "WATCHDOG_OR_COLLECTIVE_FAILURE"
    if int(rc) in {124, 137, 143}
    else "INTERNAL_ERROR"
)

def write_new(path, payload):
    unsigned = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    payload["content_sha256"] = hashlib.sha256(unsigned).hexdigest()
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

common = {
    "status": "failed",
    "fail_closed": True,
    "failure_class": failure_class,
    "failure_step": step,
    "exit_code": int(rc),
    "shell_line": int(line),
    "shell_command": command[:2000],
    "expected_commit": commit,
    "slurm_job_id": job_id,
    "validation_or_test_data_used": False,
    "metric_accessed": False,
    "paper_metric_claim_allowed": False,
    "paper_method_performance_evidence": False,
    "stage_a_release_prerequisites_satisfied": False,
    "stage_b_enabled": False,
    "official_final_consumed": False,
}
release = dict(common)
release.update(
    schema_version="duca_paper_stage_a_release_gates_failure_v1",
    claim_scope="engineering_stage_a_release_gate_failure_only",
)
write_new(root / "release_gates.failure.receipt.json", release)
if step == "production_numeric_gate":
    numeric = dict(common)
    numeric.update(
        schema_version="duca_paper_physical_exactk_numeric_gate_failure_v1",
        stage_a_release_prerequisite_satisfied=False,
        claim_scope="engineering_numeric_gate_failure_only",
    )
    write_new(root / "numeric" / "numeric_gate.failure.receipt.json", numeric)
PY
  exit "${rc}"
}
trap 'write_release_gate_failure "$?" "${LINENO}" "${BASH_COMMAND}"' ERR
trap 'write_release_gate_failure 130 "${LINENO}" "signal"' INT TERM

"${PYTHON}" -m tools.bata.validate_duca_paper_code_gate \
  --receipt "${DUCA_PAPER_CODE_GATE_RECEIPT}" \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --expected-sha256 "${DUCA_PAPER_CODE_GATE_RECEIPT_SHA256}"
"${PYTHON}" -m tools.bata.validate_duca_paper_short_window_gate \
  --receipt "${DUCA_PAPER_SHORT_WINDOW_GATE_JSON}" \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --expected-sha256 "${DUCA_PAPER_SHORT_WINDOW_GATE_SHA256}"

release_gate_step="production_numeric_gate"
numeric_root="${DUCA_PAPER_RELEASE_GATE_ROOT}/numeric"
export DUCA_PAPER_NUMERIC_GATE_WALL_TIMEOUT_SECONDS=14400
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
timeout --signal=TERM --kill-after=60s \
  "${DUCA_PAPER_NUMERIC_GATE_WALL_TIMEOUT_SECONDS}s" \
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

release_gate_step="exact211_physical_uid_gate"
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

release_gate_step="aggregate_success_receipt"
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
    "schema_version": "duca_paper_stage_a_release_gates_v2",
    "status": "passed",
    "fail_closed": True,
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
unsigned = json.dumps(
    payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
).encode("ascii")
import hashlib
payload["content_sha256"] = hashlib.sha256(unsigned).hexdigest()
target = pathlib.Path(output)
with target.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
sha256sum "${DUCA_PAPER_RELEASE_GATE_ROOT}/release_gates.receipt.json" \
  > "${DUCA_PAPER_RELEASE_GATE_ROOT}/release_gates.receipt.sha256"
release_gates_sha256="$(
  sha256sum "${DUCA_PAPER_RELEASE_GATE_ROOT}/release_gates.receipt.json" \
    | awk '{print $1}'
)"
"${PYTHON}" -m tools.bata.validate_duca_paper_release_gates \
  --receipt "${DUCA_PAPER_RELEASE_GATE_ROOT}/release_gates.receipt.json" \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --expected-sha256 "${release_gates_sha256}"

trap - ERR INT TERM
echo "[DUCA_PAPER_RELEASE_GATES] PASS ${DUCA_PAPER_RELEASE_GATE_ROOT}/release_gates.receipt.json"
