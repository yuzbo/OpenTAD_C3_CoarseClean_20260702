#!/usr/bin/env bash
set -Eeuo pipefail

fail() {
  echo "[DUCA_PAPER_CODE_GATE][FAIL] $*" >&2
  exit 1
}

required() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "${name} is required"
}

for name in \
  DUCA_PAPER_REPO_ROOT \
  DUCA_PAPER_EXPECTED_COMMIT \
  DUCA_PAPER_CODE_GATE_ROOT \
  DUCA_PAPER_PRETRAIN_PATH \
  DUCA_PAPER_PRETRAIN_SHA256 \
  DUCA_PAPER_ANNOTATION_PATH \
  DUCA_PAPER_ANNOTATION_SHA256 \
  DUCA_PAPER_CLASS_MAP_PATH \
  DUCA_PAPER_CLASS_MAP_SHA256; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "the paper code gate must run inside Slurm"
[[ "${DUCA_PAPER_EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
  || fail "an exact expected commit is required"
[[ ! -e "${DUCA_PAPER_CODE_GATE_ROOT}" ]] \
  || fail "a fresh code-gate root is required"
mkdir -p "${DUCA_PAPER_CODE_GATE_ROOT}/logs"

code_gate_step="preflight"
write_code_gate_failure() {
  local rc="$1"
  local line="$2"
  local command="$3"
  trap - ERR INT TERM
  python - \
    "${DUCA_PAPER_CODE_GATE_ROOT}/gate.failure.receipt.json" \
    "${DUCA_PAPER_EXPECTED_COMMIT}" \
    "${SLURM_JOB_ID:-}" \
    "${code_gate_step}" \
    "${rc}" \
    "${line}" \
    "${command}" <<'PY' || true
import hashlib
import json
import os
import pathlib
import sys

output, commit, job_id, step, rc, line, command = sys.argv[1:]
payload = {
    "schema_version": "duca_paper_clean_linux_code_gate_failure_v1",
    "status": "failed",
    "fail_closed": True,
    "expected_commit": commit,
    "slurm_job_id": job_id,
    "failure_step": step,
    "exit_code": int(rc),
    "shell_line": int(line),
    "shell_command": command[:2000],
    "paper_metric_claim_allowed": False,
    "paper_method_performance_evidence": False,
    "stage_a_released": False,
    "stage_b_enabled": False,
    "official_final_consumed": False,
    "claim_scope": "engineering_code_gate_failure_only",
}
unsigned = json.dumps(
    payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
).encode("ascii")
payload["content_sha256"] = hashlib.sha256(unsigned).hexdigest()
target = pathlib.Path(output)
if not target.exists():
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
PY
  exit "${rc}"
}
trap 'write_code_gate_failure "$?" "${LINENO}" "${BASH_COMMAND}"' ERR
trap 'write_code_gate_failure 130 "${LINENO}" "signal"' INT TERM

cd "${DUCA_PAPER_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_PAPER_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"

for binding in \
  "${DUCA_PAPER_PRETRAIN_PATH}|${DUCA_PAPER_PRETRAIN_SHA256}|VideoMAE initialization" \
  "${DUCA_PAPER_ANNOTATION_PATH}|${DUCA_PAPER_ANNOTATION_SHA256}|THUMOS14 annotation" \
  "${DUCA_PAPER_CLASS_MAP_PATH}|${DUCA_PAPER_CLASS_MAP_SHA256}|THUMOS14 class map"; do
  IFS='|' read -r path expected label <<<"${binding}"
  [[ -f "${path}" ]] || fail "${label} is missing: ${path}"
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] \
    || fail "${label} SHA-256 drift"
done

code_gate_step="compile_and_shell_contracts"
python -m py_compile \
  tools/train.py \
  tools/test.py \
  tools/bata/duca_paper_training.py \
  tools/bata/build_duca_paper_matrix_manifest.py \
  tools/bata/validate_duca_paper_code_gate.py \
  tools/bata/run_duca_paper_short_window_gate.py \
  tools/bata/validate_duca_paper_short_window_gate.py \
  tools/bata/run_duca_paper_numeric_gate.py \
  tools/bata/validate_duca_paper_numeric_gate.py \
  tools/bata/run_duca_paper_legacy_numeric_regression.py \
  tools/bata/validate_duca_paper_legacy_numeric_regression.py \
  tools/bata/run_duca_paper_exact211_uid_gate.py \
  tools/bata/validate_duca_paper_exact211_uid_gate.py \
  tools/bata/validate_duca_paper_release_gates.py \
  opentad/models/backbones/backbone_wrapper.py \
  opentad/models/detectors/actionformer.py \
  opentad/models/detectors/single_stage.py \
  opentad/models/selectors/duca_protected_e2e_frame_selector.py \
  opentad/models/selectors/duca_rime_frame_selector.py
bash -n \
  scripts/run_duca_paper_code_gate.sh \
  scripts/run_duca_paper_short_window_gate.sh \
  scripts/run_duca_paper_release_gates.sh \
  scripts/run_duca_paper_stage_a_cell.sh \
  scripts/run_duca_paper_stage_a_seed.sh \
  scripts/run_duca_paper_stage_a_seal.sh \
  scripts/submit_duca_paper_stage_a.sh \
  scripts/submit_duca_paper_stage_a_grouped.sh

code_gate_step="deterministic_legacy_numeric_regression"
legacy_regression="${DUCA_PAPER_CODE_GATE_ROOT}/legacy_numeric_regression.receipt.json"
python -m tools.bata.run_duca_paper_legacy_numeric_regression \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --output "${legacy_regression}"
legacy_regression_sha256="$(sha256sum "${legacy_regression}" | awk '{print $1}')"
python -m tools.bata.validate_duca_paper_legacy_numeric_regression \
  --receipt "${legacy_regression}" \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --expected-sha256 "${legacy_regression_sha256}"

code_gate_step="focused_pytest"
python -m pytest \
  tests/test_duca_paper_full200_contract.py \
  tests/test_duca_rime_backbone_mask_contract.py \
  tests/test_duca_protected_e2e_detector_contract.py \
  tests/test_duca_protected_e2e_frame_selector.py \
  tests/test_duca_rime.py \
  tests/test_duca_structured_selection.py \
  tests/test_train_engine_after_optimizer_step.py \
  tests/test_train_engine_max_train_iters.py \
  -q 2>&1 | tee "${DUCA_PAPER_CODE_GATE_ROOT}/logs/pytest.out"

code_gate_step="success_receipt"
python - \
  "${DUCA_PAPER_CODE_GATE_ROOT}/gate.receipt.json" \
  "${DUCA_PAPER_EXPECTED_COMMIT}" \
  "${SLURM_JOB_ID}" \
  "${DUCA_PAPER_CODE_GATE_ROOT}/logs/pytest.out" \
  "${legacy_regression}" \
  "${legacy_regression_sha256}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

output, commit, job_id, pytest_log, legacy_path, legacy_sha = sys.argv[1:]
sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
payload = {
    "schema_version": "duca_paper_clean_linux_code_gate_v3",
    "status": "passed",
    "fail_closed": True,
    "git_commit": commit,
    "slurm_job_id": job_id,
    "pytest_log_path": str(pathlib.Path(pytest_log).resolve()),
    "pytest_log_sha256": sha(pytest_log),
    "official_train_video_count": 200,
    "official_evaluation_video_count": 211,
    "stage_a_logical_cell_count": 12,
    "short_window_gate_pending": True,
    "stage_a_manifest_created": False,
    "stage_a_released": False,
    "stage_b_enabled": False,
    "paper_metric_claim_allowed": False,
    "legacy_numeric_regression": {
        "path": str(pathlib.Path(legacy_path).resolve()),
        "sha256": legacy_sha,
    },
}
unsigned = json.dumps(
    payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
).encode("ascii")
payload["content_sha256"] = hashlib.sha256(unsigned).hexdigest()
target = pathlib.Path(output)
with target.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
sha256sum "${DUCA_PAPER_CODE_GATE_ROOT}/gate.receipt.json" \
  > "${DUCA_PAPER_CODE_GATE_ROOT}/gate.receipt.sha256"
trap - ERR INT TERM
echo "[DUCA_PAPER_CODE_GATE] PASS ${DUCA_PAPER_CODE_GATE_ROOT}/gate.receipt.json"
