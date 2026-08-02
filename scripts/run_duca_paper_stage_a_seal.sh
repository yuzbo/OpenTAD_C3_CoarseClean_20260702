#!/usr/bin/env bash
set -Eeuo pipefail

fail() {
  echo "[DUCA_PAPER_STAGE_A_SEAL][FAIL] $*" >&2
  exit 1
}

required() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "${name} is required"
}

for name in \
  DUCA_PAPER_REPO_ROOT \
  DUCA_PAPER_EXPECTED_COMMIT \
  DUCA_PAPER_CELLS_ROOT \
  DUCA_PAPER_MATRIX_ROOT \
  DUCA_PAPER_MATRIX_MANIFEST \
  DUCA_PAPER_MATRIX_MANIFEST_SHA256 \
  DUCA_PAPER_CODE_GATE_RECEIPT \
  DUCA_PAPER_CODE_GATE_RECEIPT_SHA256 \
  DUCA_PAPER_SHORT_WINDOW_GATE_JSON \
  DUCA_PAPER_SHORT_WINDOW_GATE_SHA256; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Stage-A sealing must run inside Slurm"
cd "${DUCA_PAPER_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_PAPER_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"
[[ ! -e "${DUCA_PAPER_MATRIX_ROOT}" ]] || fail "a fresh matrix root is required"
[[ "$(sha256sum "${DUCA_PAPER_MATRIX_MANIFEST}" | awk '{print $1}')" == \
  "${DUCA_PAPER_MATRIX_MANIFEST_SHA256}" ]] || fail "matrix manifest SHA-256 drift"
[[ "$(sha256sum "${DUCA_PAPER_SHORT_WINDOW_GATE_JSON}" | awk '{print $1}')" == \
  "${DUCA_PAPER_SHORT_WINDOW_GATE_SHA256}" ]] || fail "short-window gate SHA-256 drift"
[[ "$(sha256sum "${DUCA_PAPER_CODE_GATE_RECEIPT}" | awk '{print $1}')" == \
  "${DUCA_PAPER_CODE_GATE_RECEIPT_SHA256}" ]] || fail "code-gate SHA-256 drift"
python -m tools.bata.validate_duca_paper_code_gate \
  --receipt "${DUCA_PAPER_CODE_GATE_RECEIPT}" \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --expected-sha256 "${DUCA_PAPER_CODE_GATE_RECEIPT_SHA256}"
python -m tools.bata.validate_duca_paper_short_window_gate \
  --receipt "${DUCA_PAPER_SHORT_WINDOW_GATE_JSON}" \
  --expected-commit "${DUCA_PAPER_EXPECTED_COMMIT}" \
  --expected-sha256 "${DUCA_PAPER_SHORT_WINDOW_GATE_SHA256}"

mkdir -p "${DUCA_PAPER_MATRIX_ROOT}"
python - \
  "${DUCA_PAPER_MATRIX_MANIFEST}" \
  "${DUCA_PAPER_MATRIX_MANIFEST_SHA256}" \
  "${DUCA_PAPER_CELLS_ROOT}" \
  "${DUCA_PAPER_EXPECTED_COMMIT}" \
  "${DUCA_PAPER_CODE_GATE_RECEIPT}" \
  "${DUCA_PAPER_CODE_GATE_RECEIPT_SHA256}" \
  "${DUCA_PAPER_SHORT_WINDOW_GATE_JSON}" \
  "${DUCA_PAPER_SHORT_WINDOW_GATE_SHA256}" \
  "${DUCA_PAPER_MATRIX_ROOT}/matrix_receipt.json" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

(
    manifest_path,
    manifest_sha,
    cells_root,
    commit,
    code_gate_path,
    code_gate_sha,
    short_gate_path,
    short_gate_sha,
    output,
) = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
code_gate = manifest.get("prerequisite_gates", {}).get(
    "clean_linux_pytorch_code", {}
)
short_gate = manifest.get("prerequisite_gates", {}).get(
    "real_natural_short_window_heavy_backbone", {}
)
if (
    manifest.get("schema_version") != "duca_paper_full200_matrix_v2"
    or manifest.get("status") != "frozen"
    or manifest.get("git_commit") != commit
    or sha(manifest_path) != manifest_sha
    or manifest.get("partial_matrix_claim_allowed") is not False
    or manifest.get("single_seed_claim_allowed") is not False
    or code_gate.get("status") != "passed"
    or code_gate.get("git_commit") != commit
    or pathlib.Path(str(code_gate.get("path", ""))).resolve()
    != pathlib.Path(code_gate_path).resolve()
    or code_gate.get("sha256") != code_gate_sha
    or sha(code_gate_path) != code_gate_sha
    or short_gate.get("status") != "passed"
    or short_gate.get("git_commit") != commit
    or pathlib.Path(str(short_gate.get("path", ""))).resolve()
    != pathlib.Path(short_gate_path).resolve()
    or short_gate.get("sha256") != short_gate_sha
    or sha(short_gate_path) != short_gate_sha
):
    raise SystemExit("Stage-A manifest is not the frozen paper matrix")

expected = {
    (str(cell["arm"]), int(cell["seed"])) for cell in manifest.get("cells", [])
}
if len(expected) != 12:
    raise SystemExit("Stage-A manifest does not contain exactly twelve cells")
root = pathlib.Path(cells_root).resolve()
records = []
non_dense_budget_vectors = []
for arm, seed in sorted(expected):
    cell_root = root / arm / f"seed{seed}"
    receipt_path = cell_root / "cell.receipt.json"
    if not receipt_path.is_file():
        raise SystemExit(f"Stage-A cell receipt is missing: {arm}/seed{seed}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    evaluation_path = pathlib.Path(receipt.get("terminal_evaluation_path", "")).resolve()
    training_path = pathlib.Path(receipt.get("training_receipt_path", "")).resolve()
    if (
        receipt.get("schema_version") != "duca_paper_stage_a_cell_receipt_v2"
        or receipt.get("status") != "passed"
        or receipt.get("git_commit") != commit
        or receipt.get("arm") != arm
        or int(receipt.get("seed", -1)) != seed
        or receipt.get("paper_claim_ready") is not False
        or receipt.get("requires_complete_three_seed_matrix") is not True
        or not evaluation_path.is_file()
        or not training_path.is_file()
        or receipt.get("terminal_evaluation_sha256") != sha(evaluation_path)
        or receipt.get("training_receipt_sha256") != sha(training_path)
        or pathlib.Path(str(receipt.get("code_gate_path", ""))).resolve()
        != pathlib.Path(code_gate_path).resolve()
        or receipt.get("code_gate_sha256") != code_gate_sha
        or pathlib.Path(str(receipt.get("short_window_gate_path", ""))).resolve()
        != pathlib.Path(short_gate_path).resolve()
        or receipt.get("short_window_gate_sha256") != short_gate_sha
    ):
        raise SystemExit(f"Stage-A cell receipt drift: {arm}/seed{seed}")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if (
        evaluation.get("schema_version")
        != "duca_paper_full211_terminal_evaluation_v2"
        or evaluation.get("variant") != arm
        or int(evaluation.get("seed", -1)) != seed
        or int(evaluation.get("video_count", -1)) != 211
        or evaluation.get("training_consumed_validation") is not False
        or not isinstance(evaluation.get("metrics"), dict)
        or evaluation.get("exact211_execution", {}).get(
            "official_open_tad_pipeline_completed"
        )
        is not True
    ):
        raise SystemExit(f"Stage-A terminal evaluation drift: {arm}/seed{seed}")
    budget = evaluation.get("budget_execution", {})
    vector_sha = budget.get("window_budget_vector_sha256")
    if (
        budget.get("schema_version")
        != "duca_paper_exact211_budget_execution_v1"
        or budget.get("arm") != arm
        or budget.get("requested_budget_is_dynamic") is not False
        or receipt.get("evaluation_budget_execution_sha256")
        != budget.get("content_sha256")
        or receipt.get("window_budget_vector_sha256") != vector_sha
        or (arm == "dense" and vector_sha is not None)
        or (arm != "dense" and len(str(vector_sha or "")) != 64)
    ):
        raise SystemExit(f"Stage-A budget execution drift: {arm}/seed{seed}")
    if arm != "dense":
        non_dense_budget_vectors.append(str(vector_sha))
    records.append(
        {
            "arm": arm,
            "seed": seed,
            "cell_receipt_path": str(receipt_path.resolve()),
            "cell_receipt_sha256": sha(receipt_path),
            "terminal_evaluation_path": str(evaluation_path),
            "terminal_evaluation_sha256": sha(evaluation_path),
            "window_budget_vector_sha256": vector_sha,
        }
    )

if len(non_dense_budget_vectors) != 9 or len(set(non_dense_budget_vectors)) != 1:
    raise SystemExit(
        "Stage-A K384 evaluation arms do not share one realized per-window budget vector"
    )

payload = {
    "schema_version": "duca_paper_stage_a_matrix_receipt_v2",
    "status": "passed",
    "git_commit": commit,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "matrix_manifest_path": str(pathlib.Path(manifest_path).resolve()),
    "matrix_manifest_sha256": manifest_sha,
    "code_gate_path": str(pathlib.Path(code_gate_path).resolve()),
    "code_gate_sha256": code_gate_sha,
    "short_window_gate_path": str(pathlib.Path(short_gate_path).resolve()),
    "short_window_gate_sha256": short_gate_sha,
    "cell_count": 12,
    "complete_three_seed_matrix": True,
    "exact_full200_training_per_cell": True,
    "exact_full211_evaluation_per_cell": True,
    "paper_analysis_authorized": True,
    "paper_claim_ready": False,
    "metrics_withheld_from_engineering_receipt": True,
    "non_dense_k384_window_budget_vector_sha256": non_dense_budget_vectors[0],
    "non_dense_k384_budget_vector_identical_across_arms_and_seeds": True,
    "cells": records,
}
target = pathlib.Path(output)
with target.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
sha256sum "${DUCA_PAPER_MATRIX_ROOT}/matrix_receipt.json" \
  > "${DUCA_PAPER_MATRIX_ROOT}/matrix_receipt.sha256"
echo "[DUCA_PAPER_STAGE_A_SEAL] PASS ${DUCA_PAPER_MATRIX_ROOT}/matrix_receipt.json"
