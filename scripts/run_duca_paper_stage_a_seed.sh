#!/usr/bin/env bash
set -Eeuo pipefail

fail() {
  echo "[DUCA_PAPER_STAGE_A_SEED][FAIL] $*" >&2
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
  DUCA_PAPER_SEED \
  DUCA_PAPER_MATRIX_MANIFEST \
  DUCA_PAPER_MATRIX_MANIFEST_SHA256 \
  DUCA_PAPER_PRETRAIN_PATH \
  DUCA_PAPER_PRETRAIN_SHA256 \
  DUCA_PAPER_ANNOTATION_PATH \
  DUCA_PAPER_ANNOTATION_SHA256 \
  DUCA_PAPER_CLASS_MAP_PATH \
  DUCA_PAPER_CLASS_MAP_SHA256; do
  required "${name}"
done

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "seed groups must run inside Slurm"
case "${DUCA_PAPER_SEED}" in
  5801|8123|12011) ;;
  *) fail "unregistered Stage-A seed: ${DUCA_PAPER_SEED}" ;;
esac

cd "${DUCA_PAPER_REPO_ROOT}"
[[ "$(git rev-parse HEAD)" == "${DUCA_PAPER_EXPECTED_COMMIT}" ]] \
  || fail "Git commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail "Git tree is dirty"
[[ "$(sha256sum "${DUCA_PAPER_MATRIX_MANIFEST}" | cut -d ' ' -f 1)" == \
  "${DUCA_PAPER_MATRIX_MANIFEST_SHA256}" ]] || fail "matrix manifest SHA-256 drift"

config_for_arm() {
  case "$1" in
    dense)
      echo "${DUCA_PAPER_REPO_ROOT}/configs/adatad/thumos/duca_paper_dense_actionformer_full200.py"
      ;;
    uniform_fixed_k384)
      echo "${DUCA_PAPER_REPO_ROOT}/configs/adatad/thumos/duca_paper_uniform_fixed_k384_full200.py"
      ;;
    uniform_mixed_train_k384_eval)
      echo "${DUCA_PAPER_REPO_ROOT}/configs/adatad/thumos/duca_paper_uniform_mixed_train_k384_eval_full200.py"
      ;;
    duca_fixed_k384)
      echo "${DUCA_PAPER_REPO_ROOT}/configs/adatad/thumos/duca_paper_duca_fixed_k384_full200.py"
      ;;
    *) fail "unregistered Stage-A arm: $1" ;;
  esac
}

for arm in \
  dense \
  uniform_fixed_k384 \
  uniform_mixed_train_k384_eval \
  duca_fixed_k384; do
  export DUCA_PAPER_ARM="${arm}"
  export DUCA_PAPER_CONFIG="$(config_for_arm "${arm}")"
  export DUCA_PAPER_CELL_ROOT="${DUCA_PAPER_CELLS_ROOT}/${arm}/seed${DUCA_PAPER_SEED}"
  if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
    PRECHECK_ONLY=1 bash scripts/run_duca_paper_stage_a_cell.sh
  else
    bash scripts/run_duca_paper_stage_a_cell.sh
  fi
done

if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  echo "[DUCA_PAPER_STAGE_A_SEED] PRECHECK PASS seed${DUCA_PAPER_SEED}"
  exit 0
fi

receipt_root="${DUCA_PAPER_CELLS_ROOT}/_seed_receipts"
mkdir -p "${receipt_root}"
python - \
  "${DUCA_PAPER_CELLS_ROOT}" \
  "${DUCA_PAPER_SEED}" \
  "${DUCA_PAPER_EXPECTED_COMMIT}" \
  "${receipt_root}/seed${DUCA_PAPER_SEED}.json" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

cells_root, seed, commit, output = sys.argv[1:]
arms = (
    "dense",
    "uniform_fixed_k384",
    "uniform_mixed_train_k384_eval",
    "duca_fixed_k384",
)
sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
records = []
for arm in arms:
    path = pathlib.Path(cells_root) / arm / f"seed{seed}" / "cell.receipt.json"
    if not path.is_file():
        raise SystemExit(f"grouped Stage-A cell receipt is missing: {arm}/seed{seed}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != "duca_paper_stage_a_cell_receipt_v1"
        or payload.get("status") != "passed"
        or payload.get("git_commit") != commit
        or payload.get("arm") != arm
        or int(payload.get("seed", -1)) != int(seed)
    ):
        raise SystemExit(f"grouped Stage-A cell receipt drift: {arm}/seed{seed}")
    records.append(
        {
            "arm": arm,
            "cell_receipt_path": str(path.resolve()),
            "cell_receipt_sha256": sha(path),
        }
    )
payload = {
    "schema_version": "duca_paper_stage_a_seed_receipt_v1",
    "status": "passed",
    "git_commit": commit,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "seed": int(seed),
    "logical_cell_count": 4,
    "sequential_scheduler_grouping_only": True,
    "cells": records,
}
target = pathlib.Path(output)
with target.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

echo "[DUCA_PAPER_STAGE_A_SEED] PASS seed${DUCA_PAPER_SEED}"
