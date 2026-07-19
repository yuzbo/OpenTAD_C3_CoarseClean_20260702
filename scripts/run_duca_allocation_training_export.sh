#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_ALLOCATION_EXPORT][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
PYTHON="${BASE}/conda_envs/opentad/bin/python"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
CHECKPOINT="${DUCA_ALLOCATION_CHECKPOINT:-}"
RUN_ROOT="${DUCA_ALLOCATION_RUN_ROOT:-}"
GATE_JSON="${DUCA_ALLOCATION_GATE_JSON:-}"
CONFIG="configs/adatad/thumos/duca_allocation_ceiling_training_windows.py"

[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "export requires a Slurm GPU"
[[ "${SLURM_CLUSTER_NAME:-}" == "n16r4" ]] || fail "export requires cluster n16r4"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "commit drift"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "export requires a clean tree"
[[ -f "${CHECKPOINT}" && -f "${GATE_JSON}" ]] || fail "checkpoint or gate is missing"
[[ "${RUN_ROOT}" == "${BASE}/"* && -d "${RUN_ROOT}" ]] || fail "invalid run root"
[[ ! -e "${RUN_ROOT}/training_inputs.jsonl" ]] || fail "refusing to overwrite export"

"${PYTHON}" - "${GATE_JSON}" "${EXPECTED_COMMIT}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("gate_passed") is not True or payload.get("git_commit") != sys.argv[2]:
    raise SystemExit("allocation gate binding is invalid")
PY

"${PYTHON}" -m tools.bata.export_duca_allocation_ceiling_inputs \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output-jsonl "${RUN_ROOT}/training_inputs.jsonl" \
  --summary-json "${RUN_ROOT}/training_inputs.summary.json" \
  --split train \
  --requested-budget 384 \
  --device cuda:0 \
  --use-ema true \
  --batch-size 1 \
  --num-workers 2 \
  --coordinate-tolerance-frames 0

echo "[DUCA_ALLOCATION_EXPORT] PASS"
