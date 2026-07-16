#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_CELLCF_COST][FAIL] $*" >&2
  exit 1
}

SUITE_MANIFEST="${DUCA_CELLCF_SUITE_MANIFEST:-}"
SUITE_MANIFEST_SHA256="${DUCA_CELLCF_SUITE_MANIFEST_SHA256:-}"
AGGREGATE_EVIDENCE="${DUCA_CELLCF_AGGREGATE_EVIDENCE:-}"
REQUESTED_OUTPUT_JSON="${DUCA_CELLCF_COST_EVIDENCE:-}"
while (($#)); do
  case "$1" in
    --suite-manifest) SUITE_MANIFEST="${2:-}"; shift 2 ;;
    --suite-manifest-sha256) SUITE_MANIFEST_SHA256="${2:-}"; shift 2 ;;
    --aggregate-evidence) AGGREGATE_EVIDENCE="${2:-}"; shift 2 ;;
    --output-json) REQUESTED_OUTPUT_JSON="${2:-}"; shift 2 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
BASE="${BASE:-/data/run01/sczc063/yuzibo}"
source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"
EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:-}"
CHECKPOINT="${DUCA_CELLCF_CHECKPOINT:-}"
POST_RUN_EVIDENCE="${DUCA_CELLCF_POST_RUN_EVIDENCE_JSON:-}"
POST_RUN_EVIDENCE_SHA256="${DUCA_CELLCF_POST_RUN_EVIDENCE_SHA256:-}"
OUTPUT_ROOT="${DUCA_CELLCF_COST_ROOT:-${BASE}/projects/c3_lowres_action_probe/duca_cellcf_cost}"
OUTPUT_JSON="${REQUESTED_OUTPUT_JSON:-${OUTPUT_ROOT}/cellcf_vs_bare_uniform.json}"
SAMPLES="${DUCA_CELLCF_COST_SAMPLES:-500}"
WARMUP="${DUCA_CELLCF_COST_WARMUP:-20}"
REPEATS="${DUCA_CELLCF_COST_REPEATS:-3}"
CELL_CONFIG="configs/adatad/thumos/duca_cellcf_fixed384_official_adatad_backend_full_train.py"
BARE_CONFIG="configs/adatad/thumos/duca_cellcf_bare_exact_uniform_fixed384_cost.py"
PRETRAIN="${ADATAD_PRETRAIN_PATH}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "formal cost profiling must run inside Slurm"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || fail "Slurm did not expose a logical GPU"
[[ -x "${PYTHON}" ]] || fail "Python is missing: ${PYTHON}"
if [[ -n "${SUITE_MANIFEST}" || -n "${SUITE_MANIFEST_SHA256}" ]]; then
  [[ -f "${SUITE_MANIFEST}" ]] || fail "CellCF suite manifest is missing"
  [[ "${SUITE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "CellCF suite manifest SHA256 is missing or invalid"
  [[ "$(sha256sum "${SUITE_MANIFEST}" | awk '{print $1}')" == "${SUITE_MANIFEST_SHA256}" ]] \
    || fail "CellCF suite manifest hash drift"
fi
if [[ -n "${AGGREGATE_EVIDENCE}" ]]; then
  [[ -f "${AGGREGATE_EVIDENCE}" ]] || fail "CellCF aggregate evidence is missing"
  readarray -t aggregate_binding < <("${PYTHON}" - "${AGGREGATE_EVIDENCE}" "${EXPECTED_COMMIT}" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
expected_commit = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
if (
    not isinstance(payload, dict)
    or payload.get("ok") is not True
    or payload.get("status") != "runs_complete_cost_pending"
):
    raise SystemExit("aggregate evidence is not the expected cost-pending suite artifact")
if payload.get("git_commit") != expected_commit:
    raise SystemExit("aggregate evidence commit mismatch")
completed = payload.get("completed_runs")
cellcf = completed.get("cellcf") if isinstance(completed, dict) else None
if not isinstance(cellcf, dict):
    raise SystemExit("aggregate evidence has no validated CellCF run")
values = (
    cellcf.get("path"),
    cellcf.get("sha256"),
    cellcf.get("checkpoint_path"),
    cellcf.get("checkpoint_sha256"),
)
if not all(isinstance(value, str) and value for value in values):
    raise SystemExit("aggregate CellCF binding is incomplete")
if re.fullmatch(r"[0-9a-f]{64}", values[1]) is None or re.fullmatch(r"[0-9a-f]{64}", values[3]) is None:
    raise SystemExit("aggregate CellCF hashes are invalid")
print(*values, sep="\n")
PY
  )
  [[ "${#aggregate_binding[@]}" == "4" ]] || fail "failed to reopen aggregate CellCF binding"
  [[ -z "${POST_RUN_EVIDENCE}" || "${POST_RUN_EVIDENCE}" == "${aggregate_binding[0]}" ]] \
    || fail "explicit post-run evidence path differs from aggregate evidence"
  [[ -z "${POST_RUN_EVIDENCE_SHA256}" || "${POST_RUN_EVIDENCE_SHA256}" == "${aggregate_binding[1]}" ]] \
    || fail "explicit post-run evidence SHA256 differs from aggregate evidence"
  POST_RUN_EVIDENCE="${aggregate_binding[0]}"
  POST_RUN_EVIDENCE_SHA256="${aggregate_binding[1]}"
  [[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${aggregate_binding[3]}" ]] \
    || fail "CellCF checkpoint differs from aggregate evidence"
fi
[[ -f "${CHECKPOINT}" ]] || fail "terminal CellCF checkpoint is missing"
[[ -f "${POST_RUN_EVIDENCE}" ]] || fail "CellCF post-run evidence is missing"
[[ "${POST_RUN_EVIDENCE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || fail "CellCF post-run evidence SHA256 is missing or invalid"
[[ -f "${PRETRAIN}" ]] || fail "AdaTAD pretrain is missing"
[[ "${SAMPLES}" =~ ^[0-9]+$ && "${SAMPLES}" -ge 500 ]] || fail "at least 500 measured windows are required"
[[ "${REPEATS}" =~ ^[0-9]+$ && "${REPEATS}" -ge 3 ]] || fail "at least three fresh-process repeats are required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || fail "HEAD differs from DUCA_EXPECTED_COMMIT"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "cost profiling requires a clean tree"
[[ "$(sha256sum "${POST_RUN_EVIDENCE}" | awk '{print $1}')" == "${POST_RUN_EVIDENCE_SHA256}" ]] \
  || fail "CellCF post-run evidence hash drift"
[[ "$(${PYTHON} -c 'import torch; print(torch.cuda.device_count())')" == "1" ]] || fail "exactly one Slurm-visible GPU is required"

mkdir -p "${OUTPUT_ROOT}" "$(dirname "${OUTPUT_JSON}")"
cellcf_args=()
bare_args=()
for repeat in $(seq 1 "${REPEATS}"); do
  cell_prefix="${OUTPUT_ROOT}/cellcf_repeat${repeat}"
  bare_prefix="${OUTPUT_ROOT}/bare_uniform_repeat${repeat}"
  "${PYTHON}" tools/bata/profile_duca_full_stack_cost.py "${CELL_CONFIG}" \
    --checkpoint "${CHECKPOINT}" --use-ema --backbone-pretrain "${PRETRAIN}" \
    --output-prefix "${cell_prefix}" --method-name cellcf-fixed384 \
    --config-commit "${EXPECTED_COMMIT}" --device cuda:0 --samples "${SAMPLES}" \
    --warmup-samples "${WARMUP}" --batch-size 1 --loader-workers 0 --amp \
    --post-run-evidence "${POST_RUN_EVIDENCE}" \
    --post-run-evidence-sha256 "${POST_RUN_EVIDENCE_SHA256}"
  "${PYTHON}" tools/bata/profile_duca_full_stack_cost.py "${BARE_CONFIG}" \
    --checkpoint "${CHECKPOINT}" --use-ema --backbone-pretrain "${PRETRAIN}" \
    --output-prefix "${bare_prefix}" --method-name bare-uniform384 \
    --config-commit "${EXPECTED_COMMIT}" --device cuda:0 --samples "${SAMPLES}" \
    --warmup-samples "${WARMUP}" --batch-size 1 --loader-workers 0 --amp \
    --post-run-evidence "${POST_RUN_EVIDENCE}" \
    --post-run-evidence-sha256 "${POST_RUN_EVIDENCE_SHA256}"
  cellcf_args+=(--cellcf "${cell_prefix}.json")
  bare_args+=(--bare-uniform "${bare_prefix}.json")
done

"${PYTHON}" tools/bata/summarize_duca_cellcf_cost.py \
  "${cellcf_args[@]}" "${bare_args[@]}" \
  --post-run-evidence "${POST_RUN_EVIDENCE}" \
  --post-run-evidence-sha256 "${POST_RUN_EVIDENCE_SHA256}" \
  --output-json "${OUTPUT_JSON}"
