#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PhysTime decode cross suite] ERROR: $*" >&2
  exit 1
}

BASE="${PHYSTIME_BASE:-/data/run01/sczc063/yuzibo}"
WORK_DIR="${PHYSTIME_WORK_DIR:?PHYSTIME_WORK_DIR is required}"
PYTHON="${PHYSTIME_PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
EXPECTED_COMMIT="${PHYSTIME_EXPECTED_COMMIT:?runtime commit is required}"
EXPECTED_TREE="${PHYSTIME_EXPECTED_TREE:?runtime tree is required}"
PREFLIGHT="${PHYSTIME_DECODE_PREFLIGHT:?preflight manifest is required}"
GATE="${PHYSTIME_DECODE_GATE_OUTPUT:?decode-cross gate is required}"
P0_SUITE="${PHYSTIME_P0_SUITE_COMPLETION:?P0 suite completion is required}"
P0_GATE="${PHYSTIME_P0_GATE_OUTPUT:?P0 gate is required}"
SELECTED_ONLINE="${PHYSTIME_DECODE_SELECTED_ONLINE_COMPLETION:?selected-online completion is required}"
SELECTED_EMA="${PHYSTIME_DECODE_SELECTED_EMA_COMPLETION:?selected-EMA completion is required}"
PHYSICAL_ONLINE="${PHYSTIME_DECODE_PHYSICAL_ONLINE_COMPLETION:?physical-online completion is required}"
PHYSICAL_EMA="${PHYSTIME_DECODE_PHYSICAL_EMA_COMPLETION:?physical-EMA completion is required}"
LOG_PATHS_RAW="${PHYSTIME_DECODE_LOG_PATHS:?space-separated explicit log paths are required}"
OUTPUT="${PHYSTIME_DECODE_SUITE_OUTPUT:?suite output path is required}"
VALIDATOR_LOG="${PHYSTIME_DECODE_SUITE_VALIDATOR_LOG:-${OUTPUT%.json}.validator.log}"
MARKER="${PHYSTIME_DECODE_SUITE_MARKER:-${OUTPUT%.json}.validated.json}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "suite must run inside Slurm"
[[ -n "${SLURM_JOB_NAME:-}" ]] || fail "Slurm job name is missing"
[[ ! -e "${OUTPUT}" ]] || fail "suite output already exists: ${OUTPUT}"
[[ ! -e "${MARKER}" ]] || fail "suite marker already exists: ${MARKER}"

for path in \
  "${PREFLIGHT}" \
  "${GATE}" \
  "${P0_SUITE}" \
  "${P0_GATE}" \
  "${SELECTED_ONLINE}" \
  "${SELECTED_EMA}" \
  "${PHYSICAL_ONLINE}" \
  "${PHYSICAL_EMA}"; do
  [[ -f "${path}" ]] || fail "required evidence artifact is missing: ${path}"
done

read -r -a LOG_PATHS <<<"${LOG_PATHS_RAW}"
[[ "${#LOG_PATHS[@]}" -gt 0 ]] || fail "no explicit log paths were supplied"
LOG_ARGS=()
for path in "${LOG_PATHS[@]}"; do
  [[ -f "${path}" ]] || fail "required log artifact is missing: ${path}"
  LOG_ARGS+=(--log "${path}")
done

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi
source "${BASE}/conda_envs/opentad/bin/activate"
cd "${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] \
  || fail "runtime commit mismatch"
[[ "$(git rev-parse 'HEAD^{tree}')" == "${EXPECTED_TREE}" ]] \
  || fail "runtime tree mismatch"
[[ -z "$(git status --porcelain)" ]] || fail "runtime snapshot is dirty"

mkdir -p "$(dirname "${OUTPUT}")" "$(dirname "${VALIDATOR_LOG}")" "$(dirname "${MARKER}")"
"${PYTHON}" tools/bata/validate_phystime_decode_cross_suite.py \
  --preflight "${PREFLIGHT}" \
  --gate "${GATE}" \
  --p0-suite "${P0_SUITE}" \
  --p0-gate "${P0_GATE}" \
  --completion "selected_online=${SELECTED_ONLINE}" \
  --completion "selected_ema=${SELECTED_EMA}" \
  --completion "physical_online=${PHYSICAL_ONLINE}" \
  --completion "physical_ema=${PHYSICAL_EMA}" \
  "${LOG_ARGS[@]}" \
  --output "${OUTPUT}" \
  2>&1 | tee "${VALIDATOR_LOG}"

"${PYTHON}" - "${OUTPUT}" "${MARKER}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

completion_path = Path(sys.argv[1]).resolve()
marker_path = Path(sys.argv[2]).resolve()
completion = json.loads(completion_path.read_text(encoding="utf-8"))
if (
    completion.get("schema_version")
    != "phystime_decode_cross_evidence_suite_completion_v1"
    or completion.get("validation_pass") is not True
    or completion.get("status") != "tested"
    or completion.get("evidence_mode") != "explicit_artifact_paths_v1"
):
    raise SystemExit("decode-cross evidence suite did not pass")

payload = {
    "schema_version": "phystime_decode_cross_evidence_suite_marker_v1",
    "validation_pass": True,
    "completion": {
        "path": str(completion_path),
        "sha256": hashlib.sha256(completion_path.read_bytes()).hexdigest(),
        "size_bytes": completion_path.stat().st_size,
    },
}
temporary = marker_path.with_name(f"{marker_path.name}.tmp.{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, marker_path)
PY

echo "[PhysTime decode cross suite] COMPLETE"
