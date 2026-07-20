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
RUN_ROOT="${PHYSTIME_DECODE_RUN_ROOT:?run root is required}"
EXPECTED_DEPENDENCY="${PHYSTIME_EXPECTED_DEPENDENCY:?dependency is required}"
SBATCH_PATH="${PHYSTIME_SBATCH_PATH:?sbatch path is required}"
JOBS_TSV="${PHYSTIME_DECODE_JOBS_TSV:?jobs TSV is required}"
PREFLIGHT="${PHYSTIME_DECODE_PREFLIGHT:?preflight manifest is required}"
PREFLIGHT_SHA256="${PHYSTIME_DECODE_PREFLIGHT_SHA256:?preflight SHA256 is required}"
DAG_TOKEN="${PHYSTIME_DAG_TOKEN:?DAG token is required}"
EXPECTED_COMMENT="${PHYSTIME_EXPECTED_JOB_COMMENT:?job comment is required}"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "suite must run inside Slurm"
[[ -n "${SLURM_JOB_NAME:-}" ]] || fail "suite Slurm job name is missing"
[[ "${SLURM_JOB_NAME}" == "pt_dc_suite" ]] \
  || fail "suite Slurm job name mismatch"
[[ "$(scontrol show job -o "${SLURM_JOB_ID}")" == *"Comment=${EXPECTED_COMMENT}"* ]] \
  || fail "suite Slurm comment mismatch"
[[ -f "${SBATCH_PATH}" ]] || fail "suite sbatch file is missing"
[[ -f "${PREFLIGHT}" ]] || fail "preflight manifest is missing"
[[ "${EXPECTED_DEPENDENCY}" == afterok:* ]] \
  || fail "suite dependency contract is invalid"
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

for variant in selected_online selected_ema physical_online physical_ema; do
  [[ -f "${RUN_ROOT}/${variant}/DECODE_CROSS_COMPLETE.json" ]] \
    || fail "condition completion is missing: ${variant}"
done

"${PYTHON}" - "${PREFLIGHT}" "${PREFLIGHT_SHA256}" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if digest != sys.argv[2]:
    raise SystemExit("decode-cross preflight hash changed before suite")
PY

SCHEDULER_TERMINAL="${RUN_ROOT}/scheduler_terminal.json"
"${PYTHON}" tools/bata/capture_phystime_decode_cross_scheduler.py \
  --jobs-tsv "${JOBS_TSV}" \
  --dag-token "${DAG_TOKEN}" \
  --mode terminal \
  --output "${SCHEDULER_TERMINAL}"

"${PYTHON}" tools/bata/validate_phystime_decode_cross_suite.py \
  --run-root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/DECODE_CROSS_SUITE_COMPLETE.json" \
  2>&1 | tee "${RUN_ROOT}/suite_validator.out"

"${PYTHON}" - "${RUN_ROOT}/DECODE_CROSS_SUITE_COMPLETE.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("validation_pass") is not True:
    raise SystemExit("decode cross suite did not pass")
if payload.get("status") != "tested":
    raise SystemExit("decode cross suite status is not tested")
PY

"${PYTHON}" - \
  "${RUN_ROOT}/DECODE_CROSS_SUITE_VALIDATED" \
  "${RUN_ROOT}/DECODE_CROSS_SUITE_COMPLETE.json" \
  "${SCHEDULER_TERMINAL}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

output, completion, scheduler = map(Path, sys.argv[1:])
payload = {
    "schema_version": "phystime_decode_cross_suite_marker_v1",
    "validation_pass": True,
    "completion_sha256": hashlib.sha256(
        completion.read_bytes()
    ).hexdigest(),
    "scheduler_terminal_sha256": hashlib.sha256(
        scheduler.read_bytes()
    ).hexdigest(),
}
temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
with temporary.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, output)
PY

echo "[PhysTime decode cross suite] COMPLETE"
