#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[PhysTime P0 full-precision suite] ERROR: $*" >&2
  exit 1
}

WORK_DIR="${PHYSTIME_WORK_DIR:?PHYSTIME_WORK_DIR is required}"
BASE="${PHYSTIME_BASE:-/data/run01/sczc063/yuzibo}"
PYTHON="${PHYSTIME_PYTHON:-${BASE}/conda_envs/opentad/bin/python}"
EXPECTED_COMMIT="${PHYSTIME_EXPECTED_COMMIT:?PHYSTIME_EXPECTED_COMMIT is required}"
EXPECTED_TREE="${PHYSTIME_EXPECTED_TREE:?PHYSTIME_EXPECTED_TREE is required}"
RUN_ROOT="${PHYSTIME_P0_RUN_ROOT:?PHYSTIME_P0_RUN_ROOT is required}"
OUTPUT="${RUN_ROOT}/P0_SUITE_COMPLETE.json"

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "suite validator must run inside Slurm"
[[ -d "${WORK_DIR}" && -d "${RUN_ROOT}" ]] \
  || fail "runtime snapshot or run root is missing"

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
  PHYSTIME_ENV_INIT_MODE="module_cuda11.8_miniforge3_24.11"
else
  PHYSTIME_ENV_INIT_MODE="fixed_conda_path_no_module_command"
fi
export PHYSTIME_ENV_INIT_MODE
source "${BASE}/conda_envs/opentad/bin/activate"
cd "${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"

COMMIT="$(git rev-parse HEAD)"
TREE="$(git rev-parse 'HEAD^{tree}')"
[[ "${COMMIT}" == "${EXPECTED_COMMIT}" ]] || fail "runtime commit mismatch"
[[ "${TREE}" == "${EXPECTED_TREE}" ]] || fail "runtime tree mismatch"
[[ -z "$(git status --porcelain)" ]] || fail "runtime snapshot is dirty"

for variant in selected_online selected_ema physical_online physical_ema; do
  [[ -f "${RUN_ROOT}/${variant}/P0_COMPLETE.json" ]] \
    || fail "missing independent completion for ${variant}"
done

"${PYTHON}" tools/bata/validate_phystime_p0_fullprecision_suite.py \
  --run-root "${RUN_ROOT}" \
  --output "${OUTPUT}" \
  2>&1 | tee "${RUN_ROOT}/suite_validator.out"

"${PYTHON}" - "${OUTPUT}" "${RUN_ROOT}/runtime_summary.json" <<'PY'
import json
import sys
from pathlib import Path

completion_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
completion = json.loads(completion_path.read_text(encoding="utf-8"))
if completion.get("validation_pass") is not True:
    raise SystemExit("P0 suite completion did not pass")
summary = {
    "validation_pass": True,
    "status": completion["status"],
    "new_training": False,
    "frozen_epoch": completion["frozen_epoch"],
    "runtime_commit": completion["runtime_commit"],
    "runtime_tree": completion["runtime_tree"],
    "source_commit": completion["source_commit"],
    "source_tree": completion["source_tree"],
    "mode_metrics": completion["mode_metrics"],
    "cross_arm_physical_minus_selected": completion[
        "cross_arm_physical_minus_selected"
    ],
    "weight_source_ema_minus_online": completion[
        "weight_source_ema_minus_online"
    ],
    "claim_boundary": completion["claim_boundary"],
}
summary_path.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

echo "[PhysTime P0 full-precision suite] COMPLETE output=${OUTPUT}"
