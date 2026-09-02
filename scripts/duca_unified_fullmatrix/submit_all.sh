#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
RUN_ROOT="${RUN_ROOT:-/data/run01/sczc063/yuzibo/runs/DUCA-UNIFIED-FULLMATRIX-v001-20260902}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
mkdir -p "$RUN_ROOT"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is not available; run this script on the N16R4 Slurm login node." >&2
  exit 2
fi

export PROJECT_DIR RUN_ROOT

preflight=$(sbatch --parsable scripts/duca_unified_fullmatrix/preflight.sbatch)
train=$(sbatch --parsable --dependency=afterok:$preflight --array=0-40%$MAX_CONCURRENT scripts/duca_unified_fullmatrix/train_eval_array.sbatch)
cost=$(sbatch --parsable --dependency=afterok:$train scripts/duca_unified_fullmatrix/cost_array.sbatch)
boot=$(sbatch --parsable --dependency=afterok:$train scripts/duca_unified_fullmatrix/bootstrap_array.sbatch)
finalize=$(sbatch --parsable --dependency=afterok:$train:$cost:$boot scripts/duca_unified_fullmatrix/finalize.sbatch)
audit=$(sbatch --parsable --dependency=afterany:$train:$cost:$boot:$finalize scripts/duca_unified_fullmatrix/audit_afterany.sbatch)

python - "$RUN_ROOT" <<PY
import json, pathlib
run_root = pathlib.Path("$RUN_ROOT")
payload = {
    "schema_version": "duca_unified_slurm_submission_v1",
    "matrix_id": "DUCA-UNIFIED-FULLMATRIX-v001-20260902",
    "project_dir": "$PROJECT_DIR",
    "run_root": str(run_root),
    "jobs": {
        "preflight": "$preflight",
        "train_eval_array": "$train",
        "cost_array": "$cost",
        "bootstrap_array": "$boot",
        "finalize": "$finalize",
        "audit_afterany": "$audit",
    },
}
(run_root / "slurm_submission_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
