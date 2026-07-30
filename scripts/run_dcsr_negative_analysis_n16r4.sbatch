#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH -J af-dcsr-neg

set -eo pipefail
source /etc/profile >/dev/null 2>&1
set -u
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

: "${DIAGNOSTIC_RUNTIME:?missing DIAGNOSTIC_RUNTIME}"
: "${SOURCE_RUN_ROOT:?missing SOURCE_RUN_ROOT}"
: "${DIAGNOSTIC_RUN_ROOT:?missing DIAGNOSTIC_RUN_ROOT}"
: "${OFFICIAL_DATA_ROOT:?missing OFFICIAL_DATA_ROOT}"
: "${EXPECTED_DIAGNOSTIC_COMMIT:?missing EXPECTED_DIAGNOSTIC_COMMIT}"
: "${EXPECTED_DIAGNOSTIC_TREE:?missing EXPECTED_DIAGNOSTIC_TREE}"
: "${EXPECTED_ANNOTATION_SHA256:?missing EXPECTED_ANNOTATION_SHA256}"
: "${DCSR_INTERNAL_HOLDOUT_MANIFEST:?missing DCSR_INTERNAL_HOLDOUT_MANIFEST}"
: "${DCSR_INTERNAL_HOLDOUT_MANIFEST_SHA256:?missing DCSR_INTERNAL_HOLDOUT_MANIFEST_SHA256}"
: "${DCSR_G1_AGGREGATE:?missing DCSR_G1_AGGREGATE}"
: "${EXPECTED_DCSR_G1_AGGREGATE_SHA256:?missing EXPECTED_DCSR_G1_AGGREGATE_SHA256}"
: "${ACTIONFORMER_PYTHON_ENV:?missing ACTIONFORMER_PYTHON_ENV}"
: "${ACTIONFORMER_ENVIRONMENT_RECEIPT:?missing ACTIONFORMER_ENVIRONMENT_RECEIPT}"
: "${EXPECTED_ENVIRONMENT_RECEIPT_SHA256:?missing EXPECTED_ENVIRONMENT_RECEIPT_SHA256}"
: "${ACTIONFORMER_NMS_EXTENSION:?missing ACTIONFORMER_NMS_EXTENSION}"
: "${EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256:?missing EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256}"

case "$DIAGNOSTIC_RUNTIME" in
  /data/run01/sczc063/yuzibo/projects/*) ;;
  *) echo "diagnostic runtime outside write boundary" >&2; exit 2 ;;
esac
case "$SOURCE_RUN_ROOT" in
  /data/run01/sczc063/yuzibo/projects/*/runs/*) ;;
  *) echo "source run root outside read boundary" >&2; exit 2 ;;
esac
case "$DIAGNOSTIC_RUN_ROOT" in
  /data/run01/sczc063/yuzibo/projects/*/runs/*) ;;
  *) echo "diagnostic run root outside write boundary" >&2; exit 2 ;;
esac
test -n "${SLURM_JOB_ID:-}"
test -n "${CUDA_VISIBLE_DEVICES:-}"
test -d "$SOURCE_RUN_ROOT"
test -d "$DIAGNOSTIC_RUN_ROOT"
test -f "$DCSR_INTERNAL_HOLDOUT_MANIFEST"
test "$(sha256sum "$DCSR_INTERNAL_HOLDOUT_MANIFEST" | awk '{print $1}')" = \
  "$DCSR_INTERNAL_HOLDOUT_MANIFEST_SHA256"
test -f "$DCSR_G1_AGGREGATE"
test "$(sha256sum "$DCSR_G1_AGGREGATE" | awk '{print $1}')" = \
  "$EXPECTED_DCSR_G1_AGGREGATE_SHA256"
for seed in 2026073001 2026073002 2026073003; do
  test -f "$DIAGNOSTIC_RUN_ROOT/seed_${seed}/COUNTERFACTUAL_COMPLETE.json"
  test ! -f "$DIAGNOSTIC_RUN_ROOT/seed_${seed}/ENGINEERING_FAILURE.json"
done

ANALYSIS_ROOT="$DIAGNOSTIC_RUN_ROOT/analysis"
test ! -e "$ANALYSIS_ROOT"
mkdir "$ANALYSIS_ROOT"
RUN_LOG="$ANALYSIS_ROOT/runtime.log"
STAGE=bootstrap
exec > >(tee -a "$RUN_LOG") 2>&1

failure_receipt() {
  status=$?
  if [ "$status" -ne 0 ]; then
    STATUS="$status" STAGE="$STAGE" ANALYSIS_ROOT="$ANALYSIS_ROOT" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ANALYSIS_ROOT"])
payload = {
    "schema_version": "actionformer_dcsr_analysis_engineering_failure_v1",
    "failure_class": "engineering_or_protocol_failure",
    "exit_code": int(os.environ["STATUS"]),
    "stage": os.environ["STAGE"],
    "model_result_claim_allowed": False,
}
temporary = root / "ENGINEERING_FAILURE.json.tmp"
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, root / "ENGINEERING_FAILURE.json")
PY
  fi
}
trap failure_receipt EXIT

STAGE=environment_identity
test -x "$ACTIONFORMER_PYTHON_ENV/bin/python"
test -f "$ACTIONFORMER_ENVIRONMENT_RECEIPT"
test "$(sha256sum "$ACTIONFORMER_ENVIRONMENT_RECEIPT" | awk '{print $1}')" = \
  "$EXPECTED_ENVIRONMENT_RECEIPT_SHA256"
test -f "$ACTIONFORMER_NMS_EXTENSION"
test "$(sha256sum "$ACTIONFORMER_NMS_EXTENSION" | awk '{print $1}')" = \
  "$EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256"
export PATH="$ACTIONFORMER_PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$DIAGNOSTIC_RUNTIME:$(dirname "$ACTIONFORMER_NMS_EXTENSION")${PYTHONPATH:+:$PYTHONPATH}"
hash -r
test "$(readlink -f "$(command -v python)")" = \
  "$(readlink -f "$ACTIONFORMER_PYTHON_ENV/bin/python")"

STAGE=source_identity
test "$(git -C "$DIAGNOSTIC_RUNTIME" rev-parse HEAD)" = \
  "$EXPECTED_DIAGNOSTIC_COMMIT"
test "$(git -C "$DIAGNOSTIC_RUNTIME" rev-parse 'HEAD^{tree}')" = \
  "$EXPECTED_DIAGNOSTIC_TREE"
test -z "$(git -C "$DIAGNOSTIC_RUNTIME" status --porcelain=v1 --untracked-files=all)"
annotation="$OFFICIAL_DATA_ROOT/annotations/thumos14.json"
test -f "$annotation"
test "$(sha256sum "$annotation" | awk '{print $1}')" = \
  "$EXPECTED_ANNOTATION_SHA256"

STAGE=focused_tests
(
  cd "$DIAGNOSTIC_RUNTIME"
  python -m pytest \
    tests/test_dcsr_negative_diagnostics.py \
    tests/test_dcsr_heads.py \
    tests/test_dcsr_internal_holdout.py \
    -q
) | tee "$ANALYSIS_ROOT/focused_tests.log"

STAGE=prediction_diagnostics
(
  cd "$DIAGNOSTIC_RUNTIME"
  python -m tools.analyze_dcsr_internal_predictions \
    --annotation "$annotation" \
    --expected-annotation-sha256 "$EXPECTED_ANNOTATION_SHA256" \
    --manifest "$DCSR_INTERNAL_HOLDOUT_MANIFEST" \
    --expected-manifest-sha256 "$DCSR_INTERNAL_HOLDOUT_MANIFEST_SHA256" \
    --source-run-root "$SOURCE_RUN_ROOT" \
    --counterfactual-run-root "$DIAGNOSTIC_RUN_ROOT" \
    --aggregate "$DCSR_G1_AGGREGATE" \
    --expected-aggregate-sha256 "$EXPECTED_DCSR_G1_AGGREGATE_SHA256" \
    --seeds 2026073001 2026073002 2026073003 \
    --output "$ANALYSIS_ROOT/DCSR_NEGATIVE_DIAGNOSTICS.json"
) | tee "$ANALYSIS_ROOT/prediction_diagnostics.log"

STAGE=checkpoint_dynamics
(
  cd "$DIAGNOSTIC_RUNTIME"
  python -m tools.analyze_dcsr_checkpoint_dynamics \
    --source-run-root "$SOURCE_RUN_ROOT" \
    --aggregate "$DCSR_G1_AGGREGATE" \
    --expected-aggregate-sha256 "$EXPECTED_DCSR_G1_AGGREGATE_SHA256" \
    --seeds 2026073001 2026073002 2026073003 \
    --output "$ANALYSIS_ROOT/DCSR_CHECKPOINT_DYNAMICS.json"
) | tee "$ANALYSIS_ROOT/checkpoint_dynamics.log"

STAGE=analysis_completion
ANALYSIS_ROOT="$ANALYSIS_ROOT" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["ANALYSIS_ROOT"])

def record(path):
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }

prediction_path = root / "DCSR_NEGATIVE_DIAGNOSTICS.json"
checkpoint_path = root / "DCSR_CHECKPOINT_DYNAMICS.json"
prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
if not (
    prediction["validation_pass"] is True
    and prediction["diagnostic_only"] is True
    and prediction["paper_performance_row_allowed"] is False
    and prediction["test_gt_used"] is False
    and prediction["test_predictions_used"] is False
    and checkpoint["validation_pass"] is True
    and checkpoint["training_performed"] is False
    and checkpoint["test_data_read"] is False
):
    raise ValueError("negative diagnostic completion contract failed")
payload = {
    "schema_version": "actionformer_dcsr_negative_diagnostics_complete_v1",
    "validation_pass": True,
    "diagnostic_only": True,
    "training_performed": False,
    "annotation_file_read": True,
    "test_subset_used": False,
    "paper_performance_row_allowed": False,
    "artifacts": {
        "prediction_diagnostics": record(prediction_path),
        "checkpoint_dynamics": record(checkpoint_path),
        "focused_tests": record(root / "focused_tests.log"),
    },
}
temporary = root / "DCSR_NEGATIVE_DIAGNOSTICS_COMPLETE.json.tmp"
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(
    temporary,
    root / "DCSR_NEGATIVE_DIAGNOSTICS_COMPLETE.json",
)
PY

trap - EXIT
rm -f "$ANALYSIS_ROOT/ENGINEERING_FAILURE.json"
echo "ACTIONFORMER_DCSR_NEGATIVE_DIAGNOSTICS_COMPLETE"
