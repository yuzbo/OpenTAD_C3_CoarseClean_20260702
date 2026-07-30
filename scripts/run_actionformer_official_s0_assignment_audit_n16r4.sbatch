#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH -J af-k384-assign

set -eo pipefail
source /etc/profile >/dev/null 2>&1
set -u
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

: "${CANDIDATE_ROOT:?missing CANDIDATE_ROOT}"
: "${AUDIT_ROOT:?missing AUDIT_ROOT}"
: "${OFFICIAL_DATA_ROOT:?missing OFFICIAL_DATA_ROOT}"
: "${PARENT_PAIR_ROOT:?missing PARENT_PAIR_ROOT}"
: "${RUN_ROOT:?missing RUN_ROOT}"
: "${EXPECTED_PAIR_SHA256:?missing EXPECTED_PAIR_SHA256}"
: "${EXPECTED_CANDIDATE_COMMIT:?missing EXPECTED_CANDIDATE_COMMIT}"
: "${EXPECTED_CANDIDATE_TREE:?missing EXPECTED_CANDIDATE_TREE}"
: "${EXPECTED_AUDIT_COMMIT:?missing EXPECTED_AUDIT_COMMIT}"
: "${EXPECTED_AUDIT_TREE:?missing EXPECTED_AUDIT_TREE}"
: "${ACTIONFORMER_PYTHON_ENV:?missing ACTIONFORMER_PYTHON_ENV}"
: "${ACTIONFORMER_NMS_EXTENSION:?missing ACTIONFORMER_NMS_EXTENSION}"
: "${EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256:?missing EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256}"

case "$RUN_ROOT" in
  /data/run01/sczc063/yuzibo/projects/phystime_tad/runs/*) ;;
  *) echo "RUN_ROOT is outside the remote write boundary" >&2; exit 2 ;;
esac
for root in "$CANDIDATE_ROOT" "$AUDIT_ROOT"; do
  case "$root" in
    /data/run01/sczc063/yuzibo/projects/*) ;;
    *) echo "source root is outside the remote write boundary: $root" >&2; exit 2 ;;
  esac
done

test -n "${SLURM_JOB_ID:-}"
test -n "${CUDA_VISIBLE_DEVICES:-}"
test -d "$RUN_ROOT"
test ! -e "$RUN_ROOT/ASSIGNMENT_SUPPORT_AUDIT_COMPLETE.json"
test ! -e "$RUN_ROOT/ENGINEERING_FAILURE.json"

STAGE="bootstrap"
RUN_LOG="$RUN_ROOT/assignment_audit_runtime.log"
FAILURE_RECEIPT="$RUN_ROOT/ENGINEERING_FAILURE.json"
exec > >(tee -a "$RUN_LOG") 2>&1

failure_receipt() {
  status=$?
  if [ "$status" -ne 0 ]; then
    RUN_ROOT="$RUN_ROOT" STATUS="$status" STAGE="$STAGE" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
payload = {
    "schema_version": "actionformer_s0_assignment_audit_failure_v1",
    "failure_class": "engineering_or_protocol_failure",
    "exit_code": int(os.environ["STATUS"]),
    "stage": os.environ["STAGE"],
    "new_training": False,
    "paper_main_table_eligible": False,
    "model_result_claim_allowed": False,
}
temporary = root / "ENGINEERING_FAILURE.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, root / "ENGINEERING_FAILURE.json")
PY
  fi
}
trap failure_receipt EXIT

STAGE="python_environment"
test -x "$ACTIONFORMER_PYTHON_ENV/bin/python"
test -f "$ACTIONFORMER_NMS_EXTENSION"
test "$(sha256sum "$ACTIONFORMER_NMS_EXTENSION" | awk '{print $1}')" = \
  "$EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256"
export PATH="$ACTIONFORMER_PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$(dirname "$ACTIONFORMER_NMS_EXTENSION")${PYTHONPATH:+:$PYTHONPATH}"
hash -r
test "$(readlink -f "$(command -v python)")" = \
  "$(readlink -f "$ACTIONFORMER_PYTHON_ENV/bin/python")"
python - <<'PY'
import torch
import nms_1d_cpu

assert torch.__version__ == "2.0.1"
assert torch.version.cuda == "11.8"
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
assert nms_1d_cpu.__file__
print({
    "device": torch.cuda.get_device_name(0),
    "cuda_visible_devices_preserved": True,
})
PY

STAGE="source_identity"
candidate_commit=$(git -C "$CANDIDATE_ROOT" rev-parse HEAD)
candidate_tree=$(git -C "$CANDIDATE_ROOT" rev-parse 'HEAD^{tree}')
audit_commit=$(git -C "$AUDIT_ROOT" rev-parse HEAD)
audit_tree=$(git -C "$AUDIT_ROOT" rev-parse 'HEAD^{tree}')
test "$candidate_commit" = "$EXPECTED_CANDIDATE_COMMIT"
test "$candidate_tree" = "$EXPECTED_CANDIDATE_TREE"
test "$audit_commit" = "$EXPECTED_AUDIT_COMMIT"
test "$audit_tree" = "$EXPECTED_AUDIT_TREE"
test -z "$(git -C "$CANDIDATE_ROOT" status --porcelain=v1 --untracked-files=all)"
test -z "$(git -C "$AUDIT_ROOT" status --porcelain=v1 --untracked-files=all)"

STAGE="parent_s0_binding"
PAIR_COMPLETION="$PARENT_PAIR_ROOT/MATCHED_PAIR_COMPLETE.json"
test -f "$PAIR_COMPLETION"
test "$(sha256sum "$PAIR_COMPLETION" | awk '{print $1}')" = "$EXPECTED_PAIR_SHA256"
python "$AUDIT_ROOT/tools/bata/validate_actionformer_s0_screening.py" \
  --pair-completion "$PAIR_COMPLETION" \
  --expected-pair-sha256 "$EXPECTED_PAIR_SHA256" \
  --output "$RUN_ROOT/S0_SCREENING_VERDICT.json" \
  > "$RUN_ROOT/s0_screening_validation.log"

STAGE="input_binding"
export PAIR_COMPLETION RUN_ROOT
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

def receipt(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }

pair = json.loads(Path(os.environ["PAIR_COMPLETION"]).read_text())
bindings = {}
for arm in ("dense", "sparse"):
    arm_path = Path(pair["arms"][arm]["path"])
    arm_payload = json.loads(arm_path.read_text())
    checkpoint = Path(arm_payload["receipts"]["checkpoint"]["path"])
    config = Path(arm_payload["receipts"]["config"]["path"])
    assert receipt(checkpoint)["sha256"] == arm_payload["receipts"]["checkpoint"]["sha256"]
    assert receipt(config)["sha256"] == arm_payload["receipts"]["config"]["sha256"]
    bindings[arm] = {
        "checkpoint": receipt(checkpoint),
        "config": receipt(config),
    }
output = Path(os.environ["RUN_ROOT"]) / "INPUT_BINDINGS.json"
output.write_text(json.dumps(bindings, indent=2, sort_keys=True) + "\n")
PY

DENSE_CHECKPOINT=$(python -c \
  "import json; print(json.load(open('$RUN_ROOT/INPUT_BINDINGS.json'))['dense']['checkpoint']['path'])")
DENSE_CHECKPOINT_SHA256=$(python -c \
  "import json; print(json.load(open('$RUN_ROOT/INPUT_BINDINGS.json'))['dense']['checkpoint']['sha256'])")
SPARSE_CHECKPOINT=$(python -c \
  "import json; print(json.load(open('$RUN_ROOT/INPUT_BINDINGS.json'))['sparse']['checkpoint']['path'])")
SPARSE_CHECKPOINT_SHA256=$(python -c \
  "import json; print(json.load(open('$RUN_ROOT/INPUT_BINDINGS.json'))['sparse']['checkpoint']['sha256'])")
DENSE_CONFIG="$CANDIDATE_ROOT/configs/thumos_i3d.yaml"
SPARSE_CONFIG="$CANDIDATE_ROOT/configs/thumos_i3d_sparsehead_k384_uniform.yaml"

STAGE="focused_tests"
(
  cd "$AUDIT_ROOT"
  python -m pytest \
    tests/test_actionformer_s0_assignment_support.py \
    tests/test_actionformer_s0_screening.py \
    -q
) | tee "$RUN_ROOT/focused_tests.log"

STAGE="assignment_support_audit"
python "$AUDIT_ROOT/tools/bata/audit_actionformer_s0_assignment_support.py" \
  --candidate-root "$CANDIDATE_ROOT" \
  --dense-config "$DENSE_CONFIG" \
  --sparse-config "$SPARSE_CONFIG" \
  --official-data-root "$OFFICIAL_DATA_ROOT" \
  --dense-checkpoint "$DENSE_CHECKPOINT" \
  --dense-checkpoint-sha256 "$DENSE_CHECKPOINT_SHA256" \
  --sparse-checkpoint "$SPARSE_CHECKPOINT" \
  --sparse-checkpoint-sha256 "$SPARSE_CHECKPOINT_SHA256" \
  --expected-candidate-commit "$candidate_commit" \
  --expected-candidate-tree "$candidate_tree" \
  --output-dir "$RUN_ROOT/assignment_support" \
  --seed 1234567891 \
  --num-windows 64 \
  > "$RUN_ROOT/assignment_support_stdout.json"

STAGE="completion"
export RUN_ROOT candidate_commit candidate_tree audit_commit audit_tree
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

def receipt(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }

root = Path(os.environ["RUN_ROOT"])
report_path = root / "assignment_support" / "ASSIGNMENT_SUPPORT_AUDIT_COMPLETE.json"
report = json.loads(report_path.read_text())
assert report["validation_pass"] is True
assert report["new_training"] is False
assert report["test_gt_used"] is False
assert report["paper_main_table_eligible"] is False
assert report["primary_result_allowed"] is False
assert report["summary"]["window_count"] == 64
payload = {
    "schema_version": "actionformer_s0_assignment_audit_suite_v1",
    "status": "tested",
    "validation_pass": True,
    "issues": [],
    "new_training": False,
    "test_gt_used": False,
    "paper_main_table_eligible": False,
    "primary_result_allowed": False,
    "source": {
        "candidate_commit": os.environ["candidate_commit"],
        "candidate_tree": os.environ["candidate_tree"],
        "audit_commit": os.environ["audit_commit"],
        "audit_tree": os.environ["audit_tree"],
    },
    "receipts": {
        "parent_pair": receipt(root / "S0_SCREENING_VERDICT.json"),
        "input_bindings": receipt(root / "INPUT_BINDINGS.json"),
        "focused_tests": receipt(root / "focused_tests.log"),
        "assignment_support_audit": receipt(report_path),
    },
    "summary": report["summary"],
    "claim_boundary": (
        "diagnostic_only_training_split_assignment_support;"
        "not_main_table;not_model_selection;not_efficiency_claim"
    ),
}
output = root / "ASSIGNMENT_SUPPORT_AUDIT_COMPLETE.json"
temporary = output.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, output)
PY

trap - EXIT
rm -f "$FAILURE_RECEIPT"
echo "ACTIONFORMER_S0_ASSIGNMENT_SUPPORT_AUDIT_COMPLETE"
