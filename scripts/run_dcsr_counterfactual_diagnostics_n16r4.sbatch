#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH -J af-dcsr-cf

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
: "${SOURCE_TRAINING_COMMIT:?missing SOURCE_TRAINING_COMMIT}"
: "${SOURCE_TRAINING_TREE:?missing SOURCE_TRAINING_TREE}"
: "${EXPECTED_ANNOTATION_SHA256:?missing EXPECTED_ANNOTATION_SHA256}"
: "${DCSR_INTERNAL_HOLDOUT_MANIFEST:?missing DCSR_INTERNAL_HOLDOUT_MANIFEST}"
: "${DCSR_INTERNAL_HOLDOUT_MANIFEST_SHA256:?missing DCSR_INTERNAL_HOLDOUT_MANIFEST_SHA256}"
: "${EXPECTED_DCSR_CONFIG_SHA256:?missing EXPECTED_DCSR_CONFIG_SHA256}"
: "${ACTIONFORMER_PYTHON_ENV:?missing ACTIONFORMER_PYTHON_ENV}"
: "${ACTIONFORMER_ENVIRONMENT_RECEIPT:?missing ACTIONFORMER_ENVIRONMENT_RECEIPT}"
: "${EXPECTED_ENVIRONMENT_RECEIPT_SHA256:?missing EXPECTED_ENVIRONMENT_RECEIPT_SHA256}"
: "${ACTIONFORMER_NMS_EXTENSION:?missing ACTIONFORMER_NMS_EXTENSION}"
: "${EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256:?missing EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256}"

case "${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}" in
  0)
    DEV_SEED=2026073001
    EXPECTED_PAIR_SHA256=c924ed997a438f14e3d4660906635e2cca90b34b8ac2d2dc7c4170df2a4a5867
    ;;
  1)
    DEV_SEED=2026073002
    EXPECTED_PAIR_SHA256=a2ecb27e8485c10fe97a5319b930c1e7d49d5f918f02c64269ec6281d87f88da
    ;;
  2)
    DEV_SEED=2026073003
    EXPECTED_PAIR_SHA256=9b85c1f38f5ecb4ba7fbb9e60c39c9e6005dc8201d828aa79cce0a365cdfcd40
    ;;
  *) echo "unexpected DCSR diagnostic array index" >&2; exit 2 ;;
esac
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

TASK_ROOT="$DIAGNOSTIC_RUN_ROOT/seed_${DEV_SEED}"
test ! -e "$TASK_ROOT"
mkdir "$TASK_ROOT"
RUN_LOG="$TASK_ROOT/runtime.log"
STAGE=bootstrap
exec > >(tee -a "$RUN_LOG") 2>&1

failure_receipt() {
  status=$?
  if [ "$status" -ne 0 ]; then
    STATUS="$status" STAGE="$STAGE" TASK_ROOT="$TASK_ROOT" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["TASK_ROOT"])
payload = {
    "schema_version": "actionformer_dcsr_diagnostic_engineering_failure_v1",
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
python - <<'PY'
import numpy
import torch
import nms_1d_cpu

assert torch.__version__ == "2.0.1"
assert torch.version.cuda == "11.8"
assert numpy.__version__ == "1.23.5"
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
assert nms_1d_cpu is not None
print({
    "device": torch.cuda.get_device_name(0),
    "cuda_visible_devices_preserved": True,
})
PY

STAGE=source_identity
test "$(git -C "$DIAGNOSTIC_RUNTIME" rev-parse HEAD)" = \
  "$EXPECTED_DIAGNOSTIC_COMMIT"
test "$(git -C "$DIAGNOSTIC_RUNTIME" rev-parse 'HEAD^{tree}')" = \
  "$EXPECTED_DIAGNOSTIC_TREE"
test -z "$(git -C "$DIAGNOSTIC_RUNTIME" status --porcelain=v1 --untracked-files=all)"

annotation="$OFFICIAL_DATA_ROOT/annotations/thumos14.json"
config="$DIAGNOSTIC_RUNTIME/configs/thumos_i3d_dcsr_dev_g1_uniform.yaml"
pair_receipt="$SOURCE_RUN_ROOT/seed_${DEV_SEED}/DCSR_G1_PAIR_COMPLETE.json"
checkpoint=$(find \
  "$SOURCE_RUN_ROOT/seed_${DEV_SEED}/work/ckpt" \
  -path '*_dcsr/epoch_035.pth.tar' -type f -print)
test -f "$annotation"
test "$(sha256sum "$annotation" | awk '{print $1}')" = \
  "$EXPECTED_ANNOTATION_SHA256"
test -f "$config"
test "$(sha256sum "$config" | awk '{print $1}')" = \
  "$EXPECTED_DCSR_CONFIG_SHA256"
test -f "$pair_receipt"
test "$(sha256sum "$pair_receipt" | awk '{print $1}')" = \
  "$EXPECTED_PAIR_SHA256"
test -f "$checkpoint"
test "$(printf '%s\n' "$checkpoint" | wc -l)" -eq 1

WORK_ROOT="$TASK_ROOT/work"
mkdir -p "$WORK_ROOT/data"
ln -s "$OFFICIAL_DATA_ROOT" "$WORK_ROOT/data/thumos"
test "$(readlink -f "$WORK_ROOT/data/thumos")" = \
  "$(readlink -f "$OFFICIAL_DATA_ROOT")"
export DCSR_INTERNAL_SUBSET=holdout

run_counterfactual() {
  arm="$1"
  arm_root="$TASK_ROOT/$arm"
  predictions="$arm_root/eval_results.pkl"
  receipt="$arm_root/COUNTERFACTUAL_RECEIPT.json"
  mkdir "$arm_root"
  STAGE="${arm}_counterfactual"
  (
    cd "$WORK_ROOT"
    python -m tools.run_dcsr_counterfactual_eval \
      --config "$config" \
      --expected-config-sha256 "$EXPECTED_DCSR_CONFIG_SHA256" \
      --checkpoint "$checkpoint" \
      --manifest "$DCSR_INTERNAL_HOLDOUT_MANIFEST" \
      --expected-manifest-sha256 "$DCSR_INTERNAL_HOLDOUT_MANIFEST_SHA256" \
      --source-pair-receipt "$pair_receipt" \
      --expected-pair-receipt-sha256 "$EXPECTED_PAIR_SHA256" \
      --source-training-commit "$SOURCE_TRAINING_COMMIT" \
      --source-training-tree "$SOURCE_TRAINING_TREE" \
      --expected-code-commit "$EXPECTED_DIAGNOSTIC_COMMIT" \
      --expected-code-tree "$EXPECTED_DIAGNOSTIC_TREE" \
      --seed "$DEV_SEED" \
      --arm "$arm" \
      --predictions-output "$predictions" \
      --receipt-output "$receipt" \
      --print-freq 20
  ) 2>&1 | tee "$arm_root/run.log"
  test -f "$predictions"
  test -f "$receipt"
}

run_counterfactual scaffold_only
run_counterfactual all_query_residual

STAGE=task_completion
TASK_ROOT="$TASK_ROOT" DEV_SEED="$DEV_SEED" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["TASK_ROOT"])

def record(path):
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }

payload = {
    "schema_version": "actionformer_dcsr_counterfactual_task_complete_v1",
    "validation_pass": True,
    "diagnostic_only": True,
    "training_performed": False,
    "test_data_used": False,
    "paper_performance_row_allowed": False,
    "seed": int(os.environ["DEV_SEED"]),
    "receipts": {
        arm: record(root / arm / "COUNTERFACTUAL_RECEIPT.json")
        for arm in ("scaffold_only", "all_query_residual")
    },
}
temporary = root / "COUNTERFACTUAL_COMPLETE.json.tmp"
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, root / "COUNTERFACTUAL_COMPLETE.json")
PY

trap - EXIT
rm -f "$TASK_ROOT/ENGINEERING_FAILURE.json"
echo "ACTIONFORMER_DCSR_COUNTERFACTUAL_COMPLETE seed=$DEV_SEED"
