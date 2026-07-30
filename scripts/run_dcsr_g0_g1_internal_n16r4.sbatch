#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=55G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH -J af-dcsr-g1

set -eo pipefail
source /etc/profile >/dev/null 2>&1
set -u
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

: "${CANDIDATE_ROOT:?missing CANDIDATE_ROOT}"
: "${RUN_ROOT:?missing RUN_ROOT}"
: "${OFFICIAL_DATA_ROOT:?missing OFFICIAL_DATA_ROOT}"
: "${EXPECTED_CANDIDATE_COMMIT:?missing EXPECTED_CANDIDATE_COMMIT}"
: "${EXPECTED_CANDIDATE_TREE:?missing EXPECTED_CANDIDATE_TREE}"
: "${EXPECTED_ANNOTATION_SHA256:?missing EXPECTED_ANNOTATION_SHA256}"
: "${DCSR_INTERNAL_HOLDOUT_MANIFEST:?missing DCSR_INTERNAL_HOLDOUT_MANIFEST}"
: "${DCSR_INTERNAL_HOLDOUT_MANIFEST_SHA256:?missing DCSR_INTERNAL_HOLDOUT_MANIFEST_SHA256}"
: "${ACTIONFORMER_PYTHON_ENV:?missing ACTIONFORMER_PYTHON_ENV}"
: "${ACTIONFORMER_ENVIRONMENT_RECEIPT:?missing ACTIONFORMER_ENVIRONMENT_RECEIPT}"
: "${EXPECTED_ENVIRONMENT_RECEIPT_SHA256:?missing EXPECTED_ENVIRONMENT_RECEIPT_SHA256}"
: "${ACTIONFORMER_NMS_EXTENSION:?missing ACTIONFORMER_NMS_EXTENSION}"
: "${EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256:?missing EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256}"

case "${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}" in
  0) DEV_SEED=2026073001 ;;
  1) DEV_SEED=2026073002 ;;
  2) DEV_SEED=2026073003 ;;
  *) echo "unexpected DCSR development array index" >&2; exit 2 ;;
esac
case "$CANDIDATE_ROOT" in
  /data/run01/sczc063/yuzibo/projects/*) ;;
  *) echo "candidate root outside write boundary" >&2; exit 2 ;;
esac
case "$RUN_ROOT" in
  /data/run01/sczc063/yuzibo/projects/*/runs/*) ;;
  *) echo "run root outside write boundary" >&2; exit 2 ;;
esac
test -n "${SLURM_JOB_ID:-}"
test -n "${CUDA_VISIBLE_DEVICES:-}"
test -d "$RUN_ROOT"
test -f "$DCSR_INTERNAL_HOLDOUT_MANIFEST"
test "$(sha256sum "$DCSR_INTERNAL_HOLDOUT_MANIFEST" | awk '{print $1}')" = \
  "$DCSR_INTERNAL_HOLDOUT_MANIFEST_SHA256"

TASK_ROOT="$RUN_ROOT/seed_${DEV_SEED}"
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
    "schema_version": "actionformer_dcsr_internal_engineering_failure_v1",
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
export PYTHONPATH="$(dirname "$ACTIONFORMER_NMS_EXTENSION")${PYTHONPATH:+:$PYTHONPATH}"
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
test "$(git -C "$CANDIDATE_ROOT" rev-parse HEAD)" = \
  "$EXPECTED_CANDIDATE_COMMIT"
test "$(git -C "$CANDIDATE_ROOT" rev-parse 'HEAD^{tree}')" = \
  "$EXPECTED_CANDIDATE_TREE"
test -z "$(git -C "$CANDIDATE_ROOT" status --porcelain=v1 --untracked-files=all)"

annotation="$OFFICIAL_DATA_ROOT/annotations/thumos14.json"
features="$OFFICIAL_DATA_ROOT/i3d_features"
test -f "$annotation"
test -d "$features"
test "$(sha256sum "$annotation" | awk '{print $1}')" = \
  "$EXPECTED_ANNOTATION_SHA256"

STAGE=focused_tests
(
  cd "$CANDIDATE_ROOT"
  python -m pytest \
    tests/test_native_grid_sparse_heads.py \
    tests/test_sparsehead_official_config.py \
    tests/test_dcsr_heads.py \
    tests/test_dcsr_internal_holdout.py \
    -q
) | tee "$TASK_ROOT/focused_tests.log"

STAGE=g0_exact_equivalence
(
  cd "$CANDIDATE_ROOT"
  python tools/validate_dcsr_g0_equivalence.py \
    --device cuda:0 \
    --seed "$DEV_SEED" \
    --output "$TASK_ROOT/DCSR_G0_EQUIVALENCE.json"
) | tee "$TASK_ROOT/g0.log"

WORK_ROOT="$TASK_ROOT/work"
mkdir -p "$WORK_ROOT/data"
ln -s "$OFFICIAL_DATA_ROOT" "$WORK_ROOT/data/thumos"
test "$(readlink -f "$WORK_ROOT/data/thumos")" = \
  "$(readlink -f "$OFFICIAL_DATA_ROOT")"

run_arm() {
  arm="$1"
  config_relative="$2"
  arm_root="$TASK_ROOT/$arm"
  config="$CANDIDATE_ROOT/$config_relative"
  config_stem=$(basename "$config" .yaml)
  output_name="dcsr_dev_${DEV_SEED}_${arm}"
  ckpt_dir="$WORK_ROOT/ckpt/${config_stem}_${output_name}"
  checkpoint="$ckpt_dir/epoch_035.pth.tar"
  raw_predictions="$ckpt_dir/eval_results.pkl"
  mkdir "$arm_root"

  STAGE="${arm}_train"
  export DCSR_INTERNAL_SUBSET=train
  (
    cd "$WORK_ROOT"
    python "$CANDIDATE_ROOT/train.py" "$config" \
      --output "$output_name" --seed "$DEV_SEED" -p 10 -c 5
  ) 2>&1 | tee "$arm_root/train.log"
  test -f "$checkpoint"
  test "$(grep -c '^\[Train\]: Epoch [0-9][0-9]* finished' "$arm_root/train.log")" -eq 35
  grep -q '^\[Train\]: Epoch 34 finished' "$arm_root/train.log"
  grep -q '^All done!$' "$arm_root/train.log"
  if grep -q '=> loaded checkpoint' "$arm_root/train.log"; then
    echo "resume is forbidden in the paired G1 gate" >&2
    return 2
  fi

  STAGE="${arm}_holdout_saveonly"
  export DCSR_INTERNAL_SUBSET=holdout
  (
    cd "$WORK_ROOT"
    python "$CANDIDATE_ROOT/eval.py" "$config" "$ckpt_dir" \
      -epoch 35 --saveonly -p 20
  ) 2>&1 | tee "$arm_root/saveonly.log"
  test -f "$raw_predictions"

  STAGE="${arm}_holdout_eval"
  (
    cd "$WORK_ROOT"
    python "$CANDIDATE_ROOT/eval.py" "$config" "$ckpt_dir" \
      -epoch 35 -p 20
  ) 2>&1 | tee "$arm_root/eval.log"
  grep -q 'Average mAP:' "$arm_root/eval.log"

  STAGE="${arm}_independent_holdout_eval"
  (
    cd "$CANDIDATE_ROOT"
    python tools/evaluate_dcsr_internal_predictions.py \
      --annotation "$annotation" \
      --expected-annotation-sha256 "$EXPECTED_ANNOTATION_SHA256" \
      --manifest "$DCSR_INTERNAL_HOLDOUT_MANIFEST" \
      --expected-manifest-sha256 "$DCSR_INTERNAL_HOLDOUT_MANIFEST_SHA256" \
      --raw-predictions "$raw_predictions" \
      --eval-log "$arm_root/eval.log" \
      --output "$arm_root/METRIC_ATTESTATION.json"
  ) | tee "$arm_root/independent_eval.log"
}

run_arm dense "configs/thumos_i3d_dcsr_dev_dense.yaml"
run_arm dcsr "configs/thumos_i3d_dcsr_dev_g1_uniform.yaml"

STAGE=pair_completion
(
  cd "$CANDIDATE_ROOT"
  python tools/finalize_dcsr_internal_pair.py \
    --seed "$DEV_SEED" \
    --g0-receipt "$TASK_ROOT/DCSR_G0_EQUIVALENCE.json" \
    --dense-metric "$TASK_ROOT/dense/METRIC_ATTESTATION.json" \
    --dcsr-metric "$TASK_ROOT/dcsr/METRIC_ATTESTATION.json" \
    --manifest "$DCSR_INTERNAL_HOLDOUT_MANIFEST" \
    --expected-commit "$EXPECTED_CANDIDATE_COMMIT" \
    --expected-tree "$EXPECTED_CANDIDATE_TREE" \
    --output "$TASK_ROOT/DCSR_G1_PAIR_COMPLETE.json"
) | tee "$TASK_ROOT/pair_completion.log"

trap - EXIT
rm -f "$TASK_ROOT/ENGINEERING_FAILURE.json"
echo "ACTIONFORMER_DCSR_G0_G1_INTERNAL_PAIR_COMPLETE seed=$DEV_SEED"
