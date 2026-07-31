#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --array=0-2
#SBATCH -J af-odfcr

set -o pipefail
source /etc/profile >/dev/null 2>&1
set -eu
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

: "${CANDIDATE_ROOT:?missing CANDIDATE_ROOT}"
: "${RUN_ROOT:?missing RUN_ROOT}"
: "${OFFICIAL_DATA_ROOT:?missing OFFICIAL_DATA_ROOT}"
: "${EXPECTED_CANDIDATE_COMMIT:?missing EXPECTED_CANDIDATE_COMMIT}"
: "${EXPECTED_CANDIDATE_TREE:?missing EXPECTED_CANDIDATE_TREE}"
: "${EXPECTED_ANNOTATION_SHA256:?missing EXPECTED_ANNOTATION_SHA256}"
: "${ODFCR_INTERNAL_HOLDOUT_MANIFEST:?missing ODFCR_INTERNAL_HOLDOUT_MANIFEST}"
: "${ODFCR_INTERNAL_HOLDOUT_MANIFEST_SHA256:?missing ODFCR_INTERNAL_HOLDOUT_MANIFEST_SHA256}"
: "${ODFCR_PREVIOUS_HOLDOUT_MANIFEST:?missing ODFCR_PREVIOUS_HOLDOUT_MANIFEST}"
: "${ODFCR_PREVIOUS_HOLDOUT_MANIFEST_SHA256:?missing ODFCR_PREVIOUS_HOLDOUT_MANIFEST_SHA256}"
: "${ACTIONFORMER_PYTHON_ENV:?missing ACTIONFORMER_PYTHON_ENV}"
: "${ACTIONFORMER_ENVIRONMENT_RECEIPT:?missing ACTIONFORMER_ENVIRONMENT_RECEIPT}"
: "${EXPECTED_ENVIRONMENT_RECEIPT_SHA256:?missing EXPECTED_ENVIRONMENT_RECEIPT_SHA256}"
: "${ACTIONFORMER_NMS_EXTENSION:?missing ACTIONFORMER_NMS_EXTENSION}"
: "${EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256:?missing EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256}"

ODFCR_RUN_MODE="${ODFCR_RUN_MODE:-factorial}"
case "$ODFCR_RUN_MODE" in
  g0_only|factorial) ;;
  *) echo "unexpected ODFCR_RUN_MODE" >&2; exit 2 ;;
esac
case "${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}" in
  0) DEV_SEED=2026073101 ;;
  1) DEV_SEED=2026073102 ;;
  2) DEV_SEED=2026073103 ;;
  *) echo "unexpected ODF-CR development array index" >&2; exit 2 ;;
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
test -f "$ODFCR_INTERNAL_HOLDOUT_MANIFEST"
test -f "$ODFCR_PREVIOUS_HOLDOUT_MANIFEST"
test "$(sha256sum "$ODFCR_INTERNAL_HOLDOUT_MANIFEST" | awk '{print $1}')" = \
  "$ODFCR_INTERNAL_HOLDOUT_MANIFEST_SHA256"
test "$(sha256sum "$ODFCR_PREVIOUS_HOLDOUT_MANIFEST" | awk '{print $1}')" = \
  "$ODFCR_PREVIOUS_HOLDOUT_MANIFEST_SHA256"

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
    "schema_version": "actionformer_odfcr_engineering_failure_v1",
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
    tests/test_odfcr_heads.py \
    tests/test_odfcr_internal_holdout.py \
    tests/test_odfcr_matrix_contract.py \
    tests/test_odfcr_k384_contract.py \
    tests/test_odfcr_launcher_contract.py \
    -q
) | tee "$TASK_ROOT/focused_tests.log"

STAGE=g0_exact_equivalence
(
  cd "$CANDIDATE_ROOT"
  python -m tools.validate_odfcr_g0_equivalence \
    --device cuda:0 \
    --seed "$DEV_SEED" \
    --output "$TASK_ROOT/ODFCR_G0_EQUIVALENCE.json"
) | tee "$TASK_ROOT/g0.log"

if [ "$ODFCR_RUN_MODE" = g0_only ]; then
  trap - EXIT
  rm -f "$TASK_ROOT/ENGINEERING_FAILURE.json"
  echo "ACTIONFORMER_ODFCR_G0_ONLY_COMPLETE seed=$DEV_SEED"
  exit 0
fi

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
  config_sha256=$(sha256sum "$config" | awk '{print $1}')
  config_stem=$(basename "$config" .yaml)
  output_name="odfcr_${DEV_SEED}_${arm}"
  ckpt_dir="$WORK_ROOT/ckpt/${config_stem}_${output_name}"
  checkpoint="$ckpt_dir/epoch_035.pth.tar"
  raw_predictions="$ckpt_dir/eval_results.pkl"
  mkdir "$arm_root"

  STAGE="${arm}_train"
  export ODFCR_INTERNAL_SUBSET=train
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
    echo "resume is forbidden in the ODF-CR factorial" >&2
    return 2
  fi
  checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')

  STAGE="${arm}_holdout_saveonly"
  export ODFCR_INTERNAL_SUBSET=holdout
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
    python -m tools.evaluate_odfcr_internal_predictions \
      --annotation "$annotation" \
      --expected-annotation-sha256 "$EXPECTED_ANNOTATION_SHA256" \
      --manifest "$ODFCR_INTERNAL_HOLDOUT_MANIFEST" \
      --expected-manifest-sha256 "$ODFCR_INTERNAL_HOLDOUT_MANIFEST_SHA256" \
      --previous-manifest "$ODFCR_PREVIOUS_HOLDOUT_MANIFEST" \
      --expected-previous-manifest-sha256 "$ODFCR_PREVIOUS_HOLDOUT_MANIFEST_SHA256" \
      --raw-predictions "$raw_predictions" \
      --eval-log "$arm_root/eval.log" \
      --arm "$arm" \
      --seed "$DEV_SEED" \
      --config "$config" \
      --expected-config-sha256 "$config_sha256" \
      --checkpoint "$checkpoint" \
      --expected-checkpoint-sha256 "$checkpoint_sha256" \
      --source-commit "$EXPECTED_CANDIDATE_COMMIT" \
      --source-tree "$EXPECTED_CANDIDATE_TREE" \
      --output "$arm_root/METRIC_ATTESTATION.json"
  ) | tee "$arm_root/independent_eval.log"
}

run_arm d1_off "configs/thumos_i3d_odfcr_dev_d1_off.yaml"
run_arm d1_all "configs/thumos_i3d_odfcr_dev_d1_all.yaml"
run_arm d3_off "configs/thumos_i3d_odfcr_dev_d3_off.yaml"
run_arm d3_all "configs/thumos_i3d_odfcr_dev_d3_all.yaml"

STAGE=matrix_completion
(
  cd "$CANDIDATE_ROOT"
  python -m tools.finalize_odfcr_internal_matrix \
    --seed "$DEV_SEED" \
    --g0-receipt "$TASK_ROOT/ODFCR_G0_EQUIVALENCE.json" \
    --d1-off-metric "$TASK_ROOT/d1_off/METRIC_ATTESTATION.json" \
    --d1-all-metric "$TASK_ROOT/d1_all/METRIC_ATTESTATION.json" \
    --d3-off-metric "$TASK_ROOT/d3_off/METRIC_ATTESTATION.json" \
    --d3-all-metric "$TASK_ROOT/d3_all/METRIC_ATTESTATION.json" \
    --manifest "$ODFCR_INTERNAL_HOLDOUT_MANIFEST" \
    --previous-manifest "$ODFCR_PREVIOUS_HOLDOUT_MANIFEST" \
    --annotation "$annotation" \
    --expected-annotation-sha256 "$EXPECTED_ANNOTATION_SHA256" \
    --expected-manifest-sha256 "$ODFCR_INTERNAL_HOLDOUT_MANIFEST_SHA256" \
    --expected-previous-manifest-sha256 "$ODFCR_PREVIOUS_HOLDOUT_MANIFEST_SHA256" \
    --expected-commit "$EXPECTED_CANDIDATE_COMMIT" \
    --expected-tree "$EXPECTED_CANDIDATE_TREE" \
    --output "$TASK_ROOT/ODFCR_MATRIX_COMPLETE.json"
) | tee "$TASK_ROOT/matrix_completion.log"

trap - EXIT
rm -f "$TASK_ROOT/ENGINEERING_FAILURE.json"
echo "ACTIONFORMER_ODFCR_INTERNAL_MATRIX_COMPLETE seed=$DEV_SEED"
