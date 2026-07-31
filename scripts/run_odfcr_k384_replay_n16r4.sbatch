#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH -J af-odfcr-k384

set -eo pipefail
source /etc/profile >/dev/null 2>&1
set -u
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

: "${CANDIDATE_ROOT:?missing CANDIDATE_ROOT}"
: "${RUN_ROOT:?missing RUN_ROOT}"
: "${EXPECTED_CANDIDATE_COMMIT:?missing EXPECTED_CANDIDATE_COMMIT}"
: "${EXPECTED_CANDIDATE_TREE:?missing EXPECTED_CANDIDATE_TREE}"
: "${ODFCR_INTERNAL_HOLDOUT_MANIFEST:?missing ODFCR_INTERNAL_HOLDOUT_MANIFEST}"
: "${ODFCR_INTERNAL_HOLDOUT_MANIFEST_SHA256:?missing ODFCR_INTERNAL_HOLDOUT_MANIFEST_SHA256}"
: "${ODFCR_PREVIOUS_HOLDOUT_MANIFEST:?missing ODFCR_PREVIOUS_HOLDOUT_MANIFEST}"
: "${ODFCR_PREVIOUS_HOLDOUT_MANIFEST_SHA256:?missing ODFCR_PREVIOUS_HOLDOUT_MANIFEST_SHA256}"
: "${ACTIONFORMER_PYTHON_ENV:?missing ACTIONFORMER_PYTHON_ENV}"
: "${ACTIONFORMER_ENVIRONMENT_RECEIPT:?missing ACTIONFORMER_ENVIRONMENT_RECEIPT}"
: "${EXPECTED_ENVIRONMENT_RECEIPT_SHA256:?missing EXPECTED_ENVIRONMENT_RECEIPT_SHA256}"
: "${ACTIONFORMER_NMS_EXTENSION:?missing ACTIONFORMER_NMS_EXTENSION}"
: "${EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256:?missing EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256}"

case "${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}" in
  0) DEV_SEED=2026073101 ;;
  1) DEV_SEED=2026073102 ;;
  2) DEV_SEED=2026073103 ;;
  *) echo "unexpected ODF-CR replay array index" >&2; exit 2 ;;
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
test -x "$ACTIONFORMER_PYTHON_ENV/bin/python"
test -f "$ACTIONFORMER_ENVIRONMENT_RECEIPT"
test "$(sha256sum "$ACTIONFORMER_ENVIRONMENT_RECEIPT" | awk '{print $1}')" = \
  "$EXPECTED_ENVIRONMENT_RECEIPT_SHA256"
test -f "$ACTIONFORMER_NMS_EXTENSION"
test "$(sha256sum "$ACTIONFORMER_NMS_EXTENSION" | awk '{print $1}')" = \
  "$EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256"
export PATH="$ACTIONFORMER_PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$CANDIDATE_ROOT:$(dirname "$ACTIONFORMER_NMS_EXTENSION")${PYTHONPATH:+:$PYTHONPATH}"
export ODFCR_INTERNAL_SUBSET=holdout
hash -r
test "$(git -C "$CANDIDATE_ROOT" rev-parse HEAD)" = \
  "$EXPECTED_CANDIDATE_COMMIT"
test "$(git -C "$CANDIDATE_ROOT" rev-parse 'HEAD^{tree}')" = \
  "$EXPECTED_CANDIDATE_TREE"
test -z "$(git -C "$CANDIDATE_ROOT" status --porcelain=v1 --untracked-files=all)"

G2="$RUN_ROOT/aggregate_g2/ODFCR_G2_AGGREGATE_COMPLETE.json"
test -f "$G2"
g2_sha256=$(sha256sum "$G2" | awk '{print $1}')
G2="$G2" python - <<'PY'
import json
import os

record = json.loads(open(os.environ["G2"], encoding="utf-8").read())
assert record["validation_pass"] is True
assert record["residual_utility_gate_pass"] is True
PY

TASK_ROOT="$RUN_ROOT/seed_${DEV_SEED}"
WORK_ROOT="$TASK_ROOT/work"
MATRIX="$TASK_ROOT/ODFCR_MATRIX_COMPLETE.json"
CONFIG="$CANDIDATE_ROOT/configs/thumos_i3d_odfcr_dev_d3_all.yaml"
CKPT_DIR="$WORK_ROOT/ckpt/thumos_i3d_odfcr_dev_d3_all_odfcr_${DEV_SEED}_d3_all"
CHECKPOINT="$CKPT_DIR/epoch_035.pth.tar"
test -f "$MATRIX"
test -f "$CONFIG"
test -f "$CHECKPOINT"
matrix_sha256=$(sha256sum "$MATRIX" | awk '{print $1}')
config_sha256=$(sha256sum "$CONFIG" | awk '{print $1}')
checkpoint_sha256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')

REPLAY_ROOT="$RUN_ROOT/k384_seed_${DEV_SEED}"
test ! -e "$REPLAY_ROOT"
mkdir "$REPLAY_ROOT"
RUN_LOG="$REPLAY_ROOT/runtime.log"
STAGE=k384_replay
exec > >(tee -a "$RUN_LOG") 2>&1
failure_receipt() {
  status=$?
  if [ "$status" -ne 0 ]; then
    STATUS="$status" STAGE="$STAGE" REPLAY_ROOT="$REPLAY_ROOT" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["REPLAY_ROOT"])
payload = {
    "schema_version": "actionformer_odfcr_k384_failure_v1",
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

(
  cd "$WORK_ROOT"
  python -m tools.run_odfcr_k384_replay \
    --config "$CONFIG" \
    --expected-config-sha256 "$config_sha256" \
    --checkpoint "$CHECKPOINT" \
    --expected-checkpoint-sha256 "$checkpoint_sha256" \
    --manifest "$ODFCR_INTERNAL_HOLDOUT_MANIFEST" \
    --expected-manifest-sha256 "$ODFCR_INTERNAL_HOLDOUT_MANIFEST_SHA256" \
    --previous-manifest "$ODFCR_PREVIOUS_HOLDOUT_MANIFEST" \
    --expected-previous-manifest-sha256 "$ODFCR_PREVIOUS_HOLDOUT_MANIFEST_SHA256" \
    --source-matrix-receipt "$MATRIX" \
    --expected-matrix-receipt-sha256 "$matrix_sha256" \
    --g2-aggregate "$G2" \
    --expected-g2-aggregate-sha256 "$g2_sha256" \
    --expected-code-commit "$EXPECTED_CANDIDATE_COMMIT" \
    --expected-code-tree "$EXPECTED_CANDIDATE_TREE" \
    --seed "$DEV_SEED" \
    --predictions-output "$REPLAY_ROOT/k384_predictions.pkl" \
    --receipt-output "$REPLAY_ROOT/ODFCR_K384_REPLAY_COMPLETE.json"
) | tee "$REPLAY_ROOT/replay.log"
trap - EXIT
rm -f "$REPLAY_ROOT/ENGINEERING_FAILURE.json"
echo "ACTIONFORMER_ODFCR_K384_REPLAY_COMPLETE seed=$DEV_SEED"
