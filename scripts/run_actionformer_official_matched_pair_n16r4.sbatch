#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH -J af-k384-pair

set -eo pipefail
source /etc/profile >/dev/null 2>&1
set -u
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

: "${CANDIDATE_ROOT:?missing CANDIDATE_ROOT}"
: "${AUDIT_ROOT:?missing AUDIT_ROOT}"
: "${OFFICIAL_SOURCE_ROOT:?missing OFFICIAL_SOURCE_ROOT}"
: "${OFFICIAL_DATA_ROOT:?missing OFFICIAL_DATA_ROOT}"
: "${OFFICIAL_RECORD_ROOT:?missing OFFICIAL_RECORD_ROOT}"
: "${RUN_ROOT:?missing RUN_ROOT}"
: "${PAIR_TAG:?missing PAIR_TAG}"
: "${SOURCE_DIFF_ATTESTATION:?missing SOURCE_DIFF_ATTESTATION}"
: "${EXPECTED_SOURCE_DIFF_SHA256:?missing EXPECTED_SOURCE_DIFF_SHA256}"
: "${EXPECTED_CANDIDATE_COMMIT:?missing EXPECTED_CANDIDATE_COMMIT}"
: "${EXPECTED_CANDIDATE_TREE:?missing EXPECTED_CANDIDATE_TREE}"
: "${EXPECTED_AUDIT_COMMIT:?missing EXPECTED_AUDIT_COMMIT}"
: "${EXPECTED_AUDIT_TREE:?missing EXPECTED_AUDIT_TREE}"
: "${EXPECTED_OFFICIAL_COMMIT:?missing EXPECTED_OFFICIAL_COMMIT}"
: "${EXPECTED_OFFICIAL_TREE:?missing EXPECTED_OFFICIAL_TREE}"
: "${EXPECTED_ANNOTATION_SHA256:?missing EXPECTED_ANNOTATION_SHA256}"
: "${EXPECTED_DATA_MANIFEST_SHA256:?missing EXPECTED_DATA_MANIFEST_SHA256}"
: "${EXPECTED_FEATURE_MANIFEST_SHA256:?missing EXPECTED_FEATURE_MANIFEST_SHA256}"
: "${EXPECTED_EVALUATOR_MANIFEST_SHA256:?missing EXPECTED_EVALUATOR_MANIFEST_SHA256}"

SCREENING_SEED="${SCREENING_SEED:-1234567891}"
EXPECTED_TERMINAL_EPOCH="${EXPECTED_TERMINAL_EPOCH:-35}"
STAGE="bootstrap"

case "$PAIR_TAG" in
  *..*|*/*|*\\*|*[[:space:]]*|*[!A-Za-z0-9_.-]*)
    echo "PAIR_TAG is not path-safe: $PAIR_TAG" >&2
    exit 2
    ;;
esac
case "$RUN_ROOT" in
  /data/run01/sczc063/yuzibo/projects/phystime_tad/runs/*) ;;
  *) echo "RUN_ROOT is outside the remote write boundary" >&2; exit 2 ;;
esac
for root in "$CANDIDATE_ROOT" "$AUDIT_ROOT" "$OFFICIAL_SOURCE_ROOT"; do
  case "$root" in
    /data/run01/sczc063/yuzibo/projects/*) ;;
    *) echo "source root is outside the remote write boundary: $root" >&2; exit 2 ;;
  esac
done
test "$SCREENING_SEED" = "1234567891"
test "$EXPECTED_TERMINAL_EPOCH" = "35"
test -d "$RUN_ROOT"
test ! -e "$RUN_ROOT/MATCHED_PAIR_COMPLETE.json"
test ! -e "$RUN_ROOT/ENGINEERING_FAILURE.json"
test -n "${SLURM_JOB_ID:-}"
test -n "${CUDA_VISIBLE_DEVICES:-}"

RUN_LOG="$RUN_ROOT/matched_pair_runtime.log"
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
    "schema_version": "actionformer_official_matched_pair_failure_v1",
    "failure_class": "engineering_or_protocol_failure",
    "exit_code": int(os.environ["STATUS"]),
    "stage": os.environ["STAGE"],
    "paper_main_table_eligible": False,
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

STAGE="source_identity"
candidate_commit=$(git -C "$CANDIDATE_ROOT" rev-parse HEAD)
candidate_tree=$(git -C "$CANDIDATE_ROOT" rev-parse 'HEAD^{tree}')
audit_commit=$(git -C "$AUDIT_ROOT" rev-parse HEAD)
audit_tree=$(git -C "$AUDIT_ROOT" rev-parse 'HEAD^{tree}')
official_commit=$(git -C "$OFFICIAL_SOURCE_ROOT" rev-parse HEAD)
official_tree=$(git -C "$OFFICIAL_SOURCE_ROOT" rev-parse 'HEAD^{tree}')
test "$candidate_commit" = "$EXPECTED_CANDIDATE_COMMIT"
test "$candidate_tree" = "$EXPECTED_CANDIDATE_TREE"
test "$audit_commit" = "$EXPECTED_AUDIT_COMMIT"
test "$audit_tree" = "$EXPECTED_AUDIT_TREE"
test "$official_commit" = "$EXPECTED_OFFICIAL_COMMIT"
test "$official_tree" = "$EXPECTED_OFFICIAL_TREE"
test -z "$(git -C "$CANDIDATE_ROOT" status --porcelain=v1 --untracked-files=all)"
test -z "$(git -C "$AUDIT_ROOT" status --porcelain=v1 --untracked-files=all)"
test -z "$(git -C "$OFFICIAL_SOURCE_ROOT" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$SOURCE_DIFF_ATTESTATION" | awk '{print $1}')" = \
  "$EXPECTED_SOURCE_DIFF_SHA256"

AUDIT_ROOT="$AUDIT_ROOT" SOURCE_DIFF_ATTESTATION="$SOURCE_DIFF_ATTESTATION" \
python - <<'PY'
import importlib.util
import json
import os
from pathlib import Path

module_path = (
    Path(os.environ["AUDIT_ROOT"])
    / "tools"
    / "bata"
    / "actionformer_source_diff.py"
)
spec = importlib.util.spec_from_file_location("actionformer_source_diff", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
payload = json.loads(
    Path(os.environ["SOURCE_DIFF_ATTESTATION"]).read_text(encoding="utf-8")
)
module.validate_attestation_live(payload)
print("ACTIONFORMER_SOURCE_DIFF_LIVE_VALIDATION_PASS")
PY

STAGE="official_data_identity"
annotation="$OFFICIAL_DATA_ROOT/annotations/thumos14.json"
features="$OFFICIAL_DATA_ROOT/i3d_features"
data_manifest="$OFFICIAL_RECORD_ROOT/OFFICIAL_DATA_MANIFEST.json"
feature_manifest="$OFFICIAL_RECORD_ROOT/OFFICIAL_FEATURE_MANIFEST.json"
evaluator_manifest="$OFFICIAL_RECORD_ROOT/OFFICIAL_EVALUATOR_MANIFEST.json"
test -f "$annotation"
test -d "$features"
test -f "$data_manifest"
test -f "$feature_manifest"
test -f "$evaluator_manifest"
test "$(sha256sum "$annotation" | awk '{print $1}')" = \
  "$EXPECTED_ANNOTATION_SHA256"
test "$(sha256sum "$data_manifest" | awk '{print $1}')" = \
  "$EXPECTED_DATA_MANIFEST_SHA256"
test "$(sha256sum "$feature_manifest" | awk '{print $1}')" = \
  "$EXPECTED_FEATURE_MANIFEST_SHA256"
test "$(sha256sum "$evaluator_manifest" | awk '{print $1}')" = \
  "$EXPECTED_EVALUATOR_MANIFEST_SHA256"
test "$(find "$features" -maxdepth 1 -type f -name '*.npy' | wc -l)" -eq 413

STAGE="runtime_preflight"
python - <<'PY'
import torch

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
print({
    "device_count": torch.cuda.device_count(),
    "device_name": torch.cuda.get_device_name(0),
    "cuda_visible_devices_preserved": True,
})
PY
(
  cd "$CANDIDATE_ROOT"
  python -m pytest \
    tests/test_native_grid_sparse_heads.py \
    tests/test_sparsehead_official_config.py \
    -q
) | tee "$RUN_ROOT/candidate_focused_tests.log"
(
  cd "$AUDIT_ROOT"
  python -m pytest \
    tests/test_actionformer_official_matched_pair_launcher.py \
    tests/test_actionformer_independent_raw_evaluator.py \
    -q
) | tee "$RUN_ROOT/audit_focused_tests.log"

WORK_ROOT="$RUN_ROOT/work"
test ! -e "$WORK_ROOT"
mkdir -p "$WORK_ROOT/data"
ln -s "$OFFICIAL_DATA_ROOT" "$WORK_ROOT/data/thumos"
test "$(readlink -f "$WORK_ROOT/data/thumos")" = "$(readlink -f "$OFFICIAL_DATA_ROOT")"

run_arm() {
  arm="$1"
  config_relative="$2"
  arm_root="$RUN_ROOT/$arm"
  config="$CANDIDATE_ROOT/$config_relative"
  config_stem=$(basename "$config" .yaml)
  output_name="${PAIR_TAG}_${arm}"
  ckpt_dir="$WORK_ROOT/ckpt/${config_stem}_${output_name}"
  checkpoint="$ckpt_dir/epoch_035.pth.tar"
  train_log="$arm_root/train.log"
  saveonly_log="$arm_root/saveonly_eval.log"
  eval_log="$arm_root/official_eval.log"
  raw_predictions="$ckpt_dir/eval_results.pkl"
  independent_attestation="$arm_root/INDEPENDENT_METRIC_ATTESTATION.json"
  mkdir "$arm_root"

  STAGE="${arm}_train"
  train_start=$(date +%s)
  (
    cd "$WORK_ROOT"
    python "$CANDIDATE_ROOT/train.py" "$config" \
      --output "$output_name" -p 10 -c 5
  ) 2>&1 | tee "$train_log"
  train_end=$(date +%s)
  test -f "$checkpoint"
  test "$(grep -c '^\[Train\]: Epoch [0-9][0-9]* finished' "$train_log")" -eq 35
  grep -q '^\[Train\]: Epoch 34 finished' "$train_log"
  grep -q '^All done!$' "$train_log"
  if grep -q "=> loaded checkpoint" "$train_log"; then
    echo "resume marker is forbidden in matched training" >&2
    return 2
  fi

  ARM="$arm" CHECKPOINT="$checkpoint" OUTPUT="$arm_root/CHECKPOINT_ATTESTATION.json" \
  EXPECTED_TERMINAL_EPOCH="$EXPECTED_TERMINAL_EPOCH" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import torch

path = Path(os.environ["CHECKPOINT"])
payload = torch.load(path, map_location="cpu")
required = {"epoch", "state_dict", "state_dict_ema", "optimizer", "scheduler"}
if not required.issubset(payload):
    raise SystemExit(f"checkpoint keys missing: {sorted(required - set(payload))}")
if int(payload["epoch"]) != int(os.environ["EXPECTED_TERMINAL_EPOCH"]):
    raise SystemExit(f"terminal checkpoint epoch mismatch: {payload['epoch']}")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
record = {
    "schema_version": "actionformer_terminal_ema_checkpoint_attestation_v1",
    "validation_pass": True,
    "arm": os.environ["ARM"],
    "epoch": int(payload["epoch"]),
    "has_online_state_dict": True,
    "has_ema_state_dict": True,
    "checkpoint_path": str(path),
    "checkpoint_sha256": digest,
}
output = Path(os.environ["OUTPUT"])
temporary = output.with_suffix(".json.tmp")
temporary.write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, output)
PY

  STAGE="${arm}_saveonly_eval"
  saveonly_start=$(date +%s)
  (
    cd "$WORK_ROOT"
    python "$CANDIDATE_ROOT/eval.py" "$config" "$ckpt_dir" \
      -epoch 35 --saveonly -p 20
  ) 2>&1 | tee "$saveonly_log"
  saveonly_end=$(date +%s)
  test -f "$raw_predictions"
  grep -q "Loading from EMA model" "$saveonly_log"

  STAGE="${arm}_official_eval"
  eval_start=$(date +%s)
  (
    cd "$WORK_ROOT"
    python "$CANDIDATE_ROOT/eval.py" "$config" "$ckpt_dir" \
      -epoch 35 -p 20
  ) 2>&1 | tee "$eval_log"
  eval_end=$(date +%s)
  grep -q "Loading from EMA model" "$eval_log"
  grep -q "Average mAP:" "$eval_log"

  STAGE="${arm}_independent_eval"
  python "$AUDIT_ROOT/tools/bata/evaluate_actionformer_raw_predictions.py" \
    --official-repo "$OFFICIAL_SOURCE_ROOT" \
    --annotation "$annotation" \
    --expected-annotation-sha256 "$EXPECTED_ANNOTATION_SHA256" \
    --raw-predictions "$raw_predictions" \
    --eval-log "$eval_log" \
    --output "$independent_attestation" \
    > "$arm_root/independent_eval.log"

  export arm config config_relative checkpoint train_log saveonly_log eval_log
  export raw_predictions independent_attestation train_start train_end
  export saveonly_start saveonly_end eval_start eval_end arm_root
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

metric = json.loads(
    Path(os.environ["independent_attestation"]).read_text(encoding="utf-8")
)
arm = os.environ["arm"]
intervention = (
    None
    if arm == "dense"
    else {
        "name": "native_grid_sparse_head_k384",
        "budget": 384,
        "policy": "stratified_uniform",
        "training_loss_support": "selected_native_grid_queries",
        "interpretation": "method_intervention_not_execution_only",
    }
)
payload = {
    "schema_version": "actionformer_official_matched_arm_completion_v1",
    "validation_pass": True,
    "arm": arm,
    "seed": 1234567891,
    "new_training": True,
    "schedule": {
        "optimizer_epochs": 30,
        "warmup_epochs": 5,
        "executed_epochs": 35,
        "terminal_checkpoint_epoch": 35,
        "evaluated_weights": "state_dict_ema",
        "resume": False,
    },
    "config_relative": os.environ["config_relative"],
    "method_intervention": intervention,
    "metrics": metric["recomputed_metrics"],
    "receipts": {
        "config": receipt(os.environ["config"]),
        "checkpoint": receipt(os.environ["checkpoint"]),
        "raw_predictions": receipt(os.environ["raw_predictions"]),
        "train_log": receipt(os.environ["train_log"]),
        "saveonly_log": receipt(os.environ["saveonly_log"]),
        "eval_log": receipt(os.environ["eval_log"]),
        "independent_metric_attestation": receipt(
            os.environ["independent_attestation"]
        ),
    },
    "elapsed_seconds": {
        "training": int(os.environ["train_end"]) - int(os.environ["train_start"]),
        "saveonly_evaluation": (
            int(os.environ["saveonly_end"]) - int(os.environ["saveonly_start"])
        ),
        "official_evaluation": (
            int(os.environ["eval_end"]) - int(os.environ["eval_start"])
        ),
    },
    "claim_boundary": "same_commit_official_protocol_single_seed_screening",
    "paper_main_table_eligible": False,
    "end_to_end_cost_claim_allowed": False,
    "requires_preregistered_multiseed": True,
}
output = Path(os.environ["arm_root"]) / "ARM_COMPLETE.json"
temporary = output.with_suffix(".json.tmp")
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, output)
PY
}

run_arm dense "configs/thumos_i3d.yaml"
run_arm sparse "configs/thumos_i3d_sparsehead_k384_uniform.yaml"

STAGE="pair_completion"
export RUN_ROOT RUN_LOG SOURCE_DIFF_ATTESTATION data_manifest feature_manifest
export evaluator_manifest candidate_commit candidate_tree audit_commit audit_tree
export official_commit official_tree SCREENING_SEED
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
dense = json.loads((root / "dense" / "ARM_COMPLETE.json").read_text())
sparse = json.loads((root / "sparse" / "ARM_COMPLETE.json").read_text())
if dense["seed"] != sparse["seed"] or dense["seed"] != int(os.environ["SCREENING_SEED"]):
    raise SystemExit("paired seed mismatch")
payload = {
    "schema_version": "actionformer_official_matched_pair_completion_v1",
    "validation_pass": True,
    "issues": [],
    "experiment_role": "matched_single_seed_screening",
    "source": {
        "candidate_commit": os.environ["candidate_commit"],
        "candidate_tree": os.environ["candidate_tree"],
        "audit_commit": os.environ["audit_commit"],
        "audit_tree": os.environ["audit_tree"],
        "official_commit": os.environ["official_commit"],
        "official_tree": os.environ["official_tree"],
        "source_diff_attestation": receipt(os.environ["SOURCE_DIFF_ATTESTATION"]),
    },
    "data": {
        "data_manifest": receipt(os.environ["data_manifest"]),
        "feature_manifest": receipt(os.environ["feature_manifest"]),
        "evaluator_manifest": receipt(os.environ["evaluator_manifest"]),
    },
    "arms": {
        "dense": receipt(root / "dense" / "ARM_COMPLETE.json"),
        "sparse": receipt(root / "sparse" / "ARM_COMPLETE.json"),
    },
    "metrics": {
        "dense": dense["metrics"],
        "sparse": sparse["metrics"],
        "sparse_minus_dense_average_mAP": (
            sparse["metrics"]["average_mAP"] - dense["metrics"]["average_mAP"]
        ),
    },
    "comparability": {
        "same_candidate_commit": True,
        "same_seed": True,
        "same_official_data": True,
        "same_official_schedule": True,
        "same_terminal_epoch_35_ema_rule": True,
        "same_official_evaluator": True,
        "sparse_is_method_intervention_not_execution_only": True,
    },
    "claim_boundary": "screening_only_until_preregistered_multiseed_and_cost_suite",
    "paper_main_table_eligible": False,
    "model_route_terminal_claim_allowed": False,
    "end_to_end_cost_claim_allowed": False,
}
temporary = root / "MATCHED_PAIR_COMPLETE.json.tmp"
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, root / "MATCHED_PAIR_COMPLETE.json")
PY

trap - EXIT
rm -f "$FAILURE_RECEIPT"
echo "ACTIONFORMER_OFFICIAL_MATCHED_PAIR_SCREENING_COMPLETE"
