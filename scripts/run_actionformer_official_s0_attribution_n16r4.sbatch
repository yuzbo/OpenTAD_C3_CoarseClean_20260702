#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH -J af-k384-s0-2x2

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
: "${PARENT_PAIR_ROOT:?missing PARENT_PAIR_ROOT}"
: "${RUN_ROOT:?missing RUN_ROOT}"
: "${EXPECTED_PAIR_SHA256:?missing EXPECTED_PAIR_SHA256}"
: "${EXPECTED_CANDIDATE_COMMIT:?missing EXPECTED_CANDIDATE_COMMIT}"
: "${EXPECTED_CANDIDATE_TREE:?missing EXPECTED_CANDIDATE_TREE}"
: "${EXPECTED_AUDIT_COMMIT:?missing EXPECTED_AUDIT_COMMIT}"
: "${EXPECTED_AUDIT_TREE:?missing EXPECTED_AUDIT_TREE}"
: "${EXPECTED_OFFICIAL_COMMIT:?missing EXPECTED_OFFICIAL_COMMIT}"
: "${EXPECTED_OFFICIAL_TREE:?missing EXPECTED_OFFICIAL_TREE}"
: "${EXPECTED_ANNOTATION_SHA256:?missing EXPECTED_ANNOTATION_SHA256}"
: "${ACTIONFORMER_PYTHON_ENV:?missing ACTIONFORMER_PYTHON_ENV}"
: "${ACTIONFORMER_ENVIRONMENT_RECEIPT:?missing ACTIONFORMER_ENVIRONMENT_RECEIPT}"
: "${EXPECTED_ENVIRONMENT_RECEIPT_SHA256:?missing EXPECTED_ENVIRONMENT_RECEIPT_SHA256}"
: "${ACTIONFORMER_NMS_EXTENSION:?missing ACTIONFORMER_NMS_EXTENSION}"
: "${EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256:?missing EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256}"

case "$RUN_ROOT" in
  /data/run01/sczc063/yuzibo/projects/phystime_tad/runs/*) ;;
  *) echo "RUN_ROOT is outside the remote write boundary" >&2; exit 2 ;;
esac
case "$PARENT_PAIR_ROOT" in
  /data/run01/sczc063/yuzibo/projects/phystime_tad/runs/*) ;;
  *) echo "PARENT_PAIR_ROOT is outside the remote write boundary" >&2; exit 2 ;;
esac
for root in "$CANDIDATE_ROOT" "$AUDIT_ROOT" "$OFFICIAL_SOURCE_ROOT"; do
  case "$root" in
    /data/run01/sczc063/yuzibo/projects/*) ;;
    *) echo "source root is outside the remote write boundary: $root" >&2; exit 2 ;;
  esac
done

test -n "${SLURM_JOB_ID:-}"
test -n "${CUDA_VISIBLE_DEVICES:-}"
test -d "$RUN_ROOT"
test -f "$PARENT_PAIR_ROOT/MATCHED_PAIR_COMPLETE.json"
test ! -e "$RUN_ROOT/ATTRIBUTION_COMPLETE.json"
test ! -e "$RUN_ROOT/NEGATIVE_DIAGNOSTICS.json"
test ! -e "$RUN_ROOT/S0_NEGATIVE_ANALYSIS_SUITE_COMPLETE.json"
test ! -e "$RUN_ROOT/ENGINEERING_FAILURE.json"

STAGE="bootstrap"
RUN_LOG="$RUN_ROOT/attribution_runtime.log"
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
    "schema_version": "actionformer_s0_attribution_failure_v1",
    "failure_class": "engineering_or_protocol_failure",
    "exit_code": int(os.environ["STATUS"]),
    "stage": os.environ["STAGE"],
    "new_training": False,
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

STAGE="python_environment"
case "$ACTIONFORMER_PYTHON_ENV" in
  /data/run01/sczc063/yuzibo/projects/python_envs/*) ;;
  *) echo "ACTIONFORMER_PYTHON_ENV is outside the project boundary" >&2; exit 2 ;;
esac
test -x "$ACTIONFORMER_PYTHON_ENV/bin/python"
test -f "$ACTIONFORMER_ENVIRONMENT_RECEIPT"
test "$(sha256sum "$ACTIONFORMER_ENVIRONMENT_RECEIPT" | awk '{print $1}')" = \
  "$EXPECTED_ENVIRONMENT_RECEIPT_SHA256"
case "$ACTIONFORMER_NMS_EXTENSION" in
  "$ACTIONFORMER_PYTHON_ENV"/lib/python3.10/site-packages/nms_1d_cpu*.so) ;;
  *) echo "ACTIONFORMER_NMS_EXTENSION is outside the pinned Python environment" >&2; exit 2 ;;
esac
test -f "$ACTIONFORMER_NMS_EXTENSION"
test "$(sha256sum "$ACTIONFORMER_NMS_EXTENSION" | awk '{print $1}')" = \
  "$EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256"
export PATH="$ACTIONFORMER_PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$(dirname "$ACTIONFORMER_NMS_EXTENSION")${PYTHONPATH:+:$PYTHONPATH}"
hash -r
test "$(readlink -f "$(command -v python)")" = \
  "$(readlink -f "$ACTIONFORMER_PYTHON_ENV/bin/python")"
ACTIONFORMER_PYTHON_ENV="$ACTIONFORMER_PYTHON_ENV" \
ACTIONFORMER_NMS_EXTENSION="$ACTIONFORMER_NMS_EXTENSION" \
EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256="$EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256" \
python - <<'PY'
import hashlib
import os
import sys
from pathlib import Path

import numpy
import torch
import nms_1d_cpu

assert sys.prefix == os.environ["ACTIONFORMER_PYTHON_ENV"]
assert torch.__version__ == "2.0.1"
assert torch.version.cuda == "11.8"
assert numpy.__version__ == "1.23.5"
module_path = Path(nms_1d_cpu.__file__).resolve()
expected = Path(os.environ["ACTIONFORMER_NMS_EXTENSION"]).resolve()
assert module_path == expected
assert hashlib.sha256(module_path.read_bytes()).hexdigest() == (
    os.environ["EXPECTED_ACTIONFORMER_NMS_EXTENSION_SHA256"]
)
segments = torch.tensor([[0.0, 1.0], [0.1, 1.1]], dtype=torch.float32)
scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
detections = segments.new_empty((2, 3), device="cpu")
indices = nms_1d_cpu.softnms(
    segments,
    scores,
    detections,
    0.1,
    0.5,
    0.001,
    2,
)
assert indices.numel() == 2
assert torch.isfinite(detections).all()
print({
    "torch": torch.__version__,
    "numpy": numpy.__version__,
    "nms_module": str(module_path),
    "nms_softnms_7arg_probe": True,
})
PY

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

STAGE="parent_s0_verdict"
PAIR_COMPLETION="$PARENT_PAIR_ROOT/MATCHED_PAIR_COMPLETE.json"
test "$(sha256sum "$PAIR_COMPLETION" | awk '{print $1}')" = "$EXPECTED_PAIR_SHA256"
python "$AUDIT_ROOT/tools/bata/validate_actionformer_s0_screening.py" \
  --pair-completion "$PAIR_COMPLETION" \
  --expected-pair-sha256 "$EXPECTED_PAIR_SHA256" \
  --output "$RUN_ROOT/S0_SCREENING_VERDICT.json" \
  > "$RUN_ROOT/s0_screening_validation.log"
RUN_ROOT="$RUN_ROOT" python - <<'PY'
import json
import os
from pathlib import Path

verdict = json.loads(
    (Path(os.environ["RUN_ROOT"]) / "S0_SCREENING_VERDICT.json").read_text()
)
assert verdict["validation_pass"] is True
assert verdict["legal_model_result"] is True
assert verdict["engineering_failure"] is False
assert verdict["screening_pass"] is False
assert verdict["verdict"] == "KILL_CURRENT_K384_SELECTED_LOSS_INTERVENTION"
assert verdict["continuation"]["five_seed_main_study_authorized"] is False
assert verdict["continuation"]["no_retraining_2x2_attribution_authorized"] is True
PY

STAGE="official_data_identity"
annotation="$OFFICIAL_DATA_ROOT/annotations/thumos14.json"
features="$OFFICIAL_DATA_ROOT/i3d_features"
test -f "$annotation"
test -d "$features"
test "$(sha256sum "$annotation" | awk '{print $1}')" = \
  "$EXPECTED_ANNOTATION_SHA256"
test "$(find "$features" -maxdepth 1 -type f -name '*.npy' | wc -l)" -eq 413

STAGE="focused_compatibility_gate"
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
    tests/test_actionformer_s0_screening.py \
    tests/test_actionformer_matched_prediction_diagnostics.py \
    tests/test_actionformer_independent_raw_evaluator.py \
    tests/test_actionformer_official_s0_attribution_launcher.py \
    -q
) | tee "$RUN_ROOT/audit_focused_tests.log"

DENSE_CONFIG="$CANDIDATE_ROOT/configs/thumos_i3d.yaml"
SPARSE_CONFIG="$CANDIDATE_ROOT/configs/thumos_i3d_sparsehead_k384_uniform.yaml"
export PAIR_COMPLETION CANDIDATE_ROOT DENSE_CONFIG SPARSE_CONFIG RUN_ROOT
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

pair = json.loads(Path(os.environ["PAIR_COMPLETION"]).read_text())
bindings = {}
for arm in ("dense", "sparse"):
    arm_path = Path(pair["arms"][arm]["path"])
    arm_payload = json.loads(arm_path.read_text())
    checkpoint = Path(arm_payload["receipts"]["checkpoint"]["path"])
    config = Path(arm_payload["receipts"]["config"]["path"])
    raw_predictions = Path(arm_payload["receipts"]["raw_predictions"]["path"])
    assert sha(checkpoint) == arm_payload["receipts"]["checkpoint"]["sha256"]
    assert sha(config) == arm_payload["receipts"]["config"]["sha256"]
    assert sha(raw_predictions) == arm_payload["receipts"]["raw_predictions"]["sha256"]
    bindings[arm] = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint),
        "config_path": str(config),
        "config_sha256": sha(config),
        "raw_predictions_path": str(raw_predictions),
        "raw_predictions_sha256": sha(raw_predictions),
    }
assert Path(bindings["dense"]["config_path"]).resolve() == Path(
    os.environ["DENSE_CONFIG"]
).resolve()
assert Path(bindings["sparse"]["config_path"]).resolve() == Path(
    os.environ["SPARSE_CONFIG"]
).resolve()
output = Path(os.environ["RUN_ROOT"]) / "INPUT_BINDINGS.json"
output.write_text(json.dumps(bindings, indent=2, sort_keys=True) + "\n")
PY

DENSE_CHECKPOINT=$(python -c \
  "import json; print(json.load(open('$RUN_ROOT/INPUT_BINDINGS.json'))['dense']['checkpoint_path'])")
SPARSE_CHECKPOINT=$(python -c \
  "import json; print(json.load(open('$RUN_ROOT/INPUT_BINDINGS.json'))['sparse']['checkpoint_path'])")
DENSE_RAW_PREDICTIONS=$(python -c \
  "import json; print(json.load(open('$RUN_ROOT/INPUT_BINDINGS.json'))['dense']['raw_predictions_path'])")
SPARSE_RAW_PREDICTIONS=$(python -c \
  "import json; print(json.load(open('$RUN_ROOT/INPUT_BINDINGS.json'))['sparse']['raw_predictions_path'])")
EXPECTED_DENSE_RAW_SHA256=$(python -c \
  "import json; print(json.load(open('$RUN_ROOT/INPUT_BINDINGS.json'))['dense']['raw_predictions_sha256'])")
EXPECTED_SPARSE_RAW_SHA256=$(python -c \
  "import json; print(json.load(open('$RUN_ROOT/INPUT_BINDINGS.json'))['sparse']['raw_predictions_sha256'])")
export DENSE_CHECKPOINT SPARSE_CHECKPOINT
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import torch

def canonical_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()

records = {}
for arm, env_key in (("dense", "DENSE_CHECKPOINT"), ("sparse", "SPARSE_CHECKPOINT")):
    path = Path(os.environ[env_key]).resolve()
    payload = torch.load(path, map_location="cpu")
    assert int(payload["epoch"]) == 35
    assert "state_dict_ema" in payload
    state = payload["state_dict_ema"]
    records[arm] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "keys": sorted(state),
        "shapes": {key: list(state[key].shape) for key in sorted(state)},
        "dtypes": {key: str(state[key].dtype) for key in sorted(state)},
    }
assert records["dense"]["keys"] == records["sparse"]["keys"]
assert records["dense"]["shapes"] == records["sparse"]["shapes"]
assert records["dense"]["dtypes"] == records["sparse"]["dtypes"]
receipt = {
    "schema_version": "actionformer_s0_cross_eval_compatibility_v1",
    "validation_pass": True,
    "new_training": False,
    "dense_sparse_state_dict_key_shape_dtype_equal": True,
    "state_dict_key_shape_dtype_sha256": canonical_sha({
        "keys": records["dense"]["keys"],
        "shapes": records["dense"]["shapes"],
        "dtypes": records["dense"]["dtypes"],
    }),
    "checkpoints": {
        arm: {
            "path": record["path"],
            "sha256": record["sha256"],
        }
        for arm, record in records.items()
    },
    "paper_main_table_eligible": False,
}
output = Path(os.environ["RUN_ROOT"]) / "COMPATIBILITY_GATE.json"
output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

WORK_ROOT="$RUN_ROOT/work"
test ! -e "$WORK_ROOT"
mkdir -p "$WORK_ROOT/data" "$WORK_ROOT/ckpt"
ln -s "$OFFICIAL_DATA_ROOT" "$WORK_ROOT/data/thumos"
test "$(readlink -f "$WORK_ROOT/data/thumos")" = "$(readlink -f "$OFFICIAL_DATA_ROOT")"

run_cell() {
  cell="$1"
  training_support="$2"
  evaluation_queries="$3"
  config="$4"
  source_checkpoint="$5"
  cell_root="$RUN_ROOT/$cell"
  ckpt_dir="$WORK_ROOT/ckpt/$cell"
  checkpoint_link="$ckpt_dir/epoch_035.pth.tar"
  saveonly_log="$cell_root/saveonly_eval.log"
  eval_log="$cell_root/official_eval.log"
  raw_predictions="$ckpt_dir/eval_results.pkl"
  independent_attestation="$cell_root/INDEPENDENT_METRIC_ATTESTATION.json"
  mkdir "$cell_root" "$ckpt_dir"
  ln -s "$source_checkpoint" "$checkpoint_link"
  test "$(readlink -f "$checkpoint_link")" = "$(readlink -f "$source_checkpoint")"

  STAGE="${cell}_saveonly_eval"
  (
    cd "$WORK_ROOT"
    python "$CANDIDATE_ROOT/eval.py" "$config" "$ckpt_dir" \
      -epoch 35 --saveonly -p 20
  ) 2>&1 | tee "$saveonly_log"
  test -f "$raw_predictions"
  grep -q "Loading from EMA model" "$saveonly_log"

  STAGE="${cell}_official_eval"
  (
    cd "$WORK_ROOT"
    python "$CANDIDATE_ROOT/eval.py" "$config" "$ckpt_dir" \
      -epoch 35 -p 20
  ) 2>&1 | tee "$eval_log"
  grep -q "Loading from EMA model" "$eval_log"
  grep -q "Average mAP:" "$eval_log"

  STAGE="${cell}_independent_eval"
  python "$AUDIT_ROOT/tools/bata/evaluate_actionformer_raw_predictions.py" \
    --official-repo "$OFFICIAL_SOURCE_ROOT" \
    --annotation "$annotation" \
    --expected-annotation-sha256 "$EXPECTED_ANNOTATION_SHA256" \
    --raw-predictions "$raw_predictions" \
    --eval-log "$eval_log" \
    --output "$independent_attestation" \
    > "$cell_root/independent_eval.log"

  export cell training_support evaluation_queries config source_checkpoint
  export saveonly_log eval_log raw_predictions independent_attestation cell_root
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

attestation = json.loads(Path(os.environ["independent_attestation"]).read_text())
assert attestation["validation_pass"] is True
assert attestation["issues"] == []
assert attestation["raw_predictions"]["complete_official_test_coverage"] is True
payload = {
    "schema_version": "actionformer_s0_cross_eval_cell_v1",
    "validation_pass": True,
    "issues": [],
    "cell": os.environ["cell"],
    "training_support": os.environ["training_support"],
    "evaluation_queries": os.environ["evaluation_queries"],
    "new_training": False,
    "evaluated_weights": "state_dict_ema",
    "terminal_checkpoint_epoch": 35,
    "metrics": attestation["recomputed_metrics"],
    "receipts": {
        "config": receipt(os.environ["config"]),
        "source_checkpoint": receipt(os.environ["source_checkpoint"]),
        "raw_predictions": receipt(os.environ["raw_predictions"]),
        "saveonly_log": receipt(os.environ["saveonly_log"]),
        "eval_log": receipt(os.environ["eval_log"]),
        "independent_metric_attestation": receipt(
            os.environ["independent_attestation"]
        ),
    },
    "claim_boundary": "preregistered_no_retraining_2x2_attribution_only",
    "paper_main_table_eligible": False,
    "independent_seed_count_increment": 0,
    "end_to_end_cost_claim_allowed": False,
}
output = Path(os.environ["cell_root"]) / "CELL_COMPLETE.json"
temporary = output.with_suffix(".json.tmp")
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, output)
PY
}

run_cell \
  full_train_k384_eval \
  full_native_grid_queries \
  k384_stratified_uniform \
  "$SPARSE_CONFIG" \
  "$DENSE_CHECKPOINT"
run_cell \
  selected_train_dense_eval \
  selected_native_grid_queries \
  dense_native_grid \
  "$DENSE_CONFIG" \
  "$SPARSE_CHECKPOINT"

STAGE="attribution_completion"
export RUN_ROOT PAIR_COMPLETION candidate_commit candidate_tree
export audit_commit audit_tree official_commit official_tree
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

METRIC_KEYS = (
    "average_mAP",
    "mAP@0.3",
    "mAP@0.4",
    "mAP@0.5",
    "mAP@0.6",
    "mAP@0.7",
)

def receipt(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }

def delta(left, right):
    return {key: float(left[key]) - float(right[key]) for key in METRIC_KEYS}

def interaction(full_dense, full_k384, selected_dense, selected_k384):
    return {
        key: (
            float(selected_k384[key])
            - float(selected_dense[key])
            - float(full_k384[key])
            + float(full_dense[key])
        )
        for key in METRIC_KEYS
    }

root = Path(os.environ["RUN_ROOT"])
pair = json.loads(Path(os.environ["PAIR_COMPLETION"]).read_text())
full_dense = pair["metrics"]["dense"]
selected_k384 = pair["metrics"]["sparse"]
full_k384_record = json.loads(
    (root / "full_train_k384_eval" / "CELL_COMPLETE.json").read_text()
)
selected_dense_record = json.loads(
    (root / "selected_train_dense_eval" / "CELL_COMPLETE.json").read_text()
)
full_k384 = full_k384_record["metrics"]
selected_dense = selected_dense_record["metrics"]
payload = {
    "schema_version": "actionformer_sparsehead_s0_2x2_attribution_v1",
    "validation_pass": True,
    "issues": [],
    "new_training": False,
    "same_terminal_ema_checkpoints": True,
    "independent_seed_count_increment": 0,
    "source": {
        "candidate_commit": os.environ["candidate_commit"],
        "candidate_tree": os.environ["candidate_tree"],
        "audit_commit": os.environ["audit_commit"],
        "audit_tree": os.environ["audit_tree"],
        "official_commit": os.environ["official_commit"],
        "official_tree": os.environ["official_tree"],
    },
    "receipts": {
        "parent_pair": receipt(os.environ["PAIR_COMPLETION"]),
        "s0_verdict": receipt(root / "S0_SCREENING_VERDICT.json"),
        "compatibility_gate": receipt(root / "COMPATIBILITY_GATE.json"),
        "full_train_k384_eval": receipt(
            root / "full_train_k384_eval" / "CELL_COMPLETE.json"
        ),
        "selected_train_dense_eval": receipt(
            root / "selected_train_dense_eval" / "CELL_COMPLETE.json"
        ),
    },
    "cells": {
        "full_train_dense_eval": full_dense,
        "full_train_k384_eval": full_k384,
        "selected_train_dense_eval": selected_dense,
        "selected_train_k384_eval": selected_k384,
    },
    "contrasts": {
        "execution_k384_minus_dense_under_full_training": delta(
            full_k384, full_dense
        ),
        "selected_minus_full_training_under_dense_evaluation": delta(
            selected_dense, full_dense
        ),
        "execution_k384_minus_dense_under_selected_training": delta(
            selected_k384, selected_dense
        ),
        "selected_minus_full_training_under_k384_evaluation": delta(
            selected_k384, full_k384
        ),
        "training_by_execution_interaction": interaction(
            full_dense,
            full_k384,
            selected_dense,
            selected_k384,
        ),
    },
    "interpretation_boundary": (
        "descriptive_single_seed_causal_attribution_of_two_frozen_checkpoints;"
        "not_new_training;not_new_seeds;not_model_selection"
    ),
    "paper_main_table_eligible": False,
    "paper_ready": False,
    "model_route_terminal_claim_allowed": False,
    "end_to_end_cost_claim_allowed": False,
}
output = root / "ATTRIBUTION_COMPLETE.json"
temporary = output.with_suffix(".json.tmp")
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, output)
PY

STAGE="matched_negative_diagnostics"
python "$AUDIT_ROOT/tools/bata/analyze_actionformer_matched_predictions.py" \
  --official-repo "$OFFICIAL_SOURCE_ROOT" \
  --annotation "$annotation" \
  --expected-annotation-sha256 "$EXPECTED_ANNOTATION_SHA256" \
  --dense-raw "$DENSE_RAW_PREDICTIONS" \
  --expected-dense-raw-sha256 "$EXPECTED_DENSE_RAW_SHA256" \
  --sparse-raw "$SPARSE_RAW_PREDICTIONS" \
  --expected-sparse-raw-sha256 "$EXPECTED_SPARSE_RAW_SHA256" \
  --output "$RUN_ROOT/NEGATIVE_DIAGNOSTICS.json" \
  > "$RUN_ROOT/negative_diagnostics.log"

STAGE="negative_analysis_suite_completion"
RUN_ROOT="$RUN_ROOT" python - <<'PY'
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
verdict = json.loads((root / "S0_SCREENING_VERDICT.json").read_text())
attribution = json.loads((root / "ATTRIBUTION_COMPLETE.json").read_text())
diagnostics = json.loads((root / "NEGATIVE_DIAGNOSTICS.json").read_text())
assert verdict["validation_pass"] is True
assert verdict["screening_pass"] is False
assert attribution["validation_pass"] is True
assert attribution["new_training"] is False
assert diagnostics["validation_pass"] is True
assert diagnostics["new_training"] is False
payload = {
    "schema_version": "actionformer_s0_negative_analysis_suite_v1",
    "validation_pass": True,
    "issues": [],
    "new_training": False,
    "s0_verdict": verdict["verdict"],
    "five_seed_main_study_authorized": False,
    "detector_pipeline_cost_study_authorized": False,
    "receipts": {
        "s0_screening_verdict": receipt(root / "S0_SCREENING_VERDICT.json"),
        "attribution": receipt(root / "ATTRIBUTION_COMPLETE.json"),
        "negative_diagnostics": receipt(root / "NEGATIVE_DIAGNOSTICS.json"),
    },
    "claim_boundary": (
        "completed_single_seed_negative_analysis_evidence;"
        "no_retraining;not_main_table;not_route_terminal"
    ),
    "paper_main_table_eligible": False,
    "paper_ready": False,
    "model_route_terminal_claim_allowed": False,
    "end_to_end_cost_claim_allowed": False,
}
output = root / "S0_NEGATIVE_ANALYSIS_SUITE_COMPLETE.json"
temporary = output.with_suffix(".json.tmp")
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, output)
PY

trap - EXIT
rm -f "$FAILURE_RECEIPT"
echo "ACTIONFORMER_S0_NEGATIVE_ANALYSIS_SUITE_COMPLETE"
