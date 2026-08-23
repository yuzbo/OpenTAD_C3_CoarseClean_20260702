#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --job-name=duca-sc-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
set -euo pipefail

fail() { echo "[DUCA_H65_SINGLECLOCK_EVAL][FAIL] $*" >&2; exit 1; }
ROOT="${DUCA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}}"
ROOT="$(cd -- "$ROOT" && pwd -P)"
[[ -f "$ROOT/tools/test.py" ]] || fail "exact source root is invalid: $ROOT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
[[ "$MASTER_PORT" =~ ^[0-9]+$ && "$MASTER_PORT" -ge 1024 && "$MASTER_PORT" -le 65535 ]] || fail "MASTER_PORT must be 1024-65535"
export MASTER_PORT

ENV_ROOT="/data/run01/sczc063/yuzibo/conda_envs/opentad"
[[ -f "$ENV_ROOT/bin/activate" ]] || fail "canonical environment missing: $ENV_ROOT"
source "$ENV_ROOT/bin/activate"
PYTHON="${PYTHON:-python}"

EXPECTED_COMMIT="${DUCA_EVAL_COMMIT:?DUCA_EVAL_COMMIT is required}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "evaluation checkout revision mismatch"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "evaluation checkout must be clean"

CLOCK_CHECKPOINT="${DUCA_CLOCK_CHECKPOINT:?DUCA_CLOCK_CHECKPOINT is required}"
CLOCK_SHA256="${DUCA_CLOCK_CHECKPOINT_SHA256:?DUCA_CLOCK_CHECKPOINT_SHA256 is required}"
H65_OFF_CHECKPOINT="${DUCA_H65_OFF_CHECKPOINT:?DUCA_H65_OFF_CHECKPOINT is required}"
H65_OFF_SHA256="${DUCA_H65_OFF_CHECKPOINT_SHA256:?DUCA_H65_OFF_CHECKPOINT_SHA256 is required}"
STAGE1_CHECKPOINT="${DUCA_STAGE1_CHECKPOINT:?DUCA_STAGE1_CHECKPOINT is required}"
STAGE1_SHA256="${DUCA_STAGE1_CHECKPOINT_SHA256:?DUCA_STAGE1_CHECKPOINT_SHA256 is required}"
for path in "$CLOCK_CHECKPOINT" "$H65_OFF_CHECKPOINT" "$STAGE1_CHECKPOINT"; do
  [[ -r "$path" ]] || fail "checkpoint is not readable: $path"
done
[[ "$(sha256sum "$CLOCK_CHECKPOINT" | awk '{print $1}')" == "${CLOCK_SHA256,,}" ]] || fail "SingleClock checkpoint hash mismatch"
[[ "$(sha256sum "$H65_OFF_CHECKPOINT" | awk '{print $1}')" == "${H65_OFF_SHA256,,}" ]] || fail "H65 OFF checkpoint hash mismatch"
[[ "$(sha256sum "$STAGE1_CHECKPOINT" | awk '{print $1}')" == "${STAGE1_SHA256,,}" ]] || fail "Stage-1 checkpoint hash mismatch"
export DUCA_STAGE1_CHECKPOINT="$STAGE1_CHECKPOINT"
export DUCA_STAGE1_CHECKPOINT_SHA256="${STAGE1_SHA256,,}"
export DUCA_STAGE1_CHECKPOINT_EPOCH=29

VIDEO_ROOT="${VIDEO_ROOT:-/data/run01/sczc063/yuzibo/thumos14/raw_data/video}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json}"
CATEGORY_PATH="${CATEGORY_PATH:-/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt}"
ADATAD_PRETRAIN="${ADATAD_PRETRAIN:-/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
for path in "$VIDEO_ROOT" "$ANNOTATION_PATH" "$CATEGORY_PATH" "$ADATAD_PRETRAIN"; do
  [[ -r "$path" ]] || fail "canonical read-only resource missing: $path"
done

OUTPUT_ROOT="${DUCA_EVAL_OUTPUT_ROOT:?DUCA_EVAL_OUTPUT_ROOT is required}"
[[ ! -e "$OUTPUT_ROOT" ]] || fail "terminal evaluation output root already exists: $OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
CLOCK_CONFIG="configs/adatad/thumos/duca_h65_first_singleclock_cycle4.py"
CLOCK_GATE_ZERO_CONFIG="configs/adatad/thumos/duca_h65_first_singleclock_cycle4_gate_zero.py"
CLOCK_TRAINING_IDENTITY_CONFIG="configs/adatad/thumos/duca_h65_first_singleclock_cycle4_training_identity.py"
H65_OFF_CONFIG="configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py"
BOOTSTRAP_NONCE="${DUCA_BOOTSTRAP_NONCE:-DUCA-H65-60-TRUETIME-BRIDGE-DIRECT-v001-20260823}"
BOOTSTRAP_WORKERS="${DUCA_BOOTSTRAP_WORKERS:-8}"

"$PYTHON" -m py_compile \
  opentad/models/detectors/actionformer.py \
  tools/test.py \
  tools/bata/audit_duca_h65_terminal_checkpoint.py \
  tools/bata/bootstrap_duca_h65_official_map.py \
  tools/bata/analyze_duca_h65_singleclock_strata.py
"$PYTHON" -m pytest \
  tests/test_duca_h65_cycle4_singleclock_contract.py \
  tests/test_duca_h65_bootstrap_contract.py \
  tests/test_duca_h65_singleclock_strata.py -q

"$PYTHON" tools/bata/audit_duca_h65_terminal_checkpoint.py \
  --checkpoint "$CLOCK_CHECKPOINT" --config "$CLOCK_CONFIG" \
  --stage1 "$STAGE1_CHECKPOINT" --stage1-sha256 "$STAGE1_SHA256" \
  --family clock_on --output "$OUTPUT_ROOT/clock_checkpoint_audit.json"
"$PYTHON" tools/bata/audit_duca_h65_terminal_checkpoint.py \
  --checkpoint "$H65_OFF_CHECKPOINT" --config "$H65_OFF_CONFIG" \
  --stage1 "$STAGE1_CHECKPOINT" --stage1-sha256 "$STAGE1_SHA256" \
  --family h65_off --output "$OUTPUT_ROOT/h65_off_checkpoint_audit.json"

run_eval() {
  local family="$1" config="$2" checkpoint="$3" state_key="$4" identity="$5"
  local family_root="$OUTPUT_ROOT/$family"
  mkdir -p "$family_root"
  local identity_args=()
  if [[ "$identity" == 1 ]]; then
    identity_args=(--single-clock-identity-json "$family_root/selected_input_identity.json")
  fi
  "$PYTHON" -m torch.distributed.run --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --rdzv_id="duca-singleclock-${SLURM_JOB_ID}-${family}" \
    tools/test.py "$config" --checkpoint "$checkpoint" \
    --checkpoint-state-key "$state_key" --expected-checkpoint-epoch 59 \
    --metrics-json "$family_root/terminal_evaluation.json" \
    "${identity_args[@]}" --id 0 --seed 3407 --cfg-options \
    "work_dir=$family_root/work" \
    "dataset.test.ann_file=$ANNOTATION_PATH" \
    "dataset.test.class_map=$CATEGORY_PATH" \
    "dataset.test.data_path=$VIDEO_ROOT" \
    "evaluation.ground_truth_filename=$ANNOTATION_PATH" \
    "model.backbone.custom.pretrain=$ADATAD_PRETRAIN" \
    "post_processing.save_dict=True" \
    "inference.load_from_raw_predictions=False" \
    2>&1 | tee "$family_root/eval.out"
}

# Both trained-state and EMA-state attribution pairs use the exact same
# checkpoint. Gate-zero keeps the SingleClock code path and only zeroes its
# effective residual scale.
run_eval final_on "$CLOCK_CONFIG" "$CLOCK_CHECKPOINT" state_dict 1
run_eval final_gate_zero "$CLOCK_GATE_ZERO_CONFIG" "$CLOCK_CHECKPOINT" state_dict 1
run_eval ema_on "$CLOCK_CONFIG" "$CLOCK_CHECKPOINT" state_dict_ema 1
run_eval ema_gate_zero "$CLOCK_GATE_ZERO_CONFIG" "$CLOCK_CHECKPOINT" state_dict_ema 1

# Re-infer the existing H65 OFF checkpoint under this exact evaluation commit.
# It is never retrained here.
run_eval h65_off_final "$H65_OFF_CONFIG" "$H65_OFF_CHECKPOINT" state_dict 0
run_eval h65_off_ema "$H65_OFF_CONFIG" "$H65_OFF_CHECKPOINT" state_dict_ema 0

# Freeze distortion and short-action cutpoints from the training population only.
mkdir -p "$OUTPUT_ROOT/training_identity"
"$PYTHON" -m torch.distributed.run --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-singleclock-${SLURM_JOB_ID}-training-identity" \
  tools/test.py "$CLOCK_TRAINING_IDENTITY_CONFIG" \
  --checkpoint "$CLOCK_CHECKPOINT" --checkpoint-state-key state_dict_ema \
  --expected-checkpoint-epoch 59 --not_eval \
  --single-clock-identity-json "$OUTPUT_ROOT/training_identity/selected_input_identity.json" \
  --id 0 --seed 3407 --cfg-options \
  "work_dir=$OUTPUT_ROOT/training_identity/work" \
  "dataset.test.ann_file=$ANNOTATION_PATH" \
  "dataset.test.class_map=$CATEGORY_PATH" \
  "dataset.test.data_path=$VIDEO_ROOT" \
  "model.backbone.custom.pretrain=$ADATAD_PRETRAIN" \
  "post_processing.save_dict=False" \
  "inference.load_from_raw_predictions=False" \
  2>&1 | tee "$OUTPUT_ROOT/training_identity/eval.out"

"$PYTHON" tools/bata/analyze_duca_h65_singleclock_strata.py freeze \
  --training-identity "$OUTPUT_ROOT/training_identity/selected_input_identity.json" \
  --annotation "$ANNOTATION_PATH" \
  --output "$OUTPUT_ROOT/training_strata_freeze.json"

prediction_path() {
  "$PYTHON" - "$1" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
path = payload.get("prediction_path")
if not path:
    raise SystemExit("structured evaluation lacks prediction_path")
print(path)
PY
}
FINAL_ON_PRED="$(prediction_path "$OUTPUT_ROOT/final_on/terminal_evaluation.json")"
FINAL_ZERO_PRED="$(prediction_path "$OUTPUT_ROOT/final_gate_zero/terminal_evaluation.json")"
EMA_ON_PRED="$(prediction_path "$OUTPUT_ROOT/ema_on/terminal_evaluation.json")"
EMA_ZERO_PRED="$(prediction_path "$OUTPUT_ROOT/ema_gate_zero/terminal_evaluation.json")"
OFF_FINAL_PRED="$(prediction_path "$OUTPUT_ROOT/h65_off_final/terminal_evaluation.json")"
OFF_EMA_PRED="$(prediction_path "$OUTPUT_ROOT/h65_off_ema/terminal_evaluation.json")"

"$PYTHON" tools/bata/bootstrap_duca_h65_official_map.py \
  --prediction "final_gate_zero=$FINAL_ZERO_PRED" \
  --prediction "final_on=$FINAL_ON_PRED" \
  --prediction "h65_off_final=$OFF_FINAL_PRED" \
  --baseline final_gate_zero --annotation "$ANNOTATION_PATH" \
  --nonce "$BOOTSTRAP_NONCE" --namespace SINGLECLOCK_FINAL_PAIRED_VIDEO_BOOTSTRAP_V1 \
  --workers "$BOOTSTRAP_WORKERS" --output "$OUTPUT_ROOT/final_bootstrap.json"
"$PYTHON" tools/bata/bootstrap_duca_h65_official_map.py \
  --prediction "ema_gate_zero=$EMA_ZERO_PRED" \
  --prediction "ema_on=$EMA_ON_PRED" \
  --prediction "h65_off_ema=$OFF_EMA_PRED" \
  --baseline ema_gate_zero --annotation "$ANNOTATION_PATH" \
  --nonce "$BOOTSTRAP_NONCE" --namespace SINGLECLOCK_EMA_PAIRED_VIDEO_BOOTSTRAP_V1 \
  --workers "$BOOTSTRAP_WORKERS" --output "$OUTPUT_ROOT/ema_bootstrap.json"

"$PYTHON" tools/bata/analyze_duca_h65_singleclock_strata.py evaluate \
  --frozen "$OUTPUT_ROOT/training_strata_freeze.json" \
  --validation-identity "$OUTPUT_ROOT/ema_on/selected_input_identity.json" \
  --annotation "$ANNOTATION_PATH" --on-prediction "$EMA_ON_PRED" \
  --gate-zero-prediction "$EMA_ZERO_PRED" --nonce "$BOOTSTRAP_NONCE" \
  --workers "$BOOTSTRAP_WORKERS" --output "$OUTPUT_ROOT/strata_evidence.json"

"$PYTHON" - "$OUTPUT_ROOT/terminal_eval_receipt.json" "$EXPECTED_COMMIT" \
  "$CLOCK_CHECKPOINT" "$H65_OFF_CHECKPOINT" "$STAGE1_CHECKPOINT" \
  "$CLOCK_CONFIG" "$CLOCK_GATE_ZERO_CONFIG" "$H65_OFF_CONFIG" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from tools.bata.duca_p0_training import atomic_write_json

output, commit, clock, off, stage1, clock_config, gate_zero_config, h65_off_config = sys.argv[1:]
root = Path(output).resolve().parent
families = ["final_on", "final_gate_zero", "ema_on", "ema_gate_zero", "h65_off_final", "h65_off_ema"]
family_configs = {
    "final_on": (clock_config, False),
    "final_gate_zero": (gate_zero_config, True),
    "ema_on": (clock_config, False),
    "ema_gate_zero": (gate_zero_config, True),
    "h65_off_final": (h65_off_config, None),
    "h65_off_ema": (h65_off_config, None),
}
rows = {}
for family in families:
    metrics = root / family / "terminal_evaluation.json"
    if not metrics.is_file():
        raise RuntimeError(f"missing terminal evaluation for {family}")
    config, gate_zero = family_configs[family]
    config = Path(config).resolve()
    if not config.is_file():
        raise RuntimeError(f"missing evaluation config for {family}: {config}")
    rows[family] = {
        "metrics_path": str(metrics),
        "metrics_sha256": hashlib.sha256(metrics.read_bytes()).hexdigest(),
        "config_path": str(config),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "single_clock_gate_zero": gate_zero,
    }
    identity = root / family / "selected_input_identity.json"
    if identity.is_file():
        rows[family]["selected_input_identity_path"] = str(identity)
        rows[family]["selected_input_identity_sha256"] = hashlib.sha256(identity.read_bytes()).hexdigest()
    if family in {"final_on", "final_gate_zero", "ema_on", "ema_gate_zero"} and not identity.is_file():
        raise RuntimeError(f"missing selected-input identity for frozen twin family {family}")
atomic_write_json(
    output,
    {
        "schema_version": "duca_h65_singleclock_terminal_eval_receipt_v1",
        "git_commit": commit,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "clock_checkpoint": str(Path(clock).resolve()),
        "clock_checkpoint_sha256": hashlib.sha256(Path(clock).read_bytes()).hexdigest(),
        "h65_off_checkpoint": str(Path(off).resolve()),
        "h65_off_checkpoint_sha256": hashlib.sha256(Path(off).read_bytes()).hexdigest(),
        "stage1_checkpoint": str(Path(stage1).resolve()),
        "stage1_checkpoint_sha256": hashlib.sha256(Path(stage1).read_bytes()).hexdigest(),
        "families": rows,
        "final_bootstrap_path": str(root / "final_bootstrap.json"),
        "ema_bootstrap_path": str(root / "ema_bootstrap.json"),
        "training_strata_freeze_path": str(root / "training_strata_freeze.json"),
        "strata_evidence_path": str(root / "strata_evidence.json"),
    },
)
PY
echo "[DUCA_H65_SINGLECLOCK_EVAL] complete: $OUTPUT_ROOT/terminal_eval_receipt.json"
