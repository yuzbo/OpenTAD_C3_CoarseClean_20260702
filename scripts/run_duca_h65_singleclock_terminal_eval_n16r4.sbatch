#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=gpugpu
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
H65_OFF_CONFIG="configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py"

"$PYTHON" -m py_compile \
  opentad/models/detectors/actionformer.py \
  tools/test.py \
  tools/bata/bootstrap_duca_h65_official_map.py
"$PYTHON" -m pytest \
  tests/test_duca_h65_cycle4_singleclock_contract.py \
  tests/test_duca_h65_bootstrap_contract.py -q

run_eval() {
  local family="$1" config="$2" checkpoint="$3" state_key="$4" gate_zero="$5" identity="$6"
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
    "data.test.ann_file=$ANNOTATION_PATH" \
    "data.test.category_file=$CATEGORY_PATH" \
    "model.backbone.custom.pretrain=$ADATAD_PRETRAIN" \
    "model.single_clock_gate_zero=$gate_zero" \
    "post_processing.save_dict=True" \
    "inference.load_from_raw_predictions=False" \
    2>&1 | tee "$family_root/eval.out"
}

# Both trained-state and EMA-state attribution pairs use the exact same
# checkpoint. Gate-zero keeps the SingleClock code path and only zeroes its
# effective residual scale.
run_eval final_on "$CLOCK_CONFIG" "$CLOCK_CHECKPOINT" state_dict False 1
run_eval final_gate_zero "$CLOCK_CONFIG" "$CLOCK_CHECKPOINT" state_dict True 1
run_eval ema_on "$CLOCK_CONFIG" "$CLOCK_CHECKPOINT" state_dict_ema False 1
run_eval ema_gate_zero "$CLOCK_CONFIG" "$CLOCK_CHECKPOINT" state_dict_ema True 1

# Re-infer the existing H65 OFF checkpoint under this exact evaluation commit.
# It is never retrained here.
run_eval h65_off_final "$H65_OFF_CONFIG" "$H65_OFF_CHECKPOINT" state_dict False 0
run_eval h65_off_ema "$H65_OFF_CONFIG" "$H65_OFF_CHECKPOINT" state_dict_ema False 0

"$PYTHON" - "$OUTPUT_ROOT/terminal_eval_receipt.json" "$EXPECTED_COMMIT" \
  "$CLOCK_CHECKPOINT" "$H65_OFF_CHECKPOINT" "$STAGE1_CHECKPOINT" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from tools.bata.duca_p0_training import atomic_write_json

output, commit, clock, off, stage1 = sys.argv[1:]
root = Path(output).resolve().parent
families = ["final_on", "final_gate_zero", "ema_on", "ema_gate_zero", "h65_off_final", "h65_off_ema"]
rows = {}
for family in families:
    metrics = root / family / "terminal_evaluation.json"
    if not metrics.is_file():
        raise RuntimeError(f"missing terminal evaluation for {family}")
    rows[family] = {
        "metrics_path": str(metrics),
        "metrics_sha256": hashlib.sha256(metrics.read_bytes()).hexdigest(),
    }
    identity = root / family / "selected_input_identity.json"
    if identity.is_file():
        rows[family]["selected_input_identity_path"] = str(identity)
        rows[family]["selected_input_identity_sha256"] = hashlib.sha256(identity.read_bytes()).hexdigest()
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
    },
)
PY
echo "[DUCA_H65_SINGLECLOCK_EVAL] complete: $OUTPUT_ROOT/terminal_eval_receipt.json"
