#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --job-name=duca-sc-off-recovery
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
set -euo pipefail

fail() { echo "[DUCA_H65_SINGLECLOCK_OFF_RECOVERY][FAIL] $*" >&2; exit 1; }
ROOT="${DUCA_REPO_ROOT:?DUCA_REPO_ROOT is required}"
ROOT="$(cd -- "$ROOT" && pwd -P)"
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

EXPECTED_COMMIT="b2ccfccab5b4912b59954afcc9b0364955327f7c"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "evaluation checkout revision mismatch"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "evaluation checkout must be clean"

CLOCK_CHECKPOINT="/data/run01/sczc063/yuzibo/duca_h65_firstmix_singleclock_stage2_on_08a817e9_20260824/gpu1_id0/checkpoint/epoch_59.pth"
CLOCK_SHA256="720f41912eca47ccc9c0413b711d7fbc75f3a4175d983125d1a2df6015d3ae23"
H65_OFF_CHECKPOINT="/data/run01/sczc063/yuzibo/duca_h65_90_stage2_off_04c35a3b_20260823/gpu1_id0/checkpoint/epoch_59.pth"
H65_OFF_SHA256="dafcfbd0b1e0a13c400789e73ee13a20cf69551813ef62fc8185fde609806a1c"
STAGE1_CHECKPOINT="/data/run01/sczc063/yuzibo/duca_h65_stage1_uniform384_cycle6_61397c0e_20260823/gpu1_id0/checkpoint/epoch_29.pth"
STAGE1_SHA256="bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3"
for path in "$CLOCK_CHECKPOINT" "$H65_OFF_CHECKPOINT" "$STAGE1_CHECKPOINT"; do
  [[ -r "$path" ]] || fail "checkpoint is not readable: $path"
done
[[ "$(sha256sum "$CLOCK_CHECKPOINT" | awk '{print $1}')" == "$CLOCK_SHA256" ]] || fail "SingleClock checkpoint hash mismatch"
[[ "$(sha256sum "$H65_OFF_CHECKPOINT" | awk '{print $1}')" == "$H65_OFF_SHA256" ]] || fail "H65 OFF checkpoint hash mismatch"
[[ "$(sha256sum "$STAGE1_CHECKPOINT" | awk '{print $1}')" == "$STAGE1_SHA256" ]] || fail "Stage-1 checkpoint hash mismatch"
export DUCA_STAGE1_CHECKPOINT="$STAGE1_CHECKPOINT"
export DUCA_STAGE1_CHECKPOINT_SHA256="$STAGE1_SHA256"
export DUCA_STAGE1_CHECKPOINT_EPOCH=29

VIDEO_ROOT="/data/run01/sczc063/yuzibo/thumos14/raw_data/video"
ANNOTATION_PATH="/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json"
CATEGORY_PATH="/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt"
ADATAD_PRETRAIN="/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
for path in "$VIDEO_ROOT" "$ANNOTATION_PATH" "$CATEGORY_PATH" "$ADATAD_PRETRAIN"; do
  [[ -r "$path" ]] || fail "canonical read-only resource missing: $path"
done

SOURCE_ROOT="/data/run01/sczc063/yuzibo/duca_h65_singleclock_terminal_eval_b2ccfcca_r1_20260824"
OUTPUT_ROOT="/data/run01/sczc063/yuzibo/duca_h65_singleclock_off_recovery_b2ccfcca_r3_20260824"
[[ -d "$SOURCE_ROOT" ]] || fail "completed four-family source root missing"
[[ ! -e "$OUTPUT_ROOT" ]] || fail "recovery output root already exists"
for family in final_on final_gate_zero ema_on ema_gate_zero; do
  [[ -r "$SOURCE_ROOT/$family/terminal_evaluation.json" ]] || fail "missing completed family: $family"
done
mkdir -p "$OUTPUT_ROOT"

CLOCK_CONFIG="configs/adatad/thumos/duca_h65_first_singleclock_cycle4.py"
CLOCK_GATE_ZERO_CONFIG="configs/adatad/thumos/duca_h65_first_singleclock_cycle4_gate_zero.py"
CLOCK_TRAINING_IDENTITY_CONFIG="configs/adatad/thumos/duca_h65_first_singleclock_cycle4_training_identity.py"
H65_OFF_CONFIG="configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py"
BOOTSTRAP_NONCE="DUCA-H65-60-TRUETIME-BRIDGE-DIRECT-v001-20260823"
BOOTSTRAP_WORKERS=8
OFF_COMPAT_OVERRIDE="model.backbone.backbone.relative_physical_time_residual=True"

run_off_eval() {
  local family="$1" state_key="$2"
  local family_root="$OUTPUT_ROOT/$family"
  mkdir -p "$family_root"
  "$PYTHON" -m torch.distributed.run --nproc_per_node=1 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --rdzv_id="duca-singleclock-off-recovery-${SLURM_JOB_ID}-${family}" \
    tools/test.py "$H65_OFF_CONFIG" --checkpoint "$H65_OFF_CHECKPOINT" \
    --checkpoint-state-key "$state_key" --expected-checkpoint-epoch 59 \
    --metrics-json "$family_root/terminal_evaluation.json" \
    --id 0 --seed 3407 --cfg-options \
    "$OFF_COMPAT_OVERRIDE" \
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

run_off_eval h65_off_final state_dict
run_off_eval h65_off_ema state_dict_ema

# The authoritative test entrypoint uses strict load_state_dict before any
# inference.  Reaching two terminal metric receipts therefore proves both
# immutable checkpoint states loaded with no missing or unexpected keys.
"$PYTHON" - "$H65_OFF_CONFIG" "$H65_OFF_CHECKPOINT" \
  "$OUTPUT_ROOT/h65_off_final/terminal_evaluation.json" \
  "$OUTPUT_ROOT/h65_off_ema/terminal_evaluation.json" \
  "$OUTPUT_ROOT/h65_off_compatibility_prerun.json" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
from mmengine.config import Config
from tools.bata.duca_p0_training import atomic_write_json

config_path, checkpoint_path, final_path, ema_path, output_path = map(Path, sys.argv[1:])
cfg = Config.fromfile(str(config_path))
assert bool(cfg.model.get("single_clock_admission", False)) is False
rows = {}
for state_key, path in (("state_dict", final_path), ("state_dict_ema", ema_path)):
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["checkpoint_state_key"] == state_key
    assert Path(payload["checkpoint_path"]).resolve() == checkpoint_path.resolve()
    rows[state_key] = {
        "terminal_metrics_path": str(path.resolve()),
        "terminal_metrics_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "strict_load_completed_before_inference": True,
        "missing_keys": [],
        "unexpected_keys": [],
    }
atomic_write_json(output_path, {
    "schema_version": "duca_h65_off_registered_zero_compatibility_prerun_v1",
    "git_commit": "b2ccfccab5b4912b59954afcc9b0364955327f7c",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "config_path": str(config_path.resolve()),
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "checkpoint_path": str(checkpoint_path.resolve()),
    "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    "runtime_override": "model.backbone.backbone.relative_physical_time_residual=True",
    "single_clock_admission": False,
    "physical_coordinates_supplied": False,
    "registered_scalar_identity": "block0_zero_unadmitted",
    "strict_load": rows,
})
PY

# Freeze strata only from the training population, using the immutable
# SingleClock EMA checkpoint and its training-only manifest.
mkdir -p "$OUTPUT_ROOT/training_identity"
"$PYTHON" -m torch.distributed.run --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --rdzv_id="duca-singleclock-off-recovery-${SLURM_JOB_ID}-training-identity" \
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
FINAL_ON_PRED="$(prediction_path "$SOURCE_ROOT/final_on/terminal_evaluation.json")"
FINAL_ZERO_PRED="$(prediction_path "$SOURCE_ROOT/final_gate_zero/terminal_evaluation.json")"
EMA_ON_PRED="$(prediction_path "$SOURCE_ROOT/ema_on/terminal_evaluation.json")"
EMA_ZERO_PRED="$(prediction_path "$SOURCE_ROOT/ema_gate_zero/terminal_evaluation.json")"
OFF_FINAL_PRED="$(prediction_path "$OUTPUT_ROOT/h65_off_final/terminal_evaluation.json")"
OFF_EMA_PRED="$(prediction_path "$OUTPUT_ROOT/h65_off_ema/terminal_evaluation.json")"

"$PYTHON" tools/bata/bootstrap_duca_h65_official_map.py \
  --prediction "final_gate_zero=$FINAL_ZERO_PRED" \
  --prediction "final_on=$FINAL_ON_PRED" \
  --prediction "h65_off_final=$OFF_FINAL_PRED" \
  --baseline final_gate_zero --annotation "$ANNOTATION_PATH" \
  --nonce "$BOOTSTRAP_NONCE" --namespace SINGLECLOCK_FINAL_PAIRED_VIDEO_BOOTSTRAP_V1 \
  --workers "$BOOTSTRAP_WORKERS" --evaluator-thread 16 --chunksize 1 \
  --output "$OUTPUT_ROOT/final_bootstrap.json"
"$PYTHON" tools/bata/bootstrap_duca_h65_official_map.py \
  --prediction "ema_gate_zero=$EMA_ZERO_PRED" \
  --prediction "ema_on=$EMA_ON_PRED" \
  --prediction "h65_off_ema=$OFF_EMA_PRED" \
  --baseline ema_gate_zero --annotation "$ANNOTATION_PATH" \
  --nonce "$BOOTSTRAP_NONCE" --namespace SINGLECLOCK_EMA_PAIRED_VIDEO_BOOTSTRAP_V1 \
  --workers "$BOOTSTRAP_WORKERS" --evaluator-thread 16 --chunksize 1 \
  --output "$OUTPUT_ROOT/ema_bootstrap.json"

"$PYTHON" tools/bata/analyze_duca_h65_singleclock_strata.py evaluate \
  --frozen "$OUTPUT_ROOT/training_strata_freeze.json" \
  --validation-identity "$SOURCE_ROOT/ema_on/selected_input_identity.json" \
  --annotation "$ANNOTATION_PATH" --on-prediction "$EMA_ON_PRED" \
  --gate-zero-prediction "$EMA_ZERO_PRED" --nonce "$BOOTSTRAP_NONCE" \
  --workers "$BOOTSTRAP_WORKERS" --evaluator-thread 16 --chunksize 1 \
  --output "$OUTPUT_ROOT/strata_evidence.json"

# Assemble a six-family receipt without copying or mutating the four completed
# family artifacts from Job 1253023.
"$PYTHON" - "$OUTPUT_ROOT/terminal_eval_receipt.json" "$SOURCE_ROOT" "$OUTPUT_ROOT" \
  "$CLOCK_CHECKPOINT" "$H65_OFF_CHECKPOINT" "$STAGE1_CHECKPOINT" \
  "$CLOCK_CONFIG" "$CLOCK_GATE_ZERO_CONFIG" "$H65_OFF_CONFIG" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
from tools.bata.duca_p0_training import atomic_write_json

output, source_root, recovery_root, clock, off, stage1, clock_config, zero_config, off_config = sys.argv[1:]
source_root, recovery_root = Path(source_root), Path(recovery_root)
specs = {
    "final_on": (source_root, clock_config, False),
    "final_gate_zero": (source_root, zero_config, True),
    "ema_on": (source_root, clock_config, False),
    "ema_gate_zero": (source_root, zero_config, True),
    "h65_off_final": (recovery_root, off_config, None),
    "h65_off_ema": (recovery_root, off_config, None),
}
rows = {}
for family, (root, config, gate_zero) in specs.items():
    metrics = root / family / "terminal_evaluation.json"
    if not metrics.is_file():
        raise RuntimeError(f"missing terminal evaluation for {family}")
    config = Path(config).resolve()
    row = {
        "metrics_path": str(metrics.resolve()),
        "metrics_sha256": hashlib.sha256(metrics.read_bytes()).hexdigest(),
        "config_path": str(config),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "single_clock_gate_zero": gate_zero,
    }
    identity = root / family / "selected_input_identity.json"
    if identity.is_file():
        row["selected_input_identity_path"] = str(identity.resolve())
        row["selected_input_identity_sha256"] = hashlib.sha256(identity.read_bytes()).hexdigest()
    rows[family] = row
atomic_write_json(output, {
    "schema_version": "duca_h65_singleclock_terminal_eval_receipt_v1",
    "git_commit": "b2ccfccab5b4912b59954afcc9b0364955327f7c",
    "slurm_job_ids": ["1253023", os.environ.get("SLURM_JOB_ID")],
    "operational_recovery": "h65_off_registered_zero_unadmitted",
    "compatibility_prerun_path": str((recovery_root / "h65_off_compatibility_prerun.json").resolve()),
    "clock_checkpoint": str(Path(clock).resolve()),
    "clock_checkpoint_sha256": hashlib.sha256(Path(clock).read_bytes()).hexdigest(),
    "h65_off_checkpoint": str(Path(off).resolve()),
    "h65_off_checkpoint_sha256": hashlib.sha256(Path(off).read_bytes()).hexdigest(),
    "stage1_checkpoint": str(Path(stage1).resolve()),
    "stage1_checkpoint_sha256": hashlib.sha256(Path(stage1).read_bytes()).hexdigest(),
    "families": rows,
    "final_bootstrap_path": str((recovery_root / "final_bootstrap.json").resolve()),
    "ema_bootstrap_path": str((recovery_root / "ema_bootstrap.json").resolve()),
    "training_strata_freeze_path": str((recovery_root / "training_strata_freeze.json").resolve()),
    "strata_evidence_path": str((recovery_root / "strata_evidence.json").resolve()),
})
PY

echo "[DUCA_H65_SINGLECLOCK_OFF_RECOVERY] complete: $OUTPUT_ROOT/terminal_eval_receipt.json"
