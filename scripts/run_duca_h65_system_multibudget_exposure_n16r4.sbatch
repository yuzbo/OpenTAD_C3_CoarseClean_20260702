#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --job-name=duca-h65-mb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00

source /etc/profile
set -euo pipefail

fail() { echo "[DUCA_H65_MB][FAIL] $*" >&2; exit 1; }

ROOT="${DUCA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}}"
ROOT="$(cd -- "$ROOT" && pwd -P)"
[[ -f "$ROOT/tools/train.py" ]] || fail "exact source root missing tools/train.py: $ROOT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

EXPECTED_COMMIT="${DUCA_EXPECTED_COMMIT:?DUCA_EXPECTED_COMMIT is required}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "checkout differs from DUCA_EXPECTED_COMMIT"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "formal execution requires a clean checkout"

module load cuda/11.8
module load miniforge3/24.11
ENV_ROOT="/data/run01/sczc063/yuzibo/conda_envs/opentad"
[[ -f "$ENV_ROOT/bin/activate" ]] || fail "canonical environment missing: $ENV_ROOT"
source "$ENV_ROOT/bin/activate"
PYTHON="${PYTHON:-python}"

export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-1000} % 20000))}"
[[ "$MASTER_PORT" =~ ^[0-9]+$ && "$MASTER_PORT" -ge 1024 && "$MASTER_PORT" -le 65535 ]] || fail "invalid MASTER_PORT"
export MASTER_PORT

VIDEO_ROOT="${VIDEO_ROOT:-/data/run01/sczc063/yuzibo/thumos14/raw_data/video}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/data/run01/sczc063/yuzibo/thumos14/train}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-/data/run01/sczc063/yuzibo/thumos14/test}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json}"
CATEGORY_PATH="${CATEGORY_PATH:-/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt}"
PRETRAIN="${DUCA_VIDEOMAE_PRETRAIN:-/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
PRETRAIN_SHA256="${DUCA_VIDEOMAE_PRETRAIN_SHA256:-4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251}"
STAGE1="${DUCA_STAGE1_CHECKPOINT:-/data/run01/sczc063/yuzibo/duca_h65_stage1_uniform384_cycle6_61397c0e_20260823/gpu1_id0/checkpoint/epoch_29.pth}"
STAGE1_SHA256="${DUCA_STAGE1_CHECKPOINT_SHA256:-bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3}"
for path in "$VIDEO_ROOT" "$TRAIN_DATA_ROOT" "$TEST_DATA_ROOT" "$ANNOTATION_PATH" "$CATEGORY_PATH" "$PRETRAIN" "$STAGE1"; do
  [[ -r "$path" ]] || fail "canonical resource unreadable: $path"
done

export DUCA_STAGE1_CHECKPOINT="$(realpath "$STAGE1")"
export DUCA_STAGE1_CHECKPOINT_SHA256="${STAGE1_SHA256,,}"
export DUCA_STAGE1_CHECKPOINT_EPOCH=29
export DUCA_VIDEOMAE_PRETRAIN="$(realpath "$PRETRAIN")"
export DUCA_PROFILE_RUNTIME="${DUCA_PROFILE_RUNTIME:-0}"
export DUCA_PROFILE_SYNC_CUDA="${DUCA_PROFILE_SYNC_CUDA:-1}"

CONTROL_CONFIG="configs/adatad/thumos/duca_h65_system_multibudget_exposure_control.py"
CANDIDATE_CONFIG="configs/adatad/thumos/duca_h65_system_multibudget_exposure_candidate.py"
PRE_RUN_CONFIG="configs/adatad/thumos/duca_h65_system_multibudget_exposure_pre_run.py"
VALIDATOR="tools/bata/validate_duca_h65_system_multibudget_exposure.py"
MODE="${MODE:-PRE_RUN}"
case "$MODE" in PRE_RUN|TRAIN|PREDICT) ;; *) fail "MODE must be PRE_RUN, TRAIN, or PREDICT" ;; esac

load_probabilities() {
  local report="$1"
  read -r DUCA_MB_P256 DUCA_MB_P512 < <(
    "$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8"))["probabilities"]; print(p["256"], p["512"])' "$report"
  )
  export DUCA_MB_P256 DUCA_MB_P512
}

validate_static() {
  "$PYTHON" "$VALIDATOR" \
    --video-root "$VIDEO_ROOT" \
    --annotation "$ANNOTATION_PATH" \
    --category "$CATEGORY_PATH" \
    --pretrain "$PRETRAIN" \
    --pretrain-sha256 "$PRETRAIN_SHA256" \
    --stage1 "$STAGE1" \
    --stage1-sha256 "$STAGE1_SHA256" \
    --calibration-dir "$CALIBRATION_DIR" "$@"
}

COMMON_CFG_OPTIONS=(
  "dataset.train.ann_file=$ANNOTATION_PATH"
  "dataset.train.data_path=$TRAIN_DATA_ROOT"
  "dataset.train.class_map=$CATEGORY_PATH"
  "dataset.test.ann_file=$ANNOTATION_PATH"
  "dataset.test.data_path=$TEST_DATA_ROOT"
  "dataset.test.class_map=$CATEGORY_PATH"
  "evaluation.ground_truth_filename=$ANNOTATION_PATH"
)

if [[ "$MODE" == PRE_RUN ]]; then
  RUN_ROOT="${DUCA_RUN_ROOT:?DUCA_RUN_ROOT is required for PRE_RUN}"
  case "$RUN_ROOT" in "$ROOT"|"$ROOT"/*) fail "run artifacts must remain outside the source checkout" ;; esac
  [[ ! -e "$RUN_ROOT" ]] || fail "PRE_RUN root already exists: $RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  CALIBRATION_DIR="$RUN_ROOT/calibration"
  "$PYTHON" tools/bata/prepare_duca_h65_system_multibudget_exposure.py \
    --annotation "$ANNOTATION_PATH" --output-dir "$CALIBRATION_DIR"
  load_probabilities "$CALIBRATION_DIR/pre_run_calibration.json"
  validate_static
  "$PYTHON" -m pytest \
    tests/test_duca_h65_system_multibudget_exposure.py \
    tests/test_duca_h65_60_curriculum_contract.py \
    tests/test_duca_checkpoint_resume_roundtrip.py -q
  "$PYTHON" -m py_compile \
    tools/train.py tools/test.py "$VALIDATOR" \
    tools/bata/prepare_duca_h65_system_multibudget_exposure.py \
    tools/bata/seal_duca_h65_system_multibudget_prediction.py \
    tools/bata/evaluate_duca_h65_system_multibudget_exposure.py
  export DUCA_EXPERIMENT_SEED=3407
  export DUCA_PRE_RUN_WORK_DIR="$RUN_ROOT/work"
  export DUCA_PRE_RUN_PROBE_JSON="$RUN_ROOT/training_probe.json"
  export DUCA_PRE_RUN_UPDATE_AUDIT_JSON="$RUN_ROOT/update_audit.json"
  "$PYTHON" tools/train.py "$PRE_RUN_CONFIG" --seed 3407 \
    --cfg-options "work_dir=$RUN_ROOT/work" "${COMMON_CFG_OPTIONS[@]}"
  SMOKE_CHECKPOINT="$RUN_ROOT/work/gpu1_id0/checkpoint/epoch_0.pth"
  validate_static \
    --smoke-probe "$RUN_ROOT/training_probe.json" \
    --smoke-checkpoint "$SMOKE_CHECKPOINT"
  echo "PRE_RUN_PASS calibration=$CALIBRATION_DIR checkpoint=$SMOKE_CHECKPOINT"
  exit 0
fi

CALIBRATION_DIR="${DUCA_CALIBRATION_DIR:?DUCA_CALIBRATION_DIR is required}"
[[ -f "$CALIBRATION_DIR/pre_run_calibration.json" ]] || fail "calibration report missing"
INFERENCE_ANNOTATION="$CALIBRATION_DIR/held_out_inference_annotation.json"
[[ -f "$INFERENCE_ANNOTATION" ]] || fail "label-free held-out inference annotation missing"
load_probabilities "$CALIBRATION_DIR/pre_run_calibration.json"
validate_static

if [[ "$MODE" == TRAIN ]]; then
  ARM="${ARM:?ARM is required for TRAIN}"
  SEED="${SEED:?SEED is required for TRAIN}"
  case "$ARM" in control|candidate) ;; *) fail "ARM must be control or candidate" ;; esac
  case "$SEED" in 3407|3408|3409) ;; *) fail "SEED must be 3407, 3408, or 3409" ;; esac
  RUN_ROOT="${DUCA_RUN_ROOT:?DUCA_RUN_ROOT is required for TRAIN}"
  case "$RUN_ROOT" in "$ROOT"|"$ROOT"/*) fail "run artifacts must remain outside the source checkout" ;; esac
  [[ ! -e "$RUN_ROOT" ]] || fail "training root already exists: $RUN_ROOT"
  mkdir -p "$RUN_ROOT"
  export DUCA_EXPERIMENT_SEED="$SEED"
  export DUCA_STAGE2_UPDATE_AUDIT_JSON="$RUN_ROOT/update_audit.json"
  if [[ "$ARM" == control ]]; then CFG="$CONTROL_CONFIG"; else CFG="$CANDIDATE_CONFIG"; fi
  exec "$PYTHON" tools/train.py "$CFG" --seed "$SEED" \
    --cfg-options "work_dir=$RUN_ROOT/work" "${COMMON_CFG_OPTIONS[@]}"
fi

ARM="${ARM:?ARM is required for PREDICT}"
SEED="${SEED:?SEED is required for PREDICT}"
BUDGET_VIEW="${BUDGET_VIEW:?BUDGET_VIEW is required for PREDICT}"
CHECKPOINT="${DUCA_TERMINAL_CHECKPOINT:?DUCA_TERMINAL_CHECKPOINT is required for PREDICT}"
RUN_ROOT="${DUCA_RUN_ROOT:?DUCA_RUN_ROOT is required for PREDICT}"
case "$SEED" in 3407|3408|3409) ;; *) fail "SEED must be 3407, 3408, or 3409" ;; esac
case "$RUN_ROOT" in "$ROOT"|"$ROOT"/*) fail "run artifacts must remain outside the source checkout" ;; esac
[[ ! -e "$RUN_ROOT" ]] || fail "prediction root already exists: $RUN_ROOT"
[[ -f "$CHECKPOINT" ]] || fail "terminal checkpoint unreadable: $CHECKPOINT"
mkdir -p "$RUN_ROOT"
export DUCA_EXPERIMENT_SEED="$SEED"
export DUCA_PROFILE_RUNTIME=1
export DUCA_PROFILE_SYNC_CUDA=1
unset DUCA_MB_EVAL_MANIFEST
unset DUCA_MB_EVAL_BUDGET
if [[ "$ARM" == control && "$BUDGET_VIEW" == control_k384 ]]; then
  CFG="$CONTROL_CONFIG"
elif [[ "$ARM" == candidate && "$BUDGET_VIEW" == candidate_k384 ]]; then
  CFG="$CANDIDATE_CONFIG"
  export DUCA_MB_EVAL_BUDGET=384
elif [[ "$ARM" == candidate && "$BUDGET_VIEW" == candidate_mixed ]]; then
  CFG="$CANDIDATE_CONFIG"
  export DUCA_MB_EVAL_MANIFEST="$CALIBRATION_DIR/held_out_fixed_mixed_budget_manifest.json"
else
  fail "invalid ARM/BUDGET_VIEW combination"
fi
COST_PATH="$RUN_ROOT/execution_cost.json"
"$PYTHON" tools/test.py "$CFG" \
  --checkpoint "$CHECKPOINT" --seed "$SEED" \
  --expected-checkpoint-epoch 59 --expected-successful-optimizer-updates 6000 \
  --checkpoint-state-key state_dict_ema --not_eval \
  --cfg-options \
    "work_dir=$RUN_ROOT/work" \
    "solver.test.batch_size=1" \
    "post_processing.save_dict=True" \
    "inference.execution_cost_json=$COST_PATH" \
    "dataset.test.ann_file=$INFERENCE_ANNOTATION" \
    "dataset.test.data_path=$TEST_DATA_ROOT" \
    "dataset.test.class_map=$CATEGORY_PATH" \
    "evaluation.ground_truth_filename=$INFERENCE_ANNOTATION"
PREDICTION_PATH="$RUN_ROOT/work/gpu1_id0/result_detection.json"
"$PYTHON" tools/bata/seal_duca_h65_system_multibudget_prediction.py \
  --prediction "$PREDICTION_PATH" \
  --cost "$COST_PATH" \
  --checkpoint "$CHECKPOINT" \
  --config "$CFG" \
  --calibration "$CALIBRATION_DIR/pre_run_calibration.json" \
  --held-out-ids "$CALIBRATION_DIR/held_out_video_ids.txt" \
  --inference-annotation "$INFERENCE_ANNOTATION" \
  --output "$RUN_ROOT/prediction_seal.json" \
  --expected-commit "$EXPECTED_COMMIT" \
  --arm "$ARM" --seed "$SEED" --budget-view "$BUDGET_VIEW"
echo "PREDICTION_SEALED root=$RUN_ROOT"
