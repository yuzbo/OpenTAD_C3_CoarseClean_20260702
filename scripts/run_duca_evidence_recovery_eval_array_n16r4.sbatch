#!/usr/bin/env bash
#SBATCH --job-name=duca_eval_array
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

source /etc/profile
set -euo pipefail

fail() { echo "[DUCA_EVIDENCE_EVAL][FAIL] $*" >&2; exit 1; }

PROJECT_DIR="${DUCA_REPO_ROOT:?DUCA_REPO_ROOT is required}"
EXPECTED_COMMIT="${DUCA_EVIDENCE_EXPECTED_COMMIT:?DUCA_EVIDENCE_EXPECTED_COMMIT is required}"
TRAINING_COMMIT="${DUCA_EVIDENCE_TRAINING_COMMIT:-$EXPECTED_COMMIT}"
cd "$PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || fail "checkout is not the expected commit"
git diff --quiet && git diff --cached --quiet || fail "tracked source checkout is dirty"
export DUCA_EXPECTED_COMMIT="$EXPECTED_COMMIT"
export DUCA_TRAINING_COMMIT="$TRAINING_COMMIT"

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

export HOME=/data/run01/sczc063/yuzibo/tmp/home
export XDG_CACHE_HOME=/data/run01/sczc063/yuzibo/tmp/xdg_cache
export XDG_CONFIG_HOME=/data/run01/sczc063/yuzibo/tmp/xdg_config
export HF_HOME=/data/run01/sczc063/yuzibo/hf_cache
export YUZIBO_ROOT="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
export THUMOS14_ANNOTATION_PATH="${THUMOS14_ANNOTATION_PATH:-$YUZIBO_ROOT/thumos14/annotations/thumos_14_anno.json}"
export THUMOS14_CLASS_MAP="${THUMOS14_CLASS_MAP:-$YUZIBO_ROOT/thumos14/annotations/category_idx.txt}"
export THUMOS14_TRAIN_DATA_PATH="${THUMOS14_TRAIN_DATA_PATH:-$YUZIBO_ROOT/raw/Validation Data/validation}"
export THUMOS14_TEST_DATA_PATH="${THUMOS14_TEST_DATA_PATH:-$YUZIBO_ROOT/raw/Test Data/TH14_test_set_mp4}"
export DUCA_VIDEOMAE_PRETRAIN="${DUCA_VIDEOMAE_PRETRAIN:-$YUZIBO_ROOT/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
LEDGER_ROOT="${DUCA_H65_LEDGER_ROOT:-$YUZIBO_ROOT/projects/c3_lowres_action_probe/ledger_exports/c3_official_asformer_delta_ledgers_fullgrid_ec0bb6a_20260702_065702_+0800}"
export DUCA_H65_TRAIN_LEDGER_PATH="${DUCA_H65_TRAIN_LEDGER_PATH:-$LEDGER_ROOT/train/value_transport_ledger_delta_p_action_384.jsonl}"
export DUCA_H65_VAL_LEDGER_PATH="${DUCA_H65_VAL_LEDGER_PATH:-$LEDGER_ROOT/val/value_transport_ledger_delta_p_action_384.jsonl}"
export DUCA_H65_TEST_LEDGER_PATH="${DUCA_H65_TEST_LEDGER_PATH:-$LEDGER_ROOT/test/value_transport_ledger_delta_p_action_384.jsonl}"
export DUCA_H65_LEDGER_SOURCE="c3_lowres_probe_delta_p_action"
export DUCA_H65_LEDGER_CONFIG_HASH=""

for required in \
  "$THUMOS14_ANNOTATION_PATH" "$THUMOS14_CLASS_MAP" \
  "$THUMOS14_TRAIN_DATA_PATH" "$THUMOS14_TEST_DATA_PATH" \
  "$DUCA_VIDEOMAE_PRETRAIN" "$DUCA_H65_TRAIN_LEDGER_PATH" \
  "$DUCA_H65_VAL_LEDGER_PATH" "$DUCA_H65_TEST_LEDGER_PATH"; do
  [[ -r "$required" ]] || fail "required input is not readable: $required"
done

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
RUN_ROOT="${DUCA_RUN_ROOT:?DUCA_RUN_ROOT is required}"

ARMS=(
  "duca_evidence_recovery_matched_h65_60.py"
  "duca_evidence_recovery_full.py"
  "duca_evidence_recovery_no_coverage.py"
  "duca_evidence_recovery_no_time.py"
  "duca_evidence_recovery_no_robust.py"
  "duca_evidence_recovery_no_merge.py"
  "duca_evidence_recovery_no_recovery.py"
  "duca_evidence_recovery_h65_selection.py"
)

ARM_NAMES=("C0" "F" "A1" "A2" "A3" "A4" "A5" "A6")

if [[ -n "${DUCA_SEEDS:-}" ]]; then
  read -r -a SEEDS <<< "${DUCA_SEEDS}"
else
  SEEDS=(8261 19237 31153)
fi

NUM_SEEDS=${#SEEDS[@]}
ARM_IDX=$(( TASK_ID / NUM_SEEDS ))
SEED_IDX=$(( TASK_ID % NUM_SEEDS ))

if (( TASK_ID < 0 || ARM_IDX >= ${#ARMS[@]} || SEED_IDX >= NUM_SEEDS )); then
  echo "[ERROR] SLURM_ARRAY_TASK_ID=${TASK_ID} is outside 0-$(( ${#ARMS[@]} * NUM_SEEDS - 1 ))"
  exit 1
fi

CFG_NAME="${ARMS[$ARM_IDX]}"
ARM_NAME="${ARM_NAMES[$ARM_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

CONFIG_PATH="$PROJECT_DIR/configs/adatad/thumos/${CFG_NAME}"
TRAIN_WORK_DIR="${RUN_ROOT}/${ARM_NAME}/seed_${SEED}"
CHECKPOINT="${TRAIN_WORK_DIR}/gpu1_id0/checkpoint/epoch_59.pth"
EVAL_ROOT="${DUCA_EVIDENCE_EVAL_ROOT:-$YUZIBO_ROOT/experiments/duca_evidence_eval_${EXPECTED_COMMIT:0:8}}"
EVAL_WORK_DIR="${EVAL_ROOT}/${ARM_NAME}/seed_${SEED}"
EVAL_DIR="${EVAL_WORK_DIR}/gpu1_id0/official_eval"

mkdir -p "${EVAL_DIR}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CHECKPOINT}"
  exit 1
fi

MASTER_PORT=$(( 30500 + TASK_ID % 1000 ))

echo "[INFO] Eval Task ${TASK_ID}: Arm=${ARM_NAME}, Seed=${SEED}, Checkpoint=${CHECKPOINT}"

torchrun --nproc_per_node=1 --master_port="${MASTER_PORT}" tools/test.py "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --expected-checkpoint-epoch 59 \
  --checkpoint-state-key state_dict_ema \
  --metrics-json "${EVAL_DIR}/metrics.json" \
  --seed "${SEED}" \
  --cfg-options seed="${SEED}" work_dir="${EVAL_WORK_DIR}"
