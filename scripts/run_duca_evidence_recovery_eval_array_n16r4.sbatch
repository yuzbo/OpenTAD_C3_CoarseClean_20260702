#!/usr/bin/env bash
#SBATCH --job-name=duca_eval_array
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

source /etc/profile
set -euo pipefail

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

export HOME=/data/run01/sczc063/yuzibo/tmp/home
export XDG_CACHE_HOME=/data/run01/sczc063/yuzibo/tmp/xdg_cache
export XDG_CONFIG_HOME=/data/run01/sczc063/yuzibo/tmp/xdg_config
export HF_HOME=/data/run01/sczc063/yuzibo/hf_cache

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

CONFIG_PATH="configs/adatad/thumos/${CFG_NAME}"
WORK_DIR="${RUN_ROOT}/${ARM_NAME}/seed_${SEED}"
CHECKPOINT="${WORK_DIR}/checkpoint/epoch_59.pth"
EVAL_DIR="${WORK_DIR}/official_eval"

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
  --cfg-options seed="${SEED}" work_dir="${WORK_DIR}"
