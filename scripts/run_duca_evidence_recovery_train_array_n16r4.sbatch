#!/usr/bin/env bash
#SBATCH --job-name=duca_rec_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

source /etc/profile
set -euo pipefail

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd)}"
cd "${REPO_ROOT}"
YUZIBO_ROOT="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
DUCA_VIDEOMAE_PRETRAIN="${DUCA_VIDEOMAE_PRETRAIN:-${YUZIBO_ROOT}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
DUCA_H65_LEDGER_ROOT="${DUCA_H65_LEDGER_ROOT:-${YUZIBO_ROOT}/projects/c3_lowres_action_probe/ledger_exports/c3_official_asformer_delta_ledgers_20260702_052357_+0800}"
export DUCA_H65_TRAIN_LEDGER_PATH="${DUCA_H65_TRAIN_LEDGER_PATH:-${DUCA_H65_LEDGER_ROOT}/train/value_transport_ledger_delta_p_action_384.jsonl}"
export DUCA_H65_VAL_LEDGER_PATH="${DUCA_H65_VAL_LEDGER_PATH:-${DUCA_H65_LEDGER_ROOT}/val/value_transport_ledger_delta_p_action_384.jsonl}"
export DUCA_H65_TEST_LEDGER_PATH="${DUCA_H65_TEST_LEDGER_PATH:-${DUCA_H65_LEDGER_ROOT}/test/value_transport_ledger_delta_p_action_384.jsonl}"
export DUCA_H65_LEDGER_SOURCE="${DUCA_H65_LEDGER_SOURCE:-}"
export DUCA_H65_LEDGER_CONFIG_HASH="${DUCA_H65_LEDGER_CONFIG_HASH:-}"
export YUZIBO_ROOT DUCA_VIDEOMAE_PRETRAIN DUCA_H65_LEDGER_ROOT
if [[ ! -f "${DUCA_VIDEOMAE_PRETRAIN}" ]]; then
  echo "[ERROR] VideoMAE checkpoint not found: ${DUCA_VIDEOMAE_PRETRAIN}" >&2
  exit 2
fi

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
mkdir -p "${WORK_DIR}"

MASTER_PORT=$(( 29500 + TASK_ID % 1000 ))

echo "[INFO] Task ${TASK_ID}: Arm=${ARM_NAME}, Seed=${SEED}, Config=${CONFIG_PATH}, WorkDir=${WORK_DIR}"

torchrun --nproc_per_node=1 --master_port="${MASTER_PORT}" tools/train.py "${CONFIG_PATH}" \
  --seed "${SEED}" \
  --cfg-options seed="${SEED}" work_dir="${WORK_DIR}"
