#!/usr/bin/env bash
#SBATCH --job-name=duca-ev-resume-check
#SBATCH --partition=gpu
#SBATCH --account=sczc063
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00

source /etc/profile
set -euo pipefail

cd "${DUCA_REPO_ROOT:?DUCA_REPO_ROOT is required}"
export DUCA_SEEDS=8261
export PRECHECK_ONLY=1
export DUCA_RUN_ROOT="${DUCA_RESUME_PRECHECK_ROOT:?a new precheck root is required}"
BASE=/data/run01/sczc063/yuzibo/experiments

bash scripts/run_duca_evidence_recovery_admission_n16r4.sbatch
for item in "2 A1" "7 A6" "1 F" "4 A3" "5 A4" "6 A5"; do
  read -r SLURM_ARRAY_TASK_ID ARM <<< "$item"
  export SLURM_ARRAY_TASK_ID
  if [[ "$ARM" == A1 || "$ARM" == A6 ]]; then
    export DUCA_RESUME_SOURCE_COMMIT=1570a72507491899a50767700a35a04eee3f5fe9
    OLD_ROOT="$BASE/duca_evidence_optimizer_1570a725"
    EPOCH=39
  else
    export DUCA_RESUME_SOURCE_COMMIT=73bdd34ae21c6c675a00d1927b8af7c12f9edc05
    OLD_ROOT="$BASE/duca_evidence_remaining_73bdd34a"
    EPOCH=29
  fi
  export DUCA_RESUME_CHECKPOINT="$OLD_ROOT/formal/$ARM/seed_8261/gpu1_id0/checkpoint/epoch_$EPOCH.pth"
  echo "RESUME_PRECHECK_START arm=$ARM source=$DUCA_RESUME_SOURCE_COMMIT checkpoint=$DUCA_RESUME_CHECKPOINT"
  bash scripts/run_duca_evidence_recovery_train_array_n16r4.sbatch
done
echo DUCA_EVIDENCE_ALL_RESUME_PRECHECKS_OK
