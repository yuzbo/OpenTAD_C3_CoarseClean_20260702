#!/usr/bin/env bash
#SBATCH --job-name=duca_rec_cost
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

source /etc/profile
set -euo pipefail

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

RUN_ROOT="${DUCA_RUN_ROOT:?DUCA_RUN_ROOT is required}"
OUTPUT_DIR="${RUN_ROOT}/cost_profile"
mkdir -p "${OUTPUT_DIR}"

COST_ARMS=(
  "duca_evidence_recovery_matched_h65_60.py:C0"
  "duca_evidence_recovery_full.py:F"
  "duca_evidence_recovery_no_merge.py:A4"
  "duca_evidence_recovery_no_recovery.py:A5"
)

for entry in "${COST_ARMS[@]}"; do
  IFS=":" read -r cfg_name arm_name <<< "${entry}"
  cfg_path="configs/adatad/thumos/${cfg_name}"
  out_path="${OUTPUT_DIR}/profile_${arm_name}.json"
  if [[ -f "${out_path}" ]]; then
    echo "[ERROR] Refusing to overwrite existing profile: ${out_path}"
    exit 1
  fi
  echo "[INFO] Profiling Arm=${arm_name}, Config=${cfg_path} -> ${out_path}"
  python tools/bata/profile_duca_evidence_recovery.py \
    --config "${cfg_path}" \
    --output "${out_path}" \
    --warmup 50 \
    --iterations 200 \
    --device cuda:0
done

echo "[DONE] Completed cost profiling for all designated arms."
