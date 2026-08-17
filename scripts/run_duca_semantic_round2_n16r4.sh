#!/usr/bin/env bash
set -euo pipefail

# Future-only contract.  This launcher is intentionally not submitted or run
# from a login node; PRE_RUN=1 performs validation and exits.
: "${PRE_RUN:=1}"
CONFIG="configs/adatad/thumos/duca_semantic_indirect_six_arm_n16r4.py"
DATA_ROOT="/data/run01/sczc063/yuzibo/thumos14/raw_data/video"
ANN="/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json"
CAT="/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt"
RESUME="${RESUME:-}"
[[ -f "$CONFIG" ]] || { echo "missing config: $CONFIG"; exit 2; }
[[ "$DATA_ROOT" == /data/run01/* && "$ANN" == /data/run01/* && "$CAT" == /data/run01/* ]] || exit 2
if [[ "$PRE_RUN" == 1 ]]; then
  python -m py_compile "$CONFIG"
  echo "PRE_RUN_ONLY config=$CONFIG data=$DATA_ROOT ann=$ANN category=$CAT"
  exit 0
fi
[[ -n "$SLURM_JOB_ID" ]] || { echo "refusing non-Slurm execution"; exit 3; }
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
args=(tools/train.py "$CONFIG" --launcher slurm --work-dir "${WORK_DIR:-work_dirs/duca_round2}")
[[ -n "$RESUME" ]] && args+=(--resume-from "$RESUME")
exec srun python "${args[@]}"
