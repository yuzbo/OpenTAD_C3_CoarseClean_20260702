#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
SEED=3407
STAGE=""
MAX_CONCURRENT=4
REUSE_M00=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --max-concurrent) MAX_CONCURRENT="$2"; shift 2;;
    --reuse-m00) REUSE_M00=1; shift;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[[ "$STAGE" == geometry || "$STAGE" == mechanism ]] || { echo "--stage geometry|mechanism required" >&2; exit 2; }
[[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]] || { echo "--max-concurrent must be positive" >&2; exit 2; }
cd "$PROJECT_DIR"
submit() {
  local name="$1" cfg="$2" dep="$3" dep_arg=()
  [[ -n "$dep" ]] && dep_arg=(--dependency="afterok:${dep}")
  sbatch --parsable --partition=gpu --gres=gpu:1 --cpus-per-task=4 --time=7-00:00:00 \
    --job-name="$name" --output="${BASE}/slurm_logs/%x_%j.out" --error="${BASE}/slurm_logs/%x_%j.err" \
    "${dep_arg[@]}" --wrap="bash -lc 'set -euo pipefail; source /etc/profile; module load cuda/11.8; module load miniforge3/24.11; cd \"${PROJECT_DIR}\"; BASE=\"${BASE}\" REPO_ROOT=\"${PROJECT_DIR}\" PROJECT_DIR=\"${PROJECT_DIR}\" SEED=\"${SEED}\" bash scripts/run_duca_ct_dp_revised_thumos_gpu.sh \"${cfg}\"'"
}
if [[ "$STAGE" == geometry ]]; then
  arms=(g0 g1 g2 g3)
  jobs=()
  for i in "${!arms[@]}"; do
    arm="${arms[$i]}"; dep=""
    if (( i >= MAX_CONCURRENT )); then dep="${jobs[$((i-MAX_CONCURRENT))]}"; fi
    job="$(submit "ctdp-${arm}-s${SEED}" "configs/adatad/thumos/duca_ctdp_geometry_${arm}.py" "${dep}")"
    jobs+=("${job}")
    echo "${arm}=${job}"
  done
else
  [[ "$REUSE_M00" == 1 ]] || { echo "mechanism stage requires --reuse-m00 after valid G2" >&2; exit 2; }
  arms=(m10 m01 m11)
  jobs=()
  for i in "${!arms[@]}"; do
    arm="${arms[$i]}"; dep=""
    if (( i >= MAX_CONCURRENT )); then dep="${jobs[$((i-MAX_CONCURRENT))]}"; fi
    job="$(submit "ctdp-${arm}-s${SEED}" "configs/adatad/thumos/duca_ctdp_mechanism_${arm}.py" "${dep}")"
    jobs+=("${job}")
    echo "${arm}=${job}"
  done
fi
