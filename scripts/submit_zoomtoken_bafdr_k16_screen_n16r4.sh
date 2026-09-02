#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
SEED=4407
ARMS="U16-UNIFORM-A0,BAFDR-K16-LATE,BAFDR-K16-NOKD,BAFDR-K16-FULL"
REUSE_G96=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2;;
    --arms) ARMS="$2"; shift 2;;
    --reuse-g96) REUSE_G96=1; shift;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
cd "$PROJECT_DIR"
IFS=',' read -r -a arm_list <<< "$ARMS"
for arm in "${arm_list[@]}"; do
  case "$arm" in
    U16-UNIFORM-A0) slug=u16_uniform_a0 ;;
    BAFDR-K16-LATE) slug=late ;;
    BAFDR-K16-NOKD) slug=nokd ;;
    BAFDR-K16-FULL) slug=full ;;
    G96) slug=g96 ;;
    *) echo "unsupported screen arm: $arm" >&2; exit 2 ;;
  esac
  cfg="configs/adatad/thumos/bafdr_k16_${slug}_seed${SEED}.py"
  [[ -f "$cfg" ]] || { echo "missing config: $cfg" >&2; exit 1; }
  if [[ "$arm" == BAFDR-K16-FULL && "$REUSE_G96" != 1 ]]; then
    echo "BAFDR-K16-FULL requires --reuse-g96 or an explicit matched G96 job" >&2
    exit 2
  fi
  sbatch --parsable --partition=gpu --gres=gpu:2 --cpus-per-task=8 --time=72:00:00 \
    --job-name="bafdr-${slug}-s${SEED}" \
    --output="${BASE}/slurm_logs/%x_%j.out" --error="${BASE}/slurm_logs/%x_%j.err" \
    --wrap="bash -lc 'set -euo pipefail; source /etc/profile; module load cuda/11.8; module load miniforge3/24.11; cd \"${PROJECT_DIR}\"; bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh train \"${cfg}\"'"
done
