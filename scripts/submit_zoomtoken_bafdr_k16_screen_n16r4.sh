#!/usr/bin/env bash
source /etc/profile
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
SEED=4407
ARMS="G96,U16-UNIFORM-A0,BAFDR-K16-LATE,BAFDR-K16-NOKD,BAFDR-K16-FULL"
TEACHER_CHECKPOINT="${BAFDR_TEACHER_CHECKPOINT:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2;;
    --arms) ARMS="$2"; shift 2;;
    --teacher-checkpoint) TEACHER_CHECKPOINT="$2"; shift 2;;
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
  if [[ "$arm" == BAFDR-K16-FULL ]]; then
    if [[ -z "$TEACHER_CHECKPOINT" || ! -f "$TEACHER_CHECKPOINT" ]]; then
      echo "BAFDR-K16-FULL blocked: provide an existing terminal D160 teacher checkpoint via --teacher-checkpoint or BAFDR_TEACHER_CHECKPOINT" >&2
      exit 2
    fi
    python - "$TEACHER_CHECKPOINT" <<'PY'
import sys, torch
path = sys.argv[1]
state = torch.load(path, map_location="cpu")
if not isinstance(state, dict) or "state_dict_ema" not in state:
    raise SystemExit("teacher checkpoint must contain state_dict_ema")
epoch = state.get("epoch", state.get("meta", {}).get("epoch", None))
if epoch not in (59, "59"):
    raise SystemExit(f"teacher checkpoint epoch must be 59, got {epoch!r}")
print(f"[PRECHECK] terminal teacher={path} epoch=59 state_dict_ema=present")
PY
  fi
  sbatch --parsable --partition=gpu --gres=gpu:2 --cpus-per-task=8 --time=72:00:00 \
    --job-name="bafdr-${slug}-s${SEED}" \
    --output="${BASE}/slurm_logs/%x_%j.out" --error="${BASE}/slurm_logs/%x_%j.err" \
    --wrap="source /etc/profile; set -euo pipefail; module load cuda/11.8; module load miniforge3/24.11; source ${BASE}/conda_envs/opentad/bin/activate; cd \"${PROJECT_DIR}\"; BAFDR_TEACHER_CHECKPOINT=\"${TEACHER_CHECKPOINT}\" bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh train \"${cfg}\""
done
