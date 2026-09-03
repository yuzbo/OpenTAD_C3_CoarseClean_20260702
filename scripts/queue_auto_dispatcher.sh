#!/bin/bash
# ==============================================================================
# Dynamic Queue Daemon: Automatically submits queued single-seed jobs
# as cluster quota slots become available (AssocMaxSubmitJobLimit safe)
# ==============================================================================
set -euo pipefail

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
PROJECT_DIR="${PROJECT_DIR:-${BASE}/projects/duca_unified_single_seed_20260903}"
PARTITION="${PARTITION:-gpu}"
ACCOUNT="${ACCOUNT:-sczc063}"
QOS="${QOS:-normal}"
SEED="${SEED:-3407}"
MAX_JOBS_IN_QUEUE="${MAX_JOBS_IN_QUEUE:-14}"
TIME_LIMIT="${TIME_LIMIT:-18:00:00}"
QUEUE_FILE="${QUEUE_FILE:-}"

REVISION=$(git -C "$PROJECT_DIR" rev-parse HEAD)
if [[ -n "$(git -C "$PROJECT_DIR" status --porcelain)" ]]; then
  echo "Refusing formal dispatch from dirty checkout: $PROJECT_DIR" >&2
  exit 1
fi
STATE_FILE="${STATE_FILE:-${BASE}/guardians/queue_auto_dispatcher_${REVISION:0:12}_${SEED}.state}"
mkdir -p "$(dirname "$STATE_FILE")"

LOG_DIR="${BASE}/slurm_logs/single_seed_$(date +%Y%m%d)"
mkdir -p "$LOG_DIR"

ALL_CONFIGS=(
  # H65-Pro Core & References
  "h65_pro:configs/adatad/thumos/h65_pro/h65_pro_ref_d768.py"
  "h65_pro:configs/adatad/thumos/h65_pro/h65_pro_ref_u384.py"
  "h65_pro:configs/adatad/thumos/h65_pro/h65_pro_ref_mnv3fc384.py"
  # H65-Pro Ablation Factors F01-F16
  "h65_pro:configs/adatad/thumos/h65_pro/h65_pro_f01.py"
  "h65_pro:configs/adatad/thumos/h65_pro/h65_pro_f02.py"
  "h65_pro:configs/adatad/thumos/h65_pro/h65_pro_f03.py"
  "h65_pro:configs/adatad/thumos/h65_pro/h65_pro_f04.py"
  "h65_pro:configs/adatad/thumos/h65_pro/h65_pro_f05.py"
  "h65_pro:configs/adatad/thumos/h65_pro/h65_pro_f06.py"
  "h65_pro:configs/adatad/thumos/h65_pro/h65_pro_f07.py"
  "h65_pro:configs/adatad/thumos/h65_pro/h65_pro_f08.py"
  "h65_pro:configs/adatad/thumos/h65_pro/h65_pro_f09.py"
  "h65_pro:configs/adatad/thumos/h65_pro/h65_pro_f10.py"
  "h65_pro:configs/adatad/thumos/h65_pro/h65_pro_f11.py"
  "h65_pro:configs/adatad/thumos/h65_pro/h65_pro_f12.py"
  "h65_pro:configs/adatad/thumos/h65_pro/h65_pro_f13.py"
  "h65_pro:configs/adatad/thumos/h65_pro/h65_pro_f14.py"
  "h65_pro:configs/adatad/thumos/h65_pro/h65_pro_f15.py"
  "h65_pro:configs/adatad/thumos/h65_pro/h65_pro_f16.py"
  # CT-DP Continuous-Time Dynamics
  "ctdp:configs/adatad/thumos/duca_ctdp_geometry_g0.py"
  "ctdp:configs/adatad/thumos/duca_ctdp_geometry_g1.py"
  # DUCA-Unified Full Matrix
  "duca_unified:configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_u0_seed3407.py"
  "duca_unified:configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_a00_seed3407.py"
  "duca_unified:configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_a01_seed3407.py"
  "duca_unified:configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_a10_seed3407.py"
  "duca_unified:configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_a11_seed3407.py"
  "duca_unified:configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_b00_seed3407.py"
  "duca_unified:configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_b10_seed3407.py"
  "duca_unified:configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_b11_seed3407.py"
  "duca_unified:configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_c01_seed3407.py"
  "duca_unified:configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_c11_seed3407.py"
  "duca_unified:configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_d1_seed3407.py"
  "duca_unified:configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_e01_seed3407.py"
  "duca_unified:configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_e11_seed3407.py"
  "duca_unified:configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_f11_seed3407.py"
  "duca_unified:configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_g10_seed3407.py"
  "duca_unified:configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_g11_seed3407.py"
  "duca_unified:configs/adatad/thumos/duca_unified_fullmatrix/duca_unified_development_h0_seed3407.py"
  # BAFDR Feature Disentanglement & Distillation
  "bafdr:configs/adatad/thumos/bafdr_k16_nokd_seed4407.py"
  "bafdr:configs/adatad/thumos/bafdr_k16_late_seed4407.py"
  "bafdr:configs/adatad/thumos/bafdr_k16_g96_seed4407.py"
  "bafdr:configs/adatad/thumos/bafdr_k16_d160_seed4407.py"
  "bafdr:configs/adatad/thumos/bafdr_k16_u16_uniform_a0_seed4407.py"
  "bafdr:configs/adatad/thumos/bafdr_k16_u128_all48_a0_seed4407.py"
  # ET-TRC & Continuous ROI
  "et_trc:configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_seed4407.py"
  "et_trc:configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_off_seed4407.py"
  "et_trc:configs/adatad/thumos/continuous_roi_s2_v3_d160_seed4407.py"
  "et_trc:configs/adatad/thumos/continuous_roi_s2_v3_g96_seed4407.py"
  "et_trc:configs/adatad/thumos/continuous_roi_s2_v3_u128_a0_seed4407.py"
  # Evidence-Recovery
  "evidence_recovery:configs/adatad/thumos/duca_evidence_recovery_no_coverage.py"
  "evidence_recovery:configs/adatad/thumos/duca_evidence_recovery_no_merge.py"
  "evidence_recovery:configs/adatad/thumos/duca_evidence_recovery_no_recovery.py"
  "evidence_recovery:configs/adatad/thumos/duca_evidence_recovery_no_robust.py"
  "evidence_recovery:configs/adatad/thumos/duca_evidence_recovery_no_time.py"
  "evidence_recovery:configs/adatad/thumos/duca_evidence_recovery_matched_h65_60.py"
)

if [[ -n "$QUEUE_FILE" ]]; then
  if [[ ! -f "$QUEUE_FILE" ]]; then
    echo "Queue manifest not found: $QUEUE_FILE" >&2
    exit 1
  fi
  mapfile -t ALL_CONFIGS < <(grep -Ev '^[[:space:]]*(#|$)' "$QUEUE_FILE")
fi

declare -A SUBMITTED=()
if [[ -f "$STATE_FILE" ]]; then
  while IFS=$'\t' read -r item job_id; do
    if [[ -n "$item" && -n "$job_id" ]]; then
      SUBMITTED["$item"]="$job_id"
    fi
  done < "$STATE_FILE"
fi

echo "Exact commit: $REVISION"
echo "Persistent state: $STATE_FILE"
echo "Total configured jobs to schedule: ${#ALL_CONFIGS[@]}"

INDEX=0
while [[ $INDEX -lt ${#ALL_CONFIGS[@]} ]]; do
  CURRENT_JOBS=$(squeue -u "$USER" -h | wc -l)
  if [[ $CURRENT_JOBS -ge $MAX_JOBS_IN_QUEUE ]]; then
    echo "[$(date +'%H:%M:%S')] Queue full (${CURRENT_JOBS}/${MAX_JOBS_IN_QUEUE}). Waiting 60s..."
    sleep 60
    continue
  fi

  ITEM="${ALL_CONFIGS[$INDEX]}"
  if [[ -n "${SUBMITTED[$ITEM]:-}" ]]; then
    echo "[$(date +'%H:%M:%S')] Already submitted: JobID=${SUBMITTED[$ITEM]} | ${ITEM}"
    INDEX=$((INDEX + 1))
    continue
  fi
  ROUTE="${ITEM%%:*}"
  CFG="${ITEM#*:}"
  if [[ ! -f "${PROJECT_DIR}/${CFG}" ]]; then
    echo "Config not found in exact checkout: ${CFG}" >&2
    exit 1
  fi
  JOB_TAG="$(basename "$CFG" .py)"
  PORT=$((29000 + RANDOM % 10000))
  SBATCH_SCRIPT=$(mktemp /tmp/sbatch_${JOB_TAG}_XXXXXX.sh)

  cat <<SBATCH_EOF > "$SBATCH_SCRIPT"
#!/bin/bash
#SBATCH --job-name=${JOB_TAG}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --qos=${QOS}
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOG_DIR}/${ROUTE}_${JOB_TAG}_%j.out
#SBATCH --error=${LOG_DIR}/${ROUTE}_${JOB_TAG}_%j.err

source /etc/profile
set -euo pipefail
module load cuda/11.8
module load miniforge3/24.11
source ${BASE}/conda_envs/opentad/bin/activate
cd ${PROJECT_DIR}

echo "=== STARTING JOB ${JOB_TAG} (Route: ${ROUTE}, Seed: ${SEED}) ==="
torchrun --nproc_per_node=1 --master_port=${PORT} tools/train.py ${CFG} --seed ${SEED}
echo "=== COMPLETED JOB ${JOB_TAG} ==="
SBATCH_EOF

  chmod +x "$SBATCH_SCRIPT"
  if JOB_ID=$(sbatch --parsable "$SBATCH_SCRIPT"); then
    echo "[$(date +'%H:%M:%S')] Submitted [${INDEX}/${#ALL_CONFIGS[@]}]: JobID=${JOB_ID} | Route=${ROUTE} | Config=${CFG}"
    printf '%s\t%s\n' "$ITEM" "$JOB_ID" >> "$STATE_FILE"
    SUBMITTED["$ITEM"]="$JOB_ID"
    INDEX=$((INDEX + 1))
  else
    echo "[$(date +'%H:%M:%S')] Submission rejected, waiting 60s..."
    sleep 60
  fi
  rm -f "$SBATCH_SCRIPT"
  sleep 2
done

echo "All ${#ALL_CONFIGS[@]} matrix jobs submitted successfully!"
