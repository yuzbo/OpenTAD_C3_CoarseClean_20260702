#!/usr/bin/env bash
# =============================================================================
# DUCA / C3 Remote Experiment Guardian Daemon (1-Minute Periodic Dispatcher)
# =============================================================================
# Purpose: Periodically monitors cluster queue and sequentially dispatches 
# high-value single-seed experiments respecting the cluster QOS submit limit (<= 8 jobs).
# =============================================================================

GUARDIAN_ROOT="/data/run01/sczc063/yuzibo/guardians"
LOG_FILE="/data/run01/sczc063/yuzibo/guardian_logs/experiment_daemon.log"
STATE_DIR="${GUARDIAN_ROOT}/state"
PID_FILE="${GUARDIAN_ROOT}/daemon.pid"

mkdir -p "${STATE_DIR}" "/data/run01/sczc063/yuzibo/guardian_logs"

echo $$ > "${PID_FILE}"

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log "=== Starting DUCA Experiment Guardian Daemon (PID: $$) ==="

PRIMARY_SEED="8261"
N16R4_BASE="/data/run01/sczc063/yuzibo"

while true; do
  TOTAL_ACTIVE=$(squeue -u sczc063 -h 2>/dev/null | wc -l)
  RUNNING_JOBS=$(squeue -u sczc063 -h -t R 2>/dev/null | wc -l)
  PENDING_JOBS=$(squeue -u sczc063 -h -t PD 2>/dev/null | wc -l)

  log "Heartbeat: Active=${TOTAL_ACTIVE} (Running=${RUNNING_JOBS}, Pending=${PENDING_JOBS})"

  # Cluster QOS limit allows max ~8 total active/pending jobs per user.
  # Available slots = 8 - TOTAL_ACTIVE
  MAX_ALLOWED=8
  AVAIL_SLOTS=$(( MAX_ALLOWED - TOTAL_ACTIVE ))

  # -------------------------------------------------------------------------
  # 1. DUCA Evidence Recovery Full Main Method (Arm 7, Seed 8261)
  # -------------------------------------------------------------------------
  if [ ! -f "${STATE_DIR}/duca_ev_rec_arm7_full.done" ] && [ "${AVAIL_SLOTS}" -ge 1 ]; then
    log "[DISPATCH] Submitting DUCA Evidence Recovery Full (Arm 7, Seed ${PRIMARY_SEED})..."
    PROJECT_DIR="/data/run01/sczc063/yuzibo/projects/opentad_duca_evidence_recovery"
    if [ -d "${PROJECT_DIR}" ]; then
      cd "${PROJECT_DIR}"
      RUN_ROOT="${N16R4_BASE}/duca_evidence_recovery_seed${PRIMARY_SEED}_arm7_$(date +%Y%m%d_%H%M%S)"
      mkdir -p "${RUN_ROOT}/slurm_logs"
      
      JOB_OUT=$(sbatch --parsable \
        --array=7-7 \
        --output="${RUN_ROOT}/slurm_logs/%x_%A_%a.out" \
        --error="${RUN_ROOT}/slurm_logs/%x_%A_%a.err" \
        --export=ALL,DUCA_RUN_ROOT="${RUN_ROOT}",DUCA_SEEDS="${PRIMARY_SEED}" \
        scripts/run_duca_evidence_recovery_train_array_n16r4.sbatch 2>&1 || true)
      
      if [[ "${JOB_OUT}" =~ ^[0-9]+ ]]; then
        touch "${STATE_DIR}/duca_ev_rec_arm7_full.done"
        log "[SUCCESS] DUCA Evidence Recovery Full (Arm 7) submitted with Job ID: ${JOB_OUT}"
        AVAIL_SLOTS=$(( AVAIL_SLOTS - 1 ))
      else
        log "[WARN] Submission failed: ${JOB_OUT}"
      fi
    fi
  fi

  # -------------------------------------------------------------------------
  # 2. DUCA Evidence Recovery Baseline (Arm 0, Matched H65-60 Baseline)
  # -------------------------------------------------------------------------
  if [ -f "${STATE_DIR}/duca_ev_rec_arm7_full.done" ] && [ ! -f "${STATE_DIR}/duca_ev_rec_arm0_base.done" ] && [ "${AVAIL_SLOTS}" -ge 1 ]; then
    log "[DISPATCH] Submitting DUCA Evidence Recovery Baseline (Arm 0, Seed ${PRIMARY_SEED})..."
    PROJECT_DIR="/data/run01/sczc063/yuzibo/projects/opentad_duca_evidence_recovery"
    cd "${PROJECT_DIR}"
    RUN_ROOT="${N16R4_BASE}/duca_evidence_recovery_seed${PRIMARY_SEED}_arm0_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${RUN_ROOT}/slurm_logs"
    
    JOB_OUT=$(sbatch --parsable \
      --array=0-0 \
      --output="${RUN_ROOT}/slurm_logs/%x_%A_%a.out" \
      --error="${RUN_ROOT}/slurm_logs/%x_%A_%a.err" \
      --export=ALL,DUCA_RUN_ROOT="${RUN_ROOT}",DUCA_SEEDS="${PRIMARY_SEED}" \
      scripts/run_duca_evidence_recovery_train_array_n16r4.sbatch 2>&1 || true)
    
    if [[ "${JOB_OUT}" =~ ^[0-9]+ ]]; then
      touch "${STATE_DIR}/duca_ev_rec_arm0_base.done"
      log "[SUCCESS] DUCA Evidence Recovery Baseline (Arm 0) submitted with Job ID: ${JOB_OUT}"
      AVAIL_SLOTS=$(( AVAIL_SLOTS - 1 ))
    else
      log "[WARN] Submission failed: ${JOB_OUT}"
    fi
  fi

  # -------------------------------------------------------------------------
  # Sleep 60 seconds before next cycle
  # -------------------------------------------------------------------------
  sleep 60
done
