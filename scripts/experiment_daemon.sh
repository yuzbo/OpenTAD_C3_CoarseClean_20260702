#!/usr/bin/env bash
# =============================================================================
# DUCA / C3 Remote Experiment Guardian Daemon (1-Minute Periodic Dispatcher)
# =============================================================================
# Purpose: Periodically monitors cluster queue and sequentially dispatches 
# high-value single-seed experiments when resources are available.
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

while true; do
  TOTAL_ACTIVE=$(squeue -u sczc063 -h 2>/dev/null | wc -l)
  RUNNING_JOBS=$(squeue -u sczc063 -h -t R 2>/dev/null | wc -l)
  PENDING_JOBS=$(squeue -u sczc063 -h -t PD 2>/dev/null | wc -l)

  log "Heartbeat: Active=${TOTAL_ACTIVE} (Running=${RUNNING_JOBS}, Pending=${PENDING_JOBS})"

  # -------------------------------------------------------------------------
  # 1. DUCA Evidence Recovery 8-Arm Single-Seed (Primary Seed: 8261 / 3407)
  # -------------------------------------------------------------------------
  if [ ! -f "${STATE_DIR}/duca_evidence_recovery_seed8261.done" ]; then
    log "[DISPATCH] Candidate 1: Submitting DUCA Evidence Recovery 8-Arm (Seed 8261)..."
    PROJECT_DIR="/data/run01/sczc063/yuzibo/projects/opentad_duca_evidence_recovery"
    if [ -d "${PROJECT_DIR}" ]; then
      cd "${PROJECT_DIR}"
      if bash scripts/submit_duca_evidence_recovery_single_seed_n16r4.sh 8261 >> "${LOG_FILE}" 2>&1; then
        touch "${STATE_DIR}/duca_evidence_recovery_seed8261.done"
        log "[SUCCESS] Candidate 1: DUCA Evidence Recovery (Seed 8261) submitted successfully."
      else
        log "[WARN] Candidate 1 submission encountered an issue, will retry in next cycle."
      fi
    else
      log "[ERROR] Evidence recovery project dir not found: ${PROJECT_DIR}"
    fi
  fi

  # -------------------------------------------------------------------------
  # 2. CT-DP-BAMoD 4-Arm Matrix (Seed 3408 Multi-Seed Supplement)
  # -------------------------------------------------------------------------
  if [ -f "${STATE_DIR}/duca_evidence_recovery_seed8261.done" ] && [ ! -f "${STATE_DIR}/ct_dp_bamod_seed3408.done" ]; then
    if [ "${PENDING_JOBS}" -lt 6 ]; then
      log "[DISPATCH] Candidate 2: Submitting CT-DP-BAMoD 4-Arm (Seed 3408)..."
      BAMOD_DIR="/data/run01/sczc063/yuzibo/projects/opentad_duca_ct_dp_bamod_d9bdb3f_20260901"
      if [ -d "${BAMOD_DIR}" ]; then
        cd "${BAMOD_DIR}"
        # Submit 4-arm jobs for seed 3408
        sbatch --export=ALL,SEED=3408 scripts/slurm_duca_ct_dp_bamod_seed3407.sbatch >> "${LOG_FILE}" 2>&1 || true
        touch "${STATE_DIR}/ct_dp_bamod_seed3408.done"
        log "[SUCCESS] Candidate 2: CT-DP-BAMoD (Seed 3408) submitted."
      fi
    else
      log "[WAIT] Queue has ${PENDING_JOBS} pending jobs, deferring Candidate 2 to next cycle."
    fi
  fi

  # -------------------------------------------------------------------------
  # Sleep 60 seconds before next check
  # -------------------------------------------------------------------------
  sleep 60
done
