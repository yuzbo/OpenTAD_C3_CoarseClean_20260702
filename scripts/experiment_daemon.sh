#!/usr/bin/env bash
# =============================================================================
# DUCA / C3 Remote Experiment Guardian Daemon (1-Minute Periodic Dispatcher)
# =============================================================================
# The dispatcher submits the full single-seed DAG only when Slurm has room for
# the cost gate and training array, and backs off after a failed submission.
# =============================================================================

source /etc/profile 2>/dev/null || true
set -Eeuo pipefail

GUARDIAN_ROOT="${GUARDIAN_ROOT:-/data/run01/sczc063/yuzibo/guardians}"
LOG_FILE="${LOG_FILE:-/data/run01/sczc063/yuzibo/guardian_logs/experiment_daemon.log}"
STATE_DIR="${GUARDIAN_ROOT}/state"
PID_FILE="${GUARDIAN_ROOT}/daemon.pid"
LOCK_FILE="${GUARDIAN_ROOT}/daemon.lock"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
MAX_ALLOWED="${MAX_ALLOWED:-8}"
PRIMARY_SEED="${PRIMARY_SEED:-8261}"
FULL_MATRIX_SEEDS="${DUCA_SEEDS:-8261 19237 31153}"
ENABLE_FULL_MATRIX="${ENABLE_FULL_MATRIX:-1}"
N16R4_BASE="${N16R4_BASE:-/data/run01/sczc063/yuzibo}"
PROJECT_DIR="${DUCA_PROJECT_DIR:-/data/run01/sczc063/yuzibo/projects/opentad_duca_evidence_recovery}"
YUZIBO_ROOT="${YUZIBO_ROOT:-${N16R4_BASE}}"
DUCA_VIDEOMAE_PRETRAIN="${DUCA_VIDEOMAE_PRETRAIN:-${YUZIBO_ROOT}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth}"
CONTINUATION_SCRIPT="${PROJECT_DIR}/scripts/continue_duca_evidence_recovery_n16r4.sh"

mkdir -p "${STATE_DIR}" "$(dirname "${LOG_FILE}")"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] Another DUCA daemon instance already holds ${LOCK_FILE}." >&2
  exit 0
fi

echo $$ > "${PID_FILE}"

log() {
  printf '[%s] %s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$*" >> "${LOG_FILE}"
}

cleanup() {
  if [[ -f "${PID_FILE}" ]] && [[ "$(<"${PID_FILE}")" == "$$" ]]; then
    rm -f "${PID_FILE}"
  fi
  log "=== Stopping DUCA Experiment Guardian Daemon (PID: $$) ==="
}
trap cleanup EXIT

retry_due() {
  local key="$1"
  local next_file="${STATE_DIR}/${key}.next_retry"
  local now next
  now="$(date +%s)"
  if [[ -f "${next_file}" ]]; then
    next="$(<"${next_file}")"
    if [[ "${next}" =~ ^[0-9]+$ ]] && (( now < next )); then
      return 1
    fi
  fi
  return 0
}

record_retry() {
  local key="$1"
  local attempts_file="${STATE_DIR}/${key}.attempts"
  local next_file="${STATE_DIR}/${key}.next_retry"
  local attempts delay now
  attempts=0
  if [[ -f "${attempts_file}" ]]; then
    attempts="$(<"${attempts_file}")"
  fi
  [[ "${attempts}" =~ ^[0-9]+$ ]] || attempts=0
  attempts=$((attempts + 1))
  case "${attempts}" in
    1) delay=60 ;;
    2) delay=120 ;;
    3) delay=300 ;;
    *) delay=900 ;;
  esac
  now="$(date +%s)"
  printf '%s\n' "${attempts}" > "${attempts_file}"
  printf '%s\n' "$((now + delay))" > "${next_file}"
  log "[RETRY] ${key}: attempt=${attempts}, next retry in ${delay}s."
}

clear_retry() {
  local key="$1"
  rm -f "${STATE_DIR}/${key}.attempts" "${STATE_DIR}/${key}.next_retry"
}

pipeline_job_present() {
  squeue -u sczc063 -h -o '%j' 2>/dev/null | grep -Eq '^(duca_rec_|duca_eval_array)'
}

progress_manifest_present() {
  local manifest state
  while IFS= read -r -d '' manifest; do
    state="$(grep -E '^STATE=' "${manifest}" | head -n1 | cut -d= -f2- | tr -d "'\"")"
    case "${state}" in
      TRAIN_SUBMITTED|EVAL_DEFERRED|EVAL_SUBMITTED|STATS_DEFERRED|SUBMITTED)
        return 0
        ;;
    esac
  done < <(find "${N16R4_BASE}" -maxdepth 2 -type f -name dag_state.env -print0 2>/dev/null)
  return 1
}

single_manifest_blocking() {
  local latest state
  latest="$(find "${N16R4_BASE}" -maxdepth 2 -type f -name dag_state.env -path '*single_seed*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
  [[ -n "${latest}" ]] || return 1
  state="$(grep -E '^STATE=' "${latest}" | head -n1 | cut -d= -f2- | tr -d "'\"")"
  case "${state}" in
    COMPLETE|SUPERSEDED) return 1 ;;
    *) return 0 ;;
  esac
}

full_matrix_manifest_present() {
  local manifest state
  while IFS= read -r -d '' manifest; do
    [[ "${manifest}" == *single_seed* ]] && continue
    state="$(grep -E '^STATE=' "${manifest}" | head -n1 | cut -d= -f2- | tr -d "'\"")"
    [[ "${state}" != COMPLETE ]] && return 0
  done < <(find "${N16R4_BASE}" -maxdepth 2 -type f -name dag_state.env -print0 2>/dev/null)
  return 1
}

latest_manifest_state() {
  local pattern="$1" latest
  latest="$(find "${N16R4_BASE}" -maxdepth 2 -type f -name dag_state.env -path "${pattern}" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
  [[ -n "${latest}" ]] || return 1
  grep -E '^STATE=' "${latest}" | head -n1 | cut -d= -f2- | tr -d "'\""
}

reset_completed_marker_after_runtime_failure() {
  local key="$1" marker="$2" pattern="$3" state
  [[ -f "${marker}" ]] || return 0
  state="$(latest_manifest_state "${pattern}" || true)"
  case "${state}" in
    COST_FAILED|COST_UNKNOWN|TRAIN_FAILED|TRAIN_UNKNOWN|EVAL_FAILED|EVAL_UNKNOWN|EVAL_SUBMIT_FAILED|STATS_FAILED|STATS_UNKNOWN|STATS_SUBMIT_FAILED)
      rm -f "${marker}"
      record_retry "${key}"
      log "[RETRY] Latest DAG state=${state}; cleared ${marker} for bounded redeployment."
      ;;
  esac
}

continue_manifests() {
  local manifest state output
  if [[ ! -x "${CONTINUATION_SCRIPT}" ]]; then
    log "[WARN] Continuation script not executable or missing: ${CONTINUATION_SCRIPT}"
    return 0
  fi
  while IFS= read -r -d '' manifest; do
    state="$(grep -E '^STATE=' "${manifest}" | head -n1 | cut -d= -f2- | tr -d "'\"")"
    case "${state}" in
      TRAIN_SUBMITTED|EVAL_DEFERRED|EVAL_SUBMITTED|STATS_DEFERRED|SUBMITTED)
        if output="$(MAX_ALLOWED="${MAX_ALLOWED}" bash "${CONTINUATION_SCRIPT}" "${manifest}" 2>&1)"; then
          [[ -n "${output}" ]] && log "[CONTINUE] ${manifest}: ${output}"
        else
          log "[CONTINUE-ERROR] ${manifest}: ${output}"
        fi
        ;;
    esac
  done < <(find "${N16R4_BASE}" -maxdepth 2 -type f -name dag_state.env -print0 2>/dev/null)
}

submit_full_pipeline() {
  local output
  if [[ ! -f "${DUCA_VIDEOMAE_PRETRAIN}" ]]; then
    log "[ERROR] VideoMAE checkpoint not found: ${DUCA_VIDEOMAE_PRETRAIN}"
    return 1
  fi
  if ! output="$(
    cd "${PROJECT_DIR}"
    DUCA_N16R4_BASE="${N16R4_BASE}" \
    DUCA_REPO_ROOT="${PROJECT_DIR}" \
    YUZIBO_ROOT="${YUZIBO_ROOT}" \
    DUCA_VIDEOMAE_PRETRAIN="${DUCA_VIDEOMAE_PRETRAIN}" \
    bash scripts/submit_duca_evidence_recovery_single_seed_n16r4.sh "${PRIMARY_SEED}" 2>&1
  )"; then
    log "[WARN] DUCA single-seed DAG submission failed: ${output}"
    return 1
  fi
  log "[SUCCESS] DUCA single-seed DAG submitted: ${output}"
}

submit_full_matrix() {
  local output
  if [[ ! -f "${DUCA_VIDEOMAE_PRETRAIN}" ]]; then
    log "[ERROR] VideoMAE checkpoint not found: ${DUCA_VIDEOMAE_PRETRAIN}"
    return 1
  fi
  if ! output="$(
    cd "${PROJECT_DIR}"
    DUCA_N16R4_BASE="${N16R4_BASE}" \
    DUCA_REPO_ROOT="${PROJECT_DIR}" \
    YUZIBO_ROOT="${YUZIBO_ROOT}" \
    DUCA_SEEDS="${FULL_MATRIX_SEEDS}" \
    DUCA_VIDEOMAE_PRETRAIN="${DUCA_VIDEOMAE_PRETRAIN}" \
    bash scripts/submit_duca_evidence_recovery_full_matrix_n16r4.sh 2>&1
  )"; then
    log "[WARN] DUCA full-matrix DAG submission failed: ${output}"
    return 1
  fi
  log "[SUCCESS] DUCA full-matrix DAG submitted: ${output}"
}

log "=== Starting DUCA Experiment Guardian Daemon (PID: $$) ==="

while true; do
  TOTAL_ACTIVE="$(squeue -u sczc063 -h 2>/dev/null | wc -l)"
  RUNNING_JOBS="$(squeue -u sczc063 -h -t R 2>/dev/null | wc -l)"
  PENDING_JOBS="$(squeue -u sczc063 -h -t PD 2>/dev/null | wc -l)"
  AVAIL_SLOTS=$(( MAX_ALLOWED - TOTAL_ACTIVE ))

  log "Heartbeat: Active=${TOTAL_ACTIVE} (Running=${RUNNING_JOBS}, Pending=${PENDING_JOBS}, Available=${AVAIL_SLOTS})"

  continue_manifests

  # A marker means submission succeeded, not that every downstream phase
  # succeeded.  Clear it after a runtime failure so the next heartbeat can
  # redeploy once the bounded backoff and queue limits allow it.
  reset_completed_marker_after_runtime_failure \
    "duca_evidence_recovery_seed${PRIMARY_SEED}" \
    "${STATE_DIR}/duca_evidence_recovery_seed${PRIMARY_SEED}.done" \
    "*single_seed*"
  reset_completed_marker_after_runtime_failure \
    "duca_evidence_recovery_full_matrix" \
    "${STATE_DIR}/duca_evidence_recovery_full_matrix.done" \
    "*full_matrix*"

  if [[ ! -f "${STATE_DIR}/duca_evidence_recovery_seed${PRIMARY_SEED}.done" ]]; then
    if [[ ! -d "${PROJECT_DIR}" ]]; then
      log "[ERROR] Evidence recovery project dir not found: ${PROJECT_DIR}"
    elif pipeline_job_present; then
      log "[WAIT] Existing DUCA pipeline job detected; skipping duplicate dispatch."
    elif progress_manifest_present; then
      log "[WAIT] A DUCA DAG manifest is awaiting continuation; skipping duplicate dispatch."
    elif (( AVAIL_SLOTS < 2 )); then
      log "[WAIT] Need two submission slots for cost + training array; deferring."
    elif ! retry_due "duca_evidence_recovery_seed${PRIMARY_SEED}"; then
      log "[WAIT] DUCA single-seed retry backoff is active."
    elif submit_full_pipeline; then
      touch "${STATE_DIR}/duca_evidence_recovery_seed${PRIMARY_SEED}.done"
      clear_retry "duca_evidence_recovery_seed${PRIMARY_SEED}"
    else
      record_retry "duca_evidence_recovery_seed${PRIMARY_SEED}"
    fi
  fi

  if [[ "${ENABLE_FULL_MATRIX}" == "1" && ! -f "${STATE_DIR}/duca_evidence_recovery_full_matrix.done" ]]; then
    TOTAL_ACTIVE_NOW="$(squeue -u sczc063 -h 2>/dev/null | wc -l)"
    AVAIL_SLOTS_NOW=$(( MAX_ALLOWED - TOTAL_ACTIVE_NOW ))
    if [[ ! -d "${PROJECT_DIR}" ]]; then
      log "[ERROR] Evidence recovery project dir not found: ${PROJECT_DIR}"
    elif pipeline_job_present; then
      log "[WAIT] Existing DUCA pipeline job detected; full-matrix dispatch skipped."
    elif single_manifest_blocking; then
      log "[WAIT] Latest single-seed DAG is not complete; full-matrix dispatch skipped."
    elif progress_manifest_present || full_matrix_manifest_present; then
      log "[WAIT] A DUCA DAG manifest is awaiting completion; full-matrix dispatch skipped."
    elif (( AVAIL_SLOTS_NOW < 2 )); then
      log "[WAIT] Need two submission slots for full-matrix cost + training; deferring."
    elif ! retry_due "duca_evidence_recovery_full_matrix"; then
      log "[WAIT] DUCA full-matrix retry backoff is active."
    elif submit_full_matrix; then
      touch "${STATE_DIR}/duca_evidence_recovery_full_matrix.done"
      clear_retry "duca_evidence_recovery_full_matrix"
    else
      record_retry "duca_evidence_recovery_full_matrix"
    fi
  fi

  sleep "${CHECK_INTERVAL_SECONDS}"
done
