#!/usr/bin/env bash
set -euo pipefail

# A lightweight, idempotent coordinator. Training and evaluation always run
# inside Slurm jobs; this process only reconciles missing dependency nodes.
PROJECT_DIR="${PROJECT_DIR:-/data/run01/sczc063/yuzibo/projects/bafdr_k16_fullmatrix_6ae16954}"
BASE="${YUZIBO_ROOT:-/data/run01/sczc063/yuzibo}"
RUN_ROOT="${ZOOMTOKEN_RUN_ROOT:-${BASE}/projects/bafdr_k16_fullmatrix_6ae16954_compute}"
LOG_DIR="${BASE}/slurm_logs"
MANIFEST_DIR="${RUN_ROOT}/manifest"
TAG="${BAFDR_RELEASE_TAG:-6ae16954}"
INTERVAL="${BAFDR_WATCH_INTERVAL:-60}"

mkdir -p "${LOG_DIR}" "${MANIFEST_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/bafdr_fullmatrix_submit_watch.log}"
LOCK_DIR="${MANIFEST_DIR}/.submit-watch.lock"

log() {
  printf '[%s] %s\n' "$(date '+%F %T%z')" "$*" >>"${LOG_FILE}"
}

arm_slug() {
  case "$1" in
    D160) printf 'd160' ;;
    G96) printf 'g96' ;;
    U128-ALL48-A0) printf 'u128_all48_a0' ;;
    U16-UNIFORM-A0) printf 'u16_uniform_a0' ;;
    BAFDR-K16-LATE) printf 'late' ;;
    BAFDR-K16-NOKD) printf 'nokd' ;;
    BAFDR-K16-FULL) printf 'full' ;;
    *) return 2 ;;
  esac
}

safe_arm() { printf '%s' "$1" | tr '_' '-' ; }

job_name() { printf 'zt-bafdr-%s-%s-%s' "$1" "$2" "${TAG}"; }

job_state() {
  local job_id="$1" state
  state="$(squeue -h -j "${job_id}" -o '%T' 2>/dev/null | head -1 || true)"
  if [[ -z "${state}" ]]; then
    state="$(sacct -X -n -P -j "${job_id}" --format=State 2>/dev/null | head -1 | tr -d '[:space:]' || true)"
  fi
  printf '%s' "${state%%+*}"
}

usable_state() {
  case "$1" in
    PENDING|RUNNING|CONFIGURING|COMPLETING|RESIZING|SUSPENDED|COMPLETED) return 0 ;;
    *) return 1 ;;
  esac
}

find_live_job() {
  local name="$1"
  squeue -h -u "${USER}" -o '%A|%j' 2>/dev/null \
    | awk -F'|' -v n="${name}" '$2 == n {print $1; exit}'
}

read_mapping() {
  local path="$1" key="$2"
  [[ -f "${path}" ]] || return 0
  awk -F'\t' -v k="${key}" '$1 == k {print $2; exit}' "${path}"
}

write_train_mapping() {
  local tmp="${MANIFEST_DIR}/.slurm_train_jobs.tsv.tmp.$$"
  : >"${tmp}"
  for arm in D160 G96 U128-ALL48-A0 U16-UNIFORM-A0 BAFDR-K16-LATE BAFDR-K16-NOKD BAFDR-K16-FULL; do
    local id
    id="${TRAIN_JOB_IDS[${arm}]:-}"
    [[ -n "${id}" ]] && printf '%s\t%s\n' "${arm}" "${id}" >>"${tmp}"
  done
  mv -f "${tmp}" "${MANIFEST_DIR}/slurm_train_jobs.tsv"
}

submit_job() {
  local name="$1" dependency="$2" gpus="$3" cpus="$4" time_limit="$5" command="$6"
  local dep_args=() result
  [[ -n "${dependency}" ]] && dep_args=(--dependency="${dependency}")
  result="$(sbatch --parsable --partition=gpu --nodes=1 --ntasks=1 \
    --gpus="${gpus}" --cpus-per-task="${cpus}" --time="${time_limit}" \
    --job-name="${name}" --output="${LOG_DIR}/%x_%j.out" \
    --error="${LOG_DIR}/%x_%j.err" "${dep_args[@]}" \
    --wrap "bash -lc \"cd '${PROJECT_DIR}' && export PROJECT_DIR='${PROJECT_DIR}' ZOOMTOKEN_RUN_ROOT='${RUN_ROOT}' && ${command}\"" 2>&1)" || {
      log "submit failed name=${name}: ${result}"
      return 1
    }
  result="${result%%$'\n'*}"
  [[ "${result}" =~ ^[0-9]+$ ]] || { log "submit returned invalid job id name=${name}: ${result}"; return 1; }
  printf '%s' "${result}"
}

ensure_train_job() {
  local arm="$1" slug name id state marker
  slug="$(arm_slug "${arm}")"
  name="$(job_name train "$(safe_arm "${arm}")")"
  marker="${MANIFEST_DIR}/blocked_${slug}.marker"
  id="${TRAIN_JOB_IDS[${arm}]:-}"
  if [[ -n "${id}" ]]; then
    state="$(job_state "${id}")"
    if usable_state "${state}"; then return 0; fi
    [[ -f "${marker}" ]] || { log "train job failed arm=${arm} id=${id} state=${state}; manual repair required"; : >"${marker}"; }
    return 1
  fi
  id="$(find_live_job "${name}" || true)"
  if [[ -n "${id}" ]]; then
    TRAIN_JOB_IDS[${arm}]="${id}"
    return 0
  fi
  [[ -f "${marker}" ]] && return 1
  if [[ "${arm}" == "BAFDR-K16-FULL" ]]; then
    local d160_id="${TRAIN_JOB_IDS[D160]:-}"
    [[ -n "${d160_id}" ]] || return 1
    local d160_state="$(job_state "${d160_id}")"
    [[ "${d160_state}" != "FAILED" && "${d160_state}" != "CANCELLED" ]] || return 1
    id="$(submit_job "${name}" "afterok:${d160_id}" 2 8 "168:00:00" \
      "bash scripts/run_zoomtoken_bafdr_k16_arm_batch.sh '${arm}' train")" || return 1
  else
    id="$(submit_job "${name}" "" 2 8 "168:00:00" \
      "bash scripts/run_zoomtoken_bafdr_k16_arm_batch.sh '${arm}' train")" || return 1
  fi
  TRAIN_JOB_IDS[${arm}]="${id}"
  log "submitted train arm=${arm} id=${id}"
  return 0
}

ensure_downstream() {
  local train_ids=() arm eval_id cexec_id metrics_id summary_id joined
  for arm in D160 G96 U128-ALL48-A0 U16-UNIFORM-A0 BAFDR-K16-LATE BAFDR-K16-NOKD BAFDR-K16-FULL; do
    [[ -n "${TRAIN_JOB_IDS[${arm}]:-}" ]] || return 0
    train_ids+=("${TRAIN_JOB_IDS[${arm}]}")
  done
  joined="$(IFS=:; printf '%s' "${train_ids[*]}")"

  eval_id="$(read_mapping "${MANIFEST_DIR}/slurm_eval_jobs.tsv" all)"
  if [[ -z "${eval_id}" ]]; then
    eval_id="$(find_live_job "$(job_name eval-all all)" || true)"
  fi
  if [[ -z "${eval_id}" ]]; then
    eval_id="$(submit_job "$(job_name eval-all all)" "afterok:${joined}" 2 8 "168:00:00" \
      "bash scripts/run_zoomtoken_bafdr_k16_arm_batch.sh 'ALL' eval-all")" || return 0
    printf 'all\t%s\n' "${eval_id}" >"${MANIFEST_DIR}/slurm_eval_jobs.tsv"
    log "submitted eval id=${eval_id}"
  fi

  cexec_id="$(read_mapping "${MANIFEST_DIR}/slurm_downstream_jobs.tsv" c_exec)"
  if [[ -z "${cexec_id}" ]]; then
    cexec_id="$(submit_job "$(job_name cexec all)" "afterok:${eval_id}" 1 1 "02:00:00" \
      "bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh c_exec")" || return 0
    printf 'c_exec\t%s\n' "${cexec_id}" >"${MANIFEST_DIR}/slurm_downstream_jobs.tsv"
    log "submitted c_exec id=${cexec_id}"
  fi

  metrics_id="$(read_mapping "${MANIFEST_DIR}/slurm_downstream_jobs.tsv" metrics)"
  if [[ -z "${metrics_id}" ]]; then
    metrics_id="$(submit_job "$(job_name metrics all)" "afterok:${eval_id}:${cexec_id}" 2 8 "168:00:00" \
      "bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh metrics")" || return 0
    printf 'c_exec\t%s\nmetrics\t%s\n' "${cexec_id}" "${metrics_id}" >"${MANIFEST_DIR}/slurm_downstream_jobs.tsv"
    log "submitted metrics id=${metrics_id}"
  fi

  summary_id="$(read_mapping "${MANIFEST_DIR}/slurm_downstream_jobs.tsv" summary)"
  if [[ -z "${summary_id}" ]]; then
    summary_id="$(submit_job "$(job_name summary all)" "afterok:${metrics_id}" 1 1 "02:00:00" \
      "bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh summary-strict")" || return 0
    printf 'c_exec\t%s\nmetrics\t%s\nsummary\t%s\n' "${cexec_id}" "${metrics_id}" "${summary_id}" >"${MANIFEST_DIR}/slurm_downstream_jobs.tsv"
    log "submitted summary id=${summary_id}"
  fi
}

reconcile_once() {
  declare -gA TRAIN_JOB_IDS=()
  local arm id
  for arm in D160 G96 U128-ALL48-A0 U16-UNIFORM-A0 BAFDR-K16-LATE BAFDR-K16-NOKD BAFDR-K16-FULL; do
    id="$(read_mapping "${MANIFEST_DIR}/slurm_train_jobs.tsv" "${arm}")"
    [[ -n "${id}" ]] && TRAIN_JOB_IDS[${arm}]="${id}"
  done
  for arm in D160 G96 U128-ALL48-A0 U16-UNIFORM-A0 BAFDR-K16-LATE BAFDR-K16-NOKD BAFDR-K16-FULL; do
    ensure_train_job "${arm}" || true
  done
  write_train_mapping
  ensure_downstream
}

log "watcher started project=${PROJECT_DIR} run_root=${RUN_ROOT} tag=${TAG} interval=${INTERVAL}s"
while :; do
  if mkdir "${LOCK_DIR}" 2>/dev/null; then
    reconcile_once || log "reconcile failed; will retry next interval"
    rmdir "${LOCK_DIR}" 2>/dev/null || true
  fi
  sleep "${INTERVAL}"
done
