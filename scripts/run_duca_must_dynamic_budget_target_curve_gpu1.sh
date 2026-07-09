#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_MUST_TARGET_CURVE][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
FULLTRAIN_CANDIDATE="${FULLTRAIN_CANDIDATE:-0}"
DUCA_MUST_DENSE_WINDOW_SIZE="${DUCA_MUST_DENSE_WINDOW_SIZE:-768}"
DUCA_MUST_BUDGET_MIN="${DUCA_MUST_BUDGET_MIN:-64}"
DUCA_MUST_BUDGET_MAX="${DUCA_MUST_BUDGET_MAX:-384}"
DUCA_MUST_BUDGET_MULTIPLE="${DUCA_MUST_BUDGET_MULTIPLE:-16}"
DUCA_MUST_TARGETS="${DUCA_MUST_TARGETS:-128 192 256 320}"
RUN_TAG_BASE="${RUN_TAG_BASE:-duca_must_dynamic_target_curve_$(date +%Y%m%d_%H%M%S_%z)}"
RUN_DIR_BASE="${RUN_DIR_BASE:-logs/${RUN_TAG_BASE}}"
WORK_DIR_BASE="${WORK_DIR_BASE:-exps/thumos/adatad/duca_must_dynamic_target_curve/${RUN_TAG_BASE}}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-30380}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"

read -r -a target_values <<< "${DUCA_MUST_TARGETS}"
[[ "${#target_values[@]}" -gt 0 ]] || fail "empty DUCA_MUST_TARGETS"

echo "[DUCA_MUST_TARGET_CURVE] repo=${REPO_ROOT}"
echo "[DUCA_MUST_TARGET_CURVE] head=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
echo "[DUCA_MUST_TARGET_CURVE] targets=${target_values[*]}"
echo "[DUCA_MUST_TARGET_CURVE] dense=${DUCA_MUST_DENSE_WINDOW_SIZE} min=${DUCA_MUST_BUDGET_MIN} max=${DUCA_MUST_BUDGET_MAX} multiple=${DUCA_MUST_BUDGET_MULTIPLE}"
echo "[DUCA_MUST_TARGET_CURVE] precheck_only=${PRECHECK_ONLY} fulltrain_candidate=${FULLTRAIN_CANDIDATE}"

mkdir -p "${RUN_DIR_BASE}" "${WORK_DIR_BASE}"
manifest="${RUN_DIR_BASE}/dynamic_target_curve_manifest.tsv"
printf "target_budget\tbudget_min\tbudget_max\tstatus\trun_tag\n" > "${manifest}"

idx=0
for target in "${target_values[@]}"; do
  if [[ "${target}" -le 0 ]]; then
    fail "target budget must be positive: ${target}"
  fi
  if (( target < DUCA_MUST_BUDGET_MIN || target > DUCA_MUST_BUDGET_MAX )); then
    fail "target ${target} must lie inside [${DUCA_MUST_BUDGET_MIN}, ${DUCA_MUST_BUDGET_MAX}]"
  fi
  if (( target % DUCA_MUST_BUDGET_MULTIPLE != 0 )); then
    fail "target ${target} must be divisible by budget multiple ${DUCA_MUST_BUDGET_MULTIPLE}"
  fi

  run_tag="${RUN_TAG_BASE}_target${target}"
  master_port=$((MASTER_PORT_BASE + idx))
  echo "[DUCA_MUST_TARGET_CURVE] START target=${target} run_tag=${run_tag}"

  set +e
  DUCA_MUST_DENSE_WINDOW_SIZE="${DUCA_MUST_DENSE_WINDOW_SIZE}" \
  DUCA_MUST_BUDGET_MIN="${DUCA_MUST_BUDGET_MIN}" \
  DUCA_MUST_BUDGET_MAX="${DUCA_MUST_BUDGET_MAX}" \
  DUCA_MUST_BUDGET_MULTIPLE="${DUCA_MUST_BUDGET_MULTIPLE}" \
  DUCA_MUST_BUDGET_TARGET="${target}" \
  DUCA_MUST_TARGET_CURVE_MODE=1 \
  PRECHECK_ONLY="${PRECHECK_ONLY}" \
  FULLTRAIN_CANDIDATE="${FULLTRAIN_CANDIDATE}" \
  RUN_TAG="${run_tag}" \
  RUN_DIR="${RUN_DIR_BASE}/target_${target}/logs" \
  WORK_DIR="${WORK_DIR_BASE}/target_${target}/work_dir" \
  RUN_ID="${idx}" \
  MASTER_PORT="${master_port}" \
  bash scripts/run_duca_must_dynamic_official_adatad_backend_gpu1.sh
  rc=$?
  set -e

  if [[ "${rc}" -ne 0 ]]; then
    printf "%s\t%s\t%s\tfailed:%s\t%s\n" "${target}" "${DUCA_MUST_BUDGET_MIN}" "${DUCA_MUST_BUDGET_MAX}" "${rc}" "${run_tag}" >> "${manifest}"
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      fail "target ${target} failed with rc=${rc}"
    fi
  else
    printf "%s\t%s\t%s\tcomplete\t%s\n" "${target}" "${DUCA_MUST_BUDGET_MIN}" "${DUCA_MUST_BUDGET_MAX}" "${run_tag}" >> "${manifest}"
  fi
  idx=$((idx + 1))
done

echo "[DUCA_MUST_TARGET_CURVE] COMPLETE manifest=${manifest}"
