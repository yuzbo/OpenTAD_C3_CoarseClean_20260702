#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[DUCA_OFFICIAL_BUDGET_CURVE][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
FULLTRAIN_CANDIDATE="${FULLTRAIN_CANDIDATE:-0}"
DUCA_ONLINE_BUDGET_START="${DUCA_ONLINE_BUDGET_START:-128}"
DUCA_ONLINE_BUDGET_END="${DUCA_ONLINE_BUDGET_END:-768}"
DUCA_ONLINE_BUDGET_STEP="${DUCA_ONLINE_BUDGET_STEP:-32}"
DUCA_ONLINE_DENSE_WINDOW_SIZE="${DUCA_ONLINE_DENSE_WINDOW_SIZE:-768}"
DUCA_VALIDATOR_MAX_BUDGET="${DUCA_VALIDATOR_MAX_BUDGET:-${DUCA_ONLINE_BUDGET_END}}"
RUN_TAG_BASE="${RUN_TAG_BASE:-duca_online_official_budget_curve_$(date +%Y%m%d_%H%M%S_%z)}"
RUN_DIR_BASE="${RUN_DIR_BASE:-logs/${RUN_TAG_BASE}}"
WORK_DIR_BASE="${WORK_DIR_BASE:-exps/thumos/adatad/duca_online_official_budget_curve/${RUN_TAG_BASE}}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-30320}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"

if [[ -n "${DUCA_ONLINE_BUDGETS:-}" ]]; then
  read -r -a budget_values <<< "${DUCA_ONLINE_BUDGETS}"
else
  budget_values=()
  budget="${DUCA_ONLINE_BUDGET_START}"
  while [[ "${budget}" -le "${DUCA_ONLINE_BUDGET_END}" ]]; do
    budget_values+=("${budget}")
    budget=$((budget + DUCA_ONLINE_BUDGET_STEP))
  done
fi

[[ "${#budget_values[@]}" -gt 0 ]] || fail "empty budget curve"

echo "[DUCA_OFFICIAL_BUDGET_CURVE] repo=${REPO_ROOT}"
echo "[DUCA_OFFICIAL_BUDGET_CURVE] head=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
echo "[DUCA_OFFICIAL_BUDGET_CURVE] budgets=${budget_values[*]}"
echo "[DUCA_OFFICIAL_BUDGET_CURVE] dense_window=${DUCA_ONLINE_DENSE_WINDOW_SIZE} validator_max_budget=${DUCA_VALIDATOR_MAX_BUDGET}"
echo "[DUCA_OFFICIAL_BUDGET_CURVE] precheck_only=${PRECHECK_ONLY} fulltrain_candidate=${FULLTRAIN_CANDIDATE}"

mkdir -p "${RUN_DIR_BASE}" "${WORK_DIR_BASE}"
manifest="${RUN_DIR_BASE}/budget_curve_manifest.tsv"
printf "budget\tstatus\trun_tag\n" > "${manifest}"

idx=0
for budget in "${budget_values[@]}"; do
  if [[ "${budget}" -le 0 ]]; then
    fail "budget must be positive: ${budget}"
  fi
  if (( budget > DUCA_ONLINE_DENSE_WINDOW_SIZE )); then
    fail "budget ${budget} exceeds dense window ${DUCA_ONLINE_DENSE_WINDOW_SIZE}"
  fi
  if (( budget % 16 != 0 )); then
    fail "budget ${budget} must be divisible by 16"
  fi

  run_tag="${RUN_TAG_BASE}_b${budget}"
  master_port=$((MASTER_PORT_BASE + idx))
  echo "[DUCA_OFFICIAL_BUDGET_CURVE] START budget=${budget} run_tag=${run_tag}"

  set +e
  DUCA_ONLINE_BUDGET="${budget}" \
  DUCA_ONLINE_DENSE_WINDOW_SIZE="${DUCA_ONLINE_DENSE_WINDOW_SIZE}" \
  DUCA_VALIDATOR_MAX_BUDGET="${DUCA_VALIDATOR_MAX_BUDGET}" \
  DUCA_BUDGET_CURVE_MODE=1 \
  PRECHECK_ONLY="${PRECHECK_ONLY}" \
  FULLTRAIN_CANDIDATE="${FULLTRAIN_CANDIDATE}" \
  RUN_TAG="${run_tag}" \
  RUN_DIR="${RUN_DIR_BASE}/budget_${budget}/logs" \
  WORK_DIR="${WORK_DIR_BASE}/budget_${budget}/work_dir" \
  RUN_ID="${idx}" \
  MASTER_PORT="${master_port}" \
  bash scripts/run_duca_online_official_adatad_backend_gpu1.sh
  rc=$?
  set -e

  if [[ "${rc}" -ne 0 ]]; then
    printf "%s\tfailed:%s\t%s\n" "${budget}" "${rc}" "${run_tag}" >> "${manifest}"
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      fail "budget ${budget} failed with rc=${rc}"
    fi
  else
    printf "%s\tcomplete\t%s\n" "${budget}" "${run_tag}" >> "${manifest}"
  fi
  idx=$((idx + 1))
done

echo "[DUCA_OFFICIAL_BUDGET_CURVE] COMPLETE manifest=${manifest}"
