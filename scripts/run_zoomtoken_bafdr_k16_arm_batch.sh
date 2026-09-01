#!/usr/bin/env bash
set -euo pipefail

ARM="${1:?arm is required}"
PHASE="${2:?phase is required (train or eval-all)}"
case "${ARM}" in
  D160) SLUG="d160" ;;
  G96) SLUG="g96" ;;
  U128-ALL48-A0) SLUG="u128_all48_a0" ;;
  U16-UNIFORM-A0) SLUG="u16_uniform_a0" ;;
  BAFDR-K16-LATE) SLUG="late" ;;
  BAFDR-K16-NOKD) SLUG="nokd" ;;
  BAFDR-K16-FULL) SLUG="full" ;;
  ALL) SLUG="" ;;
  *) echo "unknown arm: ${ARM}" >&2; exit 2 ;;
esac
if [[ "${PHASE}" != "train" && "${PHASE}" != "eval-all" ]]; then
  echo "unsupported phase: ${PHASE}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

SEEDS=(4407 4408 4409)
ARMS=(D160 G96 U128-ALL48-A0 U16-UNIFORM-A0 BAFDR-K16-LATE BAFDR-K16-NOKD BAFDR-K16-FULL)

run_cell() {
  local cell_arm="$1"
  local cell_slug="$2"
  local seed="$3"
  local config="configs/adatad/thumos/bafdr_k16_${cell_slug}_seed${seed}.py"
  echo "[BA-FDR] ${PHASE} ${cell_arm} seed ${seed}"
  if [[ "${PHASE}" == "train" ]]; then
    bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh train "${config}"
  else
    bash scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh eval "${config}"
  fi
}

if [[ "${PHASE}" == "train" ]]; then
  for seed in "${SEEDS[@]}"; do
    run_cell "${ARM}" "${SLUG}" "${seed}"
  done
  exit 0
fi

for cell_arm in "${ARMS[@]}"; do
  case "${cell_arm}" in
    D160) cell_slug="d160" ;;
    G96) cell_slug="g96" ;;
    U128-ALL48-A0) cell_slug="u128_all48_a0" ;;
    U16-UNIFORM-A0) cell_slug="u16_uniform_a0" ;;
    BAFDR-K16-LATE) cell_slug="late" ;;
    BAFDR-K16-NOKD) cell_slug="nokd" ;;
    BAFDR-K16-FULL) cell_slug="full" ;;
  esac
  for seed in "${SEEDS[@]}"; do
    run_cell "${cell_arm}" "${cell_slug}" "${seed}"
  done
done
