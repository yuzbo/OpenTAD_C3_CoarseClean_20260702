#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${DUCA_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${DUCA_BUDGET_CURVE_OUTPUT_DIR:?DUCA_BUDGET_CURVE_OUTPUT_DIR is required}"
SOURCE_LIST="${DUCA_BUDGET_AGGREGATES:?DUCA_BUDGET_AGGREGATES is required}"
IFS=: read -r -a sources <<<"${SOURCE_LIST}"
args=()
for source in "${sources[@]}"; do args+=(--aggregate-json "${source}"); done

cd "${REPO_ROOT}"
"${PYTHON}" -m tools.bata.aggregate_duca_budget_curve \
  "${args[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --expected-budgets 384 320 256 192 128
