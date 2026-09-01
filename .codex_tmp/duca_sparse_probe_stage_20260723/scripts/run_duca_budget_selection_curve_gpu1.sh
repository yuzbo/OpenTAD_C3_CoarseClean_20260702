#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${DUCA_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${DUCA_BUDGET_SELECTION_OUTPUT_DIR:?DUCA_BUDGET_SELECTION_OUTPUT_DIR is required}"
SOURCE_LIST="${DUCA_BUDGET_MATRIX_SUMMARIES:?DUCA_BUDGET_MATRIX_SUMMARIES is required}"
IFS=: read -r -a sources <<<"${SOURCE_LIST}"
args=()
for source in "${sources[@]}"; do args+=(--matrix-summary "${source}"); done

cd "${REPO_ROOT}"
"${PYTHON}" -m tools.bata.analyze_duca_budget_selection_curve \
  "${args[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --expected-budgets 384 320 256 192 128 \
  --backend actionformer --seed 3407 --device cuda:0
