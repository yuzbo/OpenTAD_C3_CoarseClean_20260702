#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${DUCA_REPO_ROOT:-$(pwd -P)}"
OUTPUT_DIR="${R5_OUTPUT_DIR:?set R5_OUTPUT_DIR}"
LEARNED_CONFIG="${R5_LEARNED_CONFIG:?set R5_LEARNED_CONFIG to the reviewed G1/G2 config}"
UNIFORM_CONFIG="${R5_UNIFORM_CONFIG:-${REPO_ROOT}/configs/adatad/thumos/duca_two_stage_exact_uniform_fixed384_official60.py}"
TARGET_CLUSTER="${TARGET_CLUSTER:-n16r4}"

cd "${REPO_ROOT}"
python -m tools.bata.duca_r5_paper_matrix \
  --repo-root "${REPO_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --uniform-config "${UNIFORM_CONFIG}" \
  --learned-config "${LEARNED_CONFIG}" \
  --cluster "${TARGET_CLUSTER}"

echo "Generated only; no jobs were submitted."
echo "Mechanism gate: sbatch ${OUTPUT_DIR}/jobs/temporalmaxer_one_step.sbatch"
echo "Matrix index:  ${OUTPUT_DIR}/cells.tsv"
