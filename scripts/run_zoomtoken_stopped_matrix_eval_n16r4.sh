#!/bin/bash
set -euo pipefail

: "${SLURM_JOB_ID:?Use a Slurm GPU allocation}"
: "${ZOOMTOKEN_SOURCE_ROOT:?Set the clean evaluation source}"
: "${ZOOMTOKEN_EXPECTED_COMMIT:?Set the evaluation commit}"
: "${ZOOMTOKEN_TRAINING_ROOT:?Set the cancelled training run root}"
: "${ZOOMTOKEN_TRAINING_SOURCE:?Set the immutable training source}"
: "${ZOOMTOKEN_MATRIX_KIND:?Set d2s or patad}"
: "${ZOOMTOKEN_EVAL_ROOT:?Use a new diagnostic output root}"

cd "${ZOOMTOKEN_SOURCE_ROOT}"
export PYTHONPATH="${ZOOMTOKEN_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4

MODE=evaluate
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  MODE=precheck
fi
python -u tools/bata/zoomtoken_stopped_matrix_eval.py "${MODE}" \
  --training-root "${ZOOMTOKEN_TRAINING_ROOT}" \
  --training-source "${ZOOMTOKEN_TRAINING_SOURCE}" \
  --output-root "${ZOOMTOKEN_EVAL_ROOT}" \
  --expected-commit "${ZOOMTOKEN_EXPECTED_COMMIT}"
