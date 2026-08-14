#!/usr/bin/env bash
set -eu

# PRE_RUN-only contract. This file intentionally performs no submission, data
# discovery, training, inference, evaluation, or metric/cost execution.
: "${DUCA_EVALUATOR_PRE_RUN_ADMISSION_ARTIFACT:?Refusing execution: supply explicit future Evaluator PRE_RUN admission artifact}"
test -f "$DUCA_EVALUATOR_PRE_RUN_ADMISSION_ARTIFACT" || {
  echo "missing Evaluator PRE_RUN admission artifact: $DUCA_EVALUATOR_PRE_RUN_ADMISSION_ARTIFACT" >&2
  exit 2
}
echo "PRE_RUN_ADMITTED artifact=$DUCA_EVALUATOR_PRE_RUN_ADMISSION_ARTIFACT"
echo "FUTURE_COMMAND_CONTRACT: python tools/train.py configs/adatad/thumos/duca_dynamic_b_n16r4_protocol.py --arms dense uniform_k384 dynamic_A dynamic_B k_shuffle no_risk --windows 16 --repeats 4"
