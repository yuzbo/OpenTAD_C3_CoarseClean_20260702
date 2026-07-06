#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:-c3_stage4_detector_aware_truetime_curriculum_precheck_$(date +%Y%m%d_%H%M%S_%z)}"
OUTPUT_ROOT="${C3_STAGE4_OUTPUT_ROOT:-${REPO_ROOT}/outputs/${RUN_TAG}}"
STAGE4_EVIDENCE_JSON="${STAGE4_EVIDENCE_JSON:-${OUTPUT_ROOT}/stage4_curriculum_evidence.json}"

mkdir -p "${OUTPUT_ROOT}"

: "${STAGE2_TEACHER_SUMMARY_JSON:?STAGE2_TEACHER_SUMMARY_JSON is required}"
: "${STAGE2_TEACHER_OUTPUT_JSONL:?STAGE2_TEACHER_OUTPUT_JSONL is required}"
: "${STAGE2_POLICY_SUMMARY_JSON:?STAGE2_POLICY_SUMMARY_JSON is required}"
: "${STAGE3_PROOF_JSON:?STAGE3_PROOF_JSON is required}"
: "${STAGE2_LEDGER_SUMMARY_JSONS:?STAGE2_LEDGER_SUMMARY_JSONS is required; pass a space-separated list covering fixed_384 fixed_768 dynamic, preferably train/val/test}"

python -m py_compile \
  tools/bata/validate_stage4_detector_aware_truetime_curriculum.py

python -m tools.bata.validate_stage4_detector_aware_truetime_curriculum \
  --stage2-teacher-summary-json "${STAGE2_TEACHER_SUMMARY_JSON}" \
  --stage2-teacher-output-jsonl "${STAGE2_TEACHER_OUTPUT_JSONL}" \
  --stage2-policy-summary-json "${STAGE2_POLICY_SUMMARY_JSON}" \
  --stage2-ledger-summary-json ${STAGE2_LEDGER_SUMMARY_JSONS} \
  --stage3-proof-json "${STAGE3_PROOF_JSON}" \
  --write-evidence-json "${STAGE4_EVIDENCE_JSON}"

echo "STAGE4_EVIDENCE_JSON=${STAGE4_EVIDENCE_JSON}"
