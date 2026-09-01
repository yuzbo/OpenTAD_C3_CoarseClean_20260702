---
doc_id: BUILDER_DUCA_P0_MINIMAL_CHANGE_PLAN
version: v002
stage: DRAFT
author_role: builder
parent_decision: PRO_P0_BLOCKER_DECISION-v001
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
evidence_class: BLOCKED_PRE_RESULT
---

# Builder P0 minimal-change-plan request

Produce a plan only; do not modify production or test files in this step.

The accepted Pro decision fixes two semantic defects: (1) both uniform paths
must call the one integer half-up canonical generator, with K_eff and T_v as
specified in PRO_P0_BLOCKER_DECISION-v001; (2) selected_q segment endpoints
must be transported to physical_dense exactly once at entry to each per-sample
SingleStageDetector.post_processing call, before filtering, top-k, IoU or NMS.

Return a durable Markdown plan containing:

1. The exact existing symbols and file paths to change, limited to no more than
   ten production/test/config files.
2. A call-site map proving that the data-side and selector-side uniform paths
   will share one canonical implementation, rather than duplicated arithmetic.
3. The concrete pre-NMS insertion point and the coordinate-state guard that
   prevents unknown or double mapping without changing detector/head/loss/NMS
   configuration/evaluator/split/class map.
4. The focused static/tiny-fixture tests to add later, including the T=768,
   K=384 endpoint, constant-density bit identity, and order-sensitive NMS
   counterexample. List test commands only as future P1 candidates; do not run
   them.
5. A rollback/minimality statement and any blocking ambiguity.

Strict P0 boundary: no file edits, no code execution, no local/remote CPU or
GPU, no tests, no dataset access, no metrics, no Slurm, no Git push. Do not use
or inspect the quarantined previous Builder patch. The plan must cite the
accepted Pro decision and finish with a no-execution attestation.
