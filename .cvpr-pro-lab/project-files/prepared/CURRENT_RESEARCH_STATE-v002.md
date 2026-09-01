---
doc_id: CURRENT_RESEARCH_STATE
version: v002
status: active_pending_project_source_confirmation
date: 2026-08-11
supersedes: CURRENT_RESEARCH_STATE-v001.md
stage: DRAFT
author_role: coordinator
github_repo: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
---

# DUCA current research state

## Scientific owner and active decision

The web Pro model is the acting Scientific First-Author Agent and Primary
Research Owner. The accepted decision is `PRO_INITIAL_REVIEW-v002` with
`SCIENTIFIC_DECISION: REVISE`.

The candidate primary mechanism is fixed-K bounded monotone physical-density
acquisition before the heavy backbone: deploy-visible low-cost scout, positive
density, exact/unique/increasing K positions, external physical-to-selected
label mapping, and selected-to-physical raw-proposal mapping before NMS. The
clean AdaTAD/ActionFormer detector, its assignment, losses, NMS, evaluator,
and official split are the control truth. Dynamic K, quota-policy stacks,
detector-head modifications, and current physical-grid-head routes are not
admitted as the primary route.

## Paper and evidence boundary

Task: offline THUMOS14 TAD. Candidate claims remain limited to fixed-K
physical acquisition under unchanged-detector, equal-heavy-frame and later
full-stack equal-cost comparisons. Evidence remains `BLOCKED_PRE_RESULT`:
there are no formal metrics, costs, or paper claims.

## Immediate bounded work

Builder implements the smallest faithful fixed-K plugin and clean uniform
wrapper. Critic audits novelty, invariants, fairness, leakage, NMS order, and
cost scope. Evaluator prepares the Stage-A protocol and metric seal. No role
may read official metrics or produce a paper-quality result before the next
accepted Pro decision.

Human policy: experimental runs execute remotely. No GPU run is currently
authorized by the accepted decision because it names no concrete GPU
experiment; conditional continuous GPU authorization applies only after a
specific Pro-requested experiment passes `PRE_RUN_READY`. Held-out access, Git
push, route freeze, result/claim promotion, and submission remain separately
gated.

## Current blockers

The fixed-K plugin, clean uniform parity, detector-invariance proof,
pre-NMS mapping proof, gradient-admission evidence, protocol lock, independent
novelty audit, and full-stack cost receipt do not yet exist.
