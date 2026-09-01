---
doc_id: ROUTE_DECISION
version: v001
date: 2026-08-10
stage: DRAFT
status: active
author_role: coordinator
github_repo: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
supersedes: none
---

# Route Decision

## Current decision

Continue with DUCA-RIME as a falsifiable candidate, not as the final paper method. The preferred route is a low-cost full-video scout that predicts a discrete per-video budget and a monotone exact-budget acquisition plan before the heavy backbone.

## Candidate allocation families

Before freezing the allocator, compare only a small decisive set:

1. Independent position prediction for each candidate budget K.
2. Strictly nested acquisition, where larger budgets refine smaller-budget positions.
3. At most one weak-overlap compromise, admitted only if train/utility-only Oracle regret indicates that strict nesting is materially restrictive.

## Invariants

- Budget decisions use deploy-visible inputs only.
- Average realized budget, not only requested K, is constrained.
- Selected positions are unique, monotone, and mapped to physical time.
- Dense-video endpoints, coverage, and maximum-gap protection are explicit.
- Detector proposals are mapped from selected-coordinate q to physical-coordinate t before unchanged official NMS/evaluation.
- Validation/test ground truth, teacher outputs, raw-prediction caches, or counterfactual ledgers never enter the deploy path.

## Decision status

This is a design hypothesis. No formal result currently proves dynamic K or learned placement better than cost-matched uniform controls. The first Pro review must decide CONTINUE, REVISE, PIVOT, STOP, or ESCALATE_HUMAN.
