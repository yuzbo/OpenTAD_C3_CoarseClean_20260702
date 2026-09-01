---
doc_id: RESULT_ANALYSIS
version: v001
date: 2026-08-10
stage: BLOCKED_PRE_RESULT
status: pre_result
author_role: coordinator
github_repo: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
supersedes: none
---

# Result Analysis

## Status

**BLOCKED_PRE_RESULT: no sealed matrix exists, so quantitative model analysis is not permitted.**

## Predeclared analysis once results exist

1. Validate identity, split, checkpoint, evaluator, seed, requested/effective K, and cost receipts before reading metrics.
2. Compare DUCA fixed-K against exact-uniform fixed-K and mixed-K exposure controls.
3. Decompose any change by IoU threshold, short-action subset, action duration, boundary error, and video difficulty.
4. For dynamic K, compare against the identical K sequence with uniform positions and a K-histogram shuffle control.
5. Report mean, dispersion/interval, paired seed differences, and full-stack cost rather than a best seed.
6. Test whether any gain survives scout/selection overhead and whether K allocation or position allocation caused it.
7. Apply stop rules before expanding to additional detectors or budgets.

## Prohibited interpretations

No partial metric may be used to claim superiority, feasibility, convergence, or publishability. Engineering correctness is not model-quality evidence.
