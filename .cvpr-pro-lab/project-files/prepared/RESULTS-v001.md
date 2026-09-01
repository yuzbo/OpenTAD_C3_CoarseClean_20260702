---
doc_id: RESULTS
version: v001
date: 2026-08-10
stage: BLOCKED_PRE_RESULT
status: pre_result
author_role: coordinator
github_repo: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
supersedes: none
---

# Results

## Formal result status

**BLOCKED_PRE_RESULT: no formal result is available for paper claims.**

There is no cvpr-pro-lab-sealed, full official split, multi-seed result matrix tied to the fixed GitHub revision. Therefore this source reports no DUCA mAP, no gain over AdaTAD/uniform sampling, and no dynamic-budget advantage.

## Excluded evidence

The following remain engineering or historical diagnostic evidence and must not be promoted:

- training-intermediate, subset, single-seed, failed-root, or unsealed metrics;
- over-budget or protocol-mismatched runs;
- train-domain bootstrap, Oracle, synthetic, proxy, or upstream numbers;
- results produced from unpushed local code;
- any number without official evaluator, complete split, terminal checkpoint, cost receipt, and immutable experiment identity.

The previously discussed AdaTAD uniform 0.5 reference near mAP 65 is a comparison target to reproduce under a locked official protocol, not evidence that the current DUCA implementation is better.
