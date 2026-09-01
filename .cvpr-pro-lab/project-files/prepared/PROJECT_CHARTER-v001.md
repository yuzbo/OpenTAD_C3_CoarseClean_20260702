---
doc_id: PROJECT_CHARTER
version: v001
date: 2026-08-10
stage: DRAFT
status: active
author_role: coordinator
github_repo: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
supersedes: none
---

# DUCA-RIME Project Charter

## Paper target

Develop a publication-grade method for offline Temporal Action Detection (TAD) that reduces real end-to-end temporal computation before the heavy video backbone while preserving localization, especially at high temporal IoU.

## Scientific question

Can a low-cost, deploy-visible scout allocate both a per-video heavy-frame budget and the physical-time acquisition positions more effectively than cost-matched uniform sampling, without using ground truth, teacher outputs, cached detector predictions, or held-out labels at inference?

## Intended contribution

The candidate DUCA-RIME route combines: (1) dynamic per-video budget allocation under a dataset-level average-cost constraint; (2) exact-budget, monotone physical-time acquisition with coverage and gap safeguards; and (3) an unchanged downstream AdaTAD/OpenTAD detector operating on the selected heavy features with explicit mapping back to physical time before official post-processing.

## Scope and boundaries

- Task: offline TAD, not Online TAD and not causal streaming.
- Primary benchmark: THUMOS14 using the repository's official OpenTAD train/validation protocol.
- Primary metrics: official Avg-mAP over tIoU 0.3:0.1:0.7, mAP@0.6/0.7, and full-stack cost.
- No result is paper-ready until a clean, full-split, multi-seed, cost-matched evaluation is sealed.
- Current state is pre-result. Dynamic budget superiority and learned-position superiority remain falsifiable hypotheses.

## Evidence boundary

The canonical Pro-accessible code is the fixed GitHub revision above. The local worktree contains newer unpushed and dirty material and is excluded from the canonical code context. GPU runs, held-out tests, formal result promotion, Git push, and submission are outside the current authorization.
