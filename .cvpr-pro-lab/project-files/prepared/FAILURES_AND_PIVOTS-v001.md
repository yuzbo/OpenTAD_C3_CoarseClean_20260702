---
doc_id: FAILURES_AND_PIVOTS
version: v001
date: 2026-08-10
stage: DRAFT
status: active
author_role: coordinator
github_repo: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
supersedes: none
---

# Failures and Pivots

## Scientific failures or negative evidence

- Earlier direct DUCA variants did not fairly establish learned selection over cost-matched uniform sampling.
- Naive selected-rank decoding created a coordinate mismatch; the invariant is to map q to physical t before official NMS.
- Local-cell selection, actionness top-k, selected-rank decode, hard K384 query deletion/SparseHead, DCSR-G1, and ODF-CR-G2 were rejected or negative in their evaluated scopes.
- A fixed-budget selector alone cannot support the dynamic-budget paper claim.

## Engineering failures that are not model evidence

Prior Slurm recovery lines encountered missing runtime symlinks, missing mask propagation, static temporal-axis assumptions under dynamic K, and checkpoint/receipt failures. These motivate regression tests and fail-closed launch contracts but do not imply scientific failure or success.

## Current pivot

Prioritize the smallest decisive model experiment: first prove or falsify task-aware learned positions at fixed matched K; only then test per-video dynamic budgets with strong K-sequence controls. Avoid building a general admission simulator, Monte Carlo infrastructure, or a broad scheduling platform before method feasibility is established.

## Anti-repetition rule

Do not reintroduce rejected coordinate semantics, reuse partial failed-root metrics, compare unmatched budgets, or treat infrastructure checks as paper evidence.
