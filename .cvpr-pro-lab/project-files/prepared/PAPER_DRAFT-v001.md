---
doc_id: PAPER_DRAFT
version: v001
date: 2026-08-10
stage: BLOCKED_PRE_RESULT
status: pre_result
author_role: coordinator
github_repo: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
supersedes: none
---

# Paper Draft Skeleton

## Working title

DUCA-RIME: Dynamic-Budget Task-Aware Temporal Acquisition for Offline Temporal Action Detection

## Abstract skeleton

Offline temporal action detectors commonly process a fixed temporal grid, even though video difficulty and redundancy vary strongly. We study whether a low-cost scout can allocate both the heavy-frame budget and physical-time acquisition positions before the backbone while preserving precise localization. The proposed candidate couples deploy-visible video-level budget prediction with monotone exact-budget acquisition, coverage safeguards, and physical-coordinate transport into an otherwise controlled detector. The paper will report only cost-matched, full-split, multi-seed results after the registered experiments are sealed. Quantitative claims are intentionally blank at this stage.

## Planned paper structure

1. Introduction: fixed-grid inefficiency versus TAD localization risk.
2. Related work: offline TAD, adaptive temporal computation, token/frame selection, and AdapTok.
3. Method: low-cost scout, per-video budget controller, physical-time acquisition, coordinate transport, losses, and complexity.
4. Experiments: official THUMOS protocol, strong uniform/mixed-K controls, full-stack cost, multi-seed statistics, and leakage safeguards.
5. Analysis: high-IoU, short actions, boundaries, allocation attribution, and failure modes.
6. Limitations: offline setting, scout overhead, dataset scale, and generalization.

## Current writing boundary

**BLOCKED_PRE_RESULT.** The manuscript may describe the research question and registered method, but must not state numerical superiority, final novelty, or paper readiness before formal evidence and literature verification.
