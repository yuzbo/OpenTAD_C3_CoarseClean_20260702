---
doc_id: EXPERIMENT_PLAN
version: v001
date: 2026-08-10
stage: DRAFT
status: active
author_role: coordinator
github_repo: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
supersedes: none
---

# Experiment Plan

## Stage A: feasibility and attribution

Use the complete official training set and official OpenTAD validation set, a frozen evaluation pipeline, multiple registered seeds, terminal checkpoints, and full-stack cost accounting.

Core cells at the same effective K/cost:

- Dense T768 detector reference.
- Exact uniform fixed K384.
- Uniform mixed-K training with exact-uniform K384 evaluation.
- DUCA learned fixed K384 positions.

The first paper-relevant contrast is learned placement versus exact uniform at identical realized K and detector/training exposure. Dynamic K is not admitted until fixed-K feasibility is established.

## Stage B: dynamic-budget falsification

If Stage A passes, compare dynamic DUCA-RIME against:

- the identical per-video K sequence with uniform positions;
- a shuffled K-histogram control preserving the global budget distribution;
- strong fixed-K and mixed-K controls;
- the smallest decisive independent-versus-nested allocation comparison.

## Measurements

- Official Avg-mAP at tIoU 0.3--0.7 and mAP@0.6/0.7.
- Short-action and boundary-error analysis.
- Realized per-video K distribution and average budget.
- Scout, transport, heavy backbone, detector, memory, latency, and throughput costs measured separately and end to end.
- Multi-seed uncertainty and predeclared result-to-claim thresholds.

## Stop rules

Stop or revise if learned placement cannot beat cost-matched uniform reliably, if gains disappear at high IoU, if the scout erases savings, if dynamic K gains are explained by K distribution alone, or if leakage/fairness/coordinate contracts fail.

## Authorization state

This plan is preregistration material only. No GPU, held-out test, formal experiment, or result promotion is authorized in the current task.
