---
type: design_spec
title: "DUCA-FoveaSampler / Query-Bridge final implementation contract"
status: user_approved
branch: codex/duca-fovea-query-bridge-20260819
base_commit: 7529fba607f8ddfef74d8309efa466d73a956a60
date: 2026-08-19
---

# DUCA-FoveaSampler / Query-Bridge

The approved final model keeps three manual score branches:
saliency, boundary, uncertainty. It adds query-bank cross-attention with
query-frame contribution A and internal query memory Q1, foveated sampling,
greedy MMR, coarse proposal supervision, teacher-free cycle feedback, and a
task-adapted frozen-VideoMAE heavy path.

Inference is GT-free, teacher-free, cycle-free, cache-free. Q1 never enters
the heavy detector. Official head/loss/NMS/evaluator remain unchanged.
