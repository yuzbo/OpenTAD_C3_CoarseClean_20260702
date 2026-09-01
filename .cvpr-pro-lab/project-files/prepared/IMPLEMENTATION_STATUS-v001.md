---
doc_id: IMPLEMENTATION_STATUS
version: v001
date: 2026-08-10
stage: DRAFT
status: active
author_role: coordinator
github_repo: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
supersedes: none
---

# Implementation Status

## Canonical GitHub revision

The Pro-accessible revision is `63a726a4aaf48ecbf6780bb196de43a890c6b4df` on branch `codex/duca-total60-plugin-cvpr-20260727`.

Key implemented surfaces at that revision include:

- `opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py`: pre-backbone scouts, masked slot transport, selection plans, auxiliary losses, metadata, and a local dynamic-budget path.
- `opentad/models/selectors/pc_ot_mras_dynamic_budget_controller.py`: a deploy-visible discrete budget controller with explicit rejection of GT/oracle/teacher/cache/prediction/checkpoint/result payloads.
- `opentad/models/selectors/pc_ot_mras_reader.py`: lightweight reader components.
- `opentad/models/necks/pc_ot_mras_detector_bridge.py`: selector-to-detector bridge.
- `opentad/models/utils/pc_ot_mras_raw_prediction_guard.py`: raw prediction integrity guard.
- PC-OT-MRAS configs, validators, Slurm launchers, and focused tests.

## What is not established

- The GitHub revision does not constitute a paper-ready DUCA-RIME implementation or a verified dynamic-budget result.
- The local working tree is one commit ahead and heavily modified/untracked. Those changes are not on GitHub and are excluded from this review's code identity.
- Builder, Critic, and Evaluator processes are not yet registered.
- No current formal result artifact is confirmed in the cvpr-pro-lab ledger.

## Immediate implementation audit targets

Verify end-to-end differentiability/optimization, mask and short-window semantics, exact unique K, physical-time mapping before NMS, average-budget enforcement, fair control plumbing, and cost instrumentation. New implementation work must begin from an isolated task/worktree and must not silently consume the dirty coordinator tree.
