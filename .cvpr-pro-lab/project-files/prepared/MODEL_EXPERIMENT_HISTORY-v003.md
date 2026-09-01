---
doc_id: MODEL_EXPERIMENT_HISTORY
version: v003
status: active_pending_project_source_confirmation
date: 2026-08-11
supersedes: MODEL_EXPERIMENT_HISTORY-v002.md
stage: DRAFT
author_role: coordinator
github_repo: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
---

# DUCA model and experiment history

## 2026-08-11 — fresh P0 blocker decision v001

Fresh Project turn `duca-p0-blocker-51c88fd75537120ce96a417beb7e81dd` was
accepted after matching its Project ID, nonce, GitHub revision,
`CURRENT_RESEARCH_STATE-v002`, and `MODEL_EXPERIMENT_HISTORY-v002`. It
contained no other-project material. The response is preserved locally as a
versioned browser-text transcript and summarized in
`PRO_P0_BLOCKER_DECISION-v001`.

Pro retained the fixed-K bounded monotone-density route and issued `REVISE`.
The decision quarantines all timed-out Builder/Critic partial state and
identifies two pre-result correctness blockers: inconsistent uniform endpoints
between data and selector paths, and post-NMS rather than pre-NMS coordinate
transport. These are not model results.

The sole admitted next package is a clean no-execution patch, an amended
protocol, and a static closure review. Pro explicitly withheld remote CPU P1,
P2, GPU, Slurm, dataset/metric access, and all result or claim progression.
