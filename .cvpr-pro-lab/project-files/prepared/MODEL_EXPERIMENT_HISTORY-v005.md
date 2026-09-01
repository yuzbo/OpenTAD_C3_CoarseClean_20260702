---
doc_id: MODEL_EXPERIMENT_HISTORY
version: v005
status: pending_central_project_source_sync
date: 2026-08-11
supersedes: MODEL_EXPERIMENT_HISTORY-v003.md
stage: DRAFT
author_role: coordinator
github_repo: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
evidence_class: BLOCKED_PRE_RESULT
---

# DUCA model and experiment history

## 2026-08-11 — P0 static plan exposed an unresolved method definition

`PRO_P0_BLOCKER_DECISION-v001` remains the latest accepted scientific decision
and leaves `DUCA_FIXEDK_BOUNDED_MONOTONE_DENSITY_ACQUISITION-v001` as a
candidate, not an implemented method. The subsequent Builder plan
`BUILDER_DUCA_P0_MINIMAL_CHANGE_PLAN-v002` made no code change or execution.
It located a shared canonical-uniform repair and a pre-NMS coordinate-transport
insertion, but could not bind the required constant-density specialization.

Independent Critic return
`CRITIC_DUCA_P0_PLAN_AMBIGUITY_CLASSIFICATION-v001` classifies the missing
density-to-inverse-CDF hard decoder as `SCIENTIFIC_AMBIGUITY`: no frozen symbol
is the authoritative per-time positive density, and the nearest existing slot
allocation/top-k/rank utilities have different semantics. The critic permits no
interpretive implementation choice.

No experiment changed state. There are no patches, tests, remote commands,
datasets, checkpoints, metrics, costs, GPU/Slurm tasks, result comparisons, or
claims from this event. The sole next step is a fresh Pro decision that either
binds the exact density decoder and its selector insertion point or revises the
candidate route before any patch or P1 admission.
