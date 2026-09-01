---
doc_id: CURRENT_RESEARCH_STATE
version: v006
status: prepared_pending_central_project_source_sync
date: 2026-08-11
supersedes: CURRENT_RESEARCH_STATE-v005.md
stage: DRAFT
author_role: coordinator
github_repo: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
pro_decision: PRO_P0_ROUTE_ADJUDICATION-v002
evidence_class: BLOCKED_PRE_RESULT
---

# DUCA current research state

## Scientific owner and route

Web Pro is the acting Scientific First-Author Agent and Primary Research Owner.
Its accepted fresh decision `PRO_P0_ROUTE_ADJUDICATION-v002` is `REVISE`.
The active candidate is now
`DUCA_FIXEDK_BOUNDED_DENSITY_QUANTILE_ACQUISITION-v002`: fixed requested K=384
physical pre-backbone acquisition, a newly named per-time density-only input,
endpoint-inclusive inverse-CDF positions, bounded integer geometry, an unchanged
detector, and selected-q-to-physical-dense transport before unchanged NMS.

## Frozen P0 definition

`duca_density_logits[b,t]` is the sole learned density input. It comes from a
dedicated density-only reader attached to dense `browser_memory` with
`selection_unit=1` and an identity physical candidate grid. Existing slots,
allocation, actionness/boundary heads, top-k/rank/quota policies and soft
transport are excluded as density sources. A per-sample hard decoder uses fixed
`1e-6 + softplus`, trapezoidal interval masses, endpoint-inclusive inverse-CDF
quantiles, exact-constant specialization to the canonical uniform generator,
and deterministic constrained integer positions. It has no admitted gradient
surrogate or detector-gradient path.

The canonical integer-half-up exact-uniform generator and pre-NMS
selected-q-to-physical-dense transport remain separate claim-neutral correctness
corrections. They are not performance or novelty evidence.

## Immediate work and boundaries

Builder has only a plan-only task: produce a current minimal change plan at the
pinned commit before any edit or execution. Critic begins only after the full
Builder diff; Evaluator later prepares a no-execution protocol amendment. P1,
PRE_RUN, tests, data traversal, CPU/GPU/Slurm, metrics, checkpoints, push,
result promotion and paper claims remain blocked.

Evidence remains `BLOCKED_PRE_RESULT`. The full original Pro response is stored
locally with the accepted route decision; both await a central Project-Sources
lease for remote confirmation.
