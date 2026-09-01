---
doc_id: PRO_P0_BLOCKER_DECISION
version: v001
status: accepted_pending_project_source_confirmation
date: 2026-08-11
supersedes: PRO_INITIAL_REVIEW-v002.md
stage: DRAFT
author_role: pro
github_repo: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
project_id: g-p-6a796fef9a00819194024cf1de3bd697
turn_id: duca-p0-blocker-51c88fd75537120ce96a417beb7e81dd
nonce: 51c88fd75537120ce96a417beb7e81dd
source_transcript: .cvpr-pro-lab/pro-reviews/runs/duca-p0-blocker-51c88fd75537120ce96a417beb7e81dd/raw-response.md
---

# PRO_P0_BLOCKER_DECISION-v001 — accepted first-author decision

## Route and evidence status

The web Pro model, acting as Scientific First-Author Agent and Primary
Research Owner, issued `SCIENTIFIC_DECISION: REVISE`. The active candidate
remains `DUCA_FIXEDK_BOUNDED_MONOTONE_DENSITY_ACQUISITION-v001`: fixed
requested K=384 physical-frame acquisition before the heavy backbone, positive
temporal density with monotone inverse-CDF positions, bounded geometry, an
unchanged detector, and selected-to-physical raw-proposal transport before
unchanged NMS.

This is a narrow P0 semantic repair, not a new method route. Evidence remains
`BLOCKED_PRE_RESULT`; no result, cost, model-quality, GPU, dataset, metric, or
held-out evidence was used or created. The complete browser-text transcript is
preserved at `source_transcript` above.

## P0 blockers confirmed by Pro

1. The data and selector uniform constructions disagree at normal T=768,
   K=384: their terminal physical indices are 766 and 767 respectively.
2. The generic `SingleStageDetector` path applies NMS before the existing
   selected-to-dense inverse mapping, which violates the unchanged-detector
   coordinate contract.

## Frozen repair contract

- Prefix-contiguous valid length `T_v` uses
  `K_eff = min(384, 16 * floor(T_v / 16))`; `T_v < 16` fails closed.
- Canonical uniform positions use integer half-up endpoint arithmetic:
  `u_j = floor((2*j*(T_v-1)+(K_eff-1))/(2*(K_eff-1)))`.
  At T=768 and K=384 this ends at 767. Floating linspace, banker rounding,
  tolerance-based repair, clipping, deduplication, and a second generator are
  forbidden.
- The exact constant-density hard-forward case must be bit-identical to the
  canonical generator; near-constant learned density does not use this
  specialization.
- Raw segment endpoints in `selected_q` use the end-exclusive domain
  `[0,K_eff]`; they are mapped exactly once, via strictly increasing knots, to
  `physical_dense` `[0,T_v]` at the entry to each per-sample
  `SingleStageDetector.post_processing`, before filtering, top-k, IoU, or NMS.
  Scores, labels, detector/head/losses, NMS callable/configuration, evaluator,
  split, and class map stay unchanged. Unknown or double coordinate mapping
  fails closed.

## Authorized bounded queues

- `BUILDER_DUCA_P0_CANONICAL_TRANSPORT-v001`: one clean exact-commit worktree,
  one patch attempt, at most six task-hours and ten production/test files.
  It writes the canonical generator, replaces the two call sites, adds the
  constant-density specialization and pre-NMS adapter, prepares fixtures/tests
  and config diff, and returns a no-execution attestation.
- `EVALUATOR_DUCA_P0P1_AMENDMENT-v001`: protocol-only amendment, maximum three
  task-hours, no data/model/CPU/GPU/metric access. It records P1 remote-CPU
  commands and receipt schema without executing them and keeps PRE_RUN blocked.
- `CRITIC_DUCA_P0_CLOSURE-v001`: a read-only three-hour closure review after
  the complete Builder diff, including one order-sensitive NMS counterexample.
  It returns only `P0_STATIC_PASS` or `P0_BLOCKED`.

Timed-out Builder/Critic worktree content is excluded: it must not be resumed,
cherry-picked, or used as evidence.

## Stop and authority boundary

P0 is blocked until canonical fixture identity, endpoint identity,
constant-density bit identity, detector/config invariance, pre-NMS mapping,
coordinate-state safety, Critic closure, and Evaluator amendment all have
durable receipts. P1 is not authorized. P2 PRE_RUN is blocked.

`REMOTE_GPU=PROHIBITED`: the decision authorizes no GPU initialization,
training/evaluation, Slurm submission, dataset traversal, metric computation,
validation/test access, dynamic-K work, Git push, result promotion, or paper
claim expansion. A later fresh Pro decision must admit P1 before any remote
CPU execution, and must separately specify any future GPU experiment.
