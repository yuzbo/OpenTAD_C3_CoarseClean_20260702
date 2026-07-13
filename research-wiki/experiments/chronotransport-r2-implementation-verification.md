---
type: experiment
node_id: exp:chronotransport-r2-implementation-verification
title: "ChronoTransport r2 implementation verification"
idea: idea:chronotransport
verdict: ongoing
confidence: high
commit: "33378af"
jobs: "remote CPU focused pytest in workdirs/chronotransport_r2/repo"
updated: 2026-07-13
---

# ChronoTransport r2 Implementation Verification

## Verified manifest/protocol repair slice

- Exact 200-video label-free manifest/deep re-derivation, canonical raw-byte and sidecar validation,
  duplicate-key rejection, Stage-B hash-bound exposure artifact, Stage-C balance/cursor validation,
  frozen control/library identities, and the legacy formal-runner hard lock are remotely `tested`.
- The TDD chain recorded the expected missing-symbol RED (1 collection error, 35.37 seconds), then a
  source-vector integrity RED where a registry-provided sampled index could be trusted, followed by
  strict-type/canonical-byte negative checks. After repair, the focused remote manifest/protocol suite
  passed 27/27 in 53.81 seconds.
- The remote protocol/control/legacy-runner compatibility matrix passed 55/55 in 91.90 seconds. Local
  `py_compile` and `git diff --check` also passed; these local checks are static evidence only.
- This tested slice does not change the overall `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION` verdict,
  create registration `I/R`, or unlock any formal Gate.

## Verified scope

### Gate-1 cost/profile slice under revision

The first exact-cost/full-stack implementation draft passed a remote focused/adjudication/core matrix
of 36/36 in 37.88 seconds, correctly deriving `B*` from direct `periodic4_transport` total-ms p50 and
enforcing the 20% dense-saving hard condition. Independent specification and code-quality reviews both
returned `REVISE_GATE1_SLICE`: arbitrary factories/provenance and arbitrary 30+30 record IDs were not
bound to registration; the exact 23-item profile set/order was not frozen; strict scalar schemas and
the safety-override invalidation were incomplete. This draft is `implemented_under_revision`, not an
approved profiler, formal cost artifact, or Gate-1 result. Deep registration-bound repair is in
progress.

- Protocol canonicalization, label-free split/window helpers, Stage-B exposure and Stage-C exposure:
  remote `tests/test_chronotransport_r2_protocol.py`, 7 passed.
- Frozen r2 candidate library, motion/random exact-count controls, dual-age cache contract, and legacy
  core cache regression: remote focused suite, 36 passed.
- Runtime all-row adapter writeback, current-row live gradient, detached historical cache,
  requested/executed action separation, forced-dense/integration regressions: remote focused suite,
  35 passed.
- Fixed window-level D=23 mean/max quantile head, true-age feature, dense external safety semantics,
  and core scheduler regressions: remote focused suite, 30 passed.
- Gate 1 equal-cost oracle-headroom and Gate 2 matched TRANSPORT/HOLD pure adjudicators: remote
  synthetic focused suite, 4 passed.
- Stage-C object-identity ownership and loss-specific AMP gradient assignment: remote focused suite,
  4 passed. Overflow retry and the formal 4,200-update runner remain pending.
- Pre-Gate1 registration schema/claim chain, Gate 1 CLI, r2 Stage-B/C config overlays, and guarded
  GPU1 launcher: remote 4 tests passed plus launcher `bash -n`.

## Evidence boundary

These are behavioral implementation checks, not Gate 1--4 results. Formal Stage B/C, full-stack
timing, registration, and formal GPU1 deployment are not yet verified by this record. No scientific
claim is unlocked.

## Remote provenance

Environment: `/data/run01/sczc063/yuzibo/conda_envs/opentad`; isolated verification workdir:
`/data/run01/sczc063/yuzibo/workdirs/chronotransport_r2/repo`. The bounded protocol repair commit is
`33378af`; it is not the final implementation commit `I`.

Remote scheduling audit found no reusable physical-GPU1 allocation. On protected job `1137541`,
Slurm reported physical `GRES IDX:4` and the in-step read-only probe reported
`SLURM_STEP_GPUS=4`, `CUDA_VISIBLE_DEVICES=0`, confirming task/cgroup ordinal remapping. The old
launcher invariant `CUDA_VISIBLE_DEVICES=1` is therefore invalid under a single-GPU protected step.
The corrected guard must require physical `SLURM_STEP_GPUS` (or `SLURM_JOB_GPUS`) exactly `1`,
`SLURM_GPUS_ON_NODE=1`, and remapped local `CUDA_VISIBLE_DEVICES=0`; no current allocation satisfies
that contract. Formal GPU1 execution remains unauthorized and login-node training remains forbidden.

## Combined regression

Remote combined static/focused verification passed: 110 tests in 84.58 seconds, including every new
r2 test, existing ChronoTransport core/integration/formal-Stage-B/repository contracts, and the two
required C3 focused suites. This confirms the currently implemented surfaces only; it does not fill the
known missing Gate-3/Gate-4 adjudicators, overflow retry, full formal runners, or create registration R.

## Independent audit

Two independent follow-up reviewers approved the bounded protocol repair as
`APPROVE_PROTOCOL_SLICE` and `APPROVE_PROTOCOL_QUALITY` after reproducing and closing strict-type,
path, source-vector, canonical-byte, and rehashed-identity fail-open cases. The approved slice was
committed and pushed as `33378af`. These approvals are not approval of full r2 registration or Gates.

A fresh no-conversation-context agent returned `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`. Seven
registration-blocking gaps remain: Gate 3/4, executable r2 Stage B/C/matched dense, overflow retry, B*
and exact cost feasibility, full-stack profiling/provenance, strict derived registration validation,
and a registration-bound fixed Gate-1 input chain. I/R and formal deployment remain locked.

## External Pro GitHub audit

A GitHub-only Pro review of immutable snapshot `4b07020acb2611c3f085488d2f678f3be037f1be`
independently affirmed all seven blockers and found two additional P0 defects in surfaces that the
project-reported suite had treated as covered:

1. The r2 config overlay targets `model.backbone.chronotransport` instead of the actual inner
   `model.backbone.backbone.chronotransport` runtime.
2. The conformal helper flattens `window×candidate` residuals instead of taking a per-window maximum
   before rank 28 over the 30 calibration windows.

It also records incomplete formal manifest, per-window Spearman, Stage-C exposure/resume, exact-cost
ledger, and measured-stage profiler contracts. This audit is repository evidence, not an experiment or
independent test rerun. Verdict remains `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`; registration
readiness is `NOT_READY`, and the formal execution chain remains locked.

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
