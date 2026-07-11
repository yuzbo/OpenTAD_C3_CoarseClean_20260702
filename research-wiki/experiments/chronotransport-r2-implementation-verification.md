---
type: experiment
node_id: exp:chronotransport-r2-implementation-verification
title: "ChronoTransport r2 implementation verification"
idea: idea:chronotransport
verdict: ongoing
confidence: high
commit: "pending-batch-commit"
jobs: "remote CPU focused pytest in workdirs/chronotransport_r2/repo"
updated: 2026-07-12
---

# ChronoTransport r2 Implementation Verification

## Verified scope

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
`/data/run01/sczc063/yuzibo/workdirs/chronotransport_r2/repo`. The final implementation commit will
replace `pending-batch-commit` after the source batch is committed.

Remote scheduling audit found no active allocation in the SSH session (`CUDA_VISIBLE_DEVICES` unset,
no `SLURM_JOB_ID`). Formal GPU1 execution is therefore not yet authorized; login-node training remains
forbidden.

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
