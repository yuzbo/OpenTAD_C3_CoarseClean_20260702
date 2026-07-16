---
type: experiment_tracker
node_id: exp:chronotransport-r2-stop-chain
idea: idea:chronotransport
protocol: CT-P3R-3S-r2
updated: 2026-07-16
implementation_status: repair_authorized
registration_status: NOT_READY
experiment_running: false
---

# ChronoTransport r2 execution tracker

| ID | Dependency | Purpose | Status | Immutable evidence |
|---|---|---|---|---|
| W0 | approved spec | current plan and source classification | IN_PROGRESS | plan commit pending |
| W1 | W0 | A1 registration/random controls | LOCKED | none |
| W2 | W1 | A2 Slurm/environment identity | LOCKED | none |
| W3 | W2 | filesystem/source/import integrity | LOCKED | none |
| W4 | W3 | real ActionFormer per-window loss | LOCKED | none |
| W5 | W4 | Stage-C A3/A4 transaction | LOCKED | none |
| W6 | W5 | formal Stage-C/matched/Gate-4 workflows | LOCKED | none |
| W7 | W6 | remote verification, I and R | LOCKED | none |
| E0 | valid R | Slurm PRECHECK | LOCKED | none |
| E1 | E0 | Gate 1 | LOCKED | none |
| E2 | Gate 1 PASS | Stage B, three seeds | LOCKED | none |
| E3 | E2 complete | Gates 2/3 | LOCKED | none |
| E4 | Gates 2/3 PASS | Stage C plus matched dense | LOCKED | none |
| E5 | E4 plus post-Stage-C Gate 3 PASS | Gate 4 | LOCKED | none |

## Current facts

- Authority: spec commit `537f692189cf0c5a6ee7d40ad8c4ed1032bf1d37`, SHA-256
  `E79DFAAB8F9B0093E96CBD6B46BEF4ECF8D6433009E2DCB922AD0F4C473B27A6`.
- Spec review: `APPROVE_SPEC_FOR_PLAN`.
- Implementation review: `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`.
- No implementation I, registration R, PRECHECK, ChronoTransport Job ID, Gate artifact or experiment result
  exists at tracker creation.
- The 2026-07-16 read-only Slurm check showed unrelated DUCA/S1 jobs only; none is reused or modified.

## Status semantics

- `LOCKED`: dependency or registration contract is not met; execution forbidden.
- `IN_PROGRESS`: implementation work only unless the row is E0--E5 and has a recorded Slurm job.
- `TESTED`: exact listed implementation bytes passed the recorded verification; not scientific evidence.
- `PASS`/`FAIL`: only a validated formal terminal artifact may assign this status to E1/E3/E5.
- `INVALID_IMPLEMENTATION`: isolate the run, repair, re-review and rerun; do not convert it to science FAIL.
