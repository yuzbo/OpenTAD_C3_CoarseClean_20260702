---
type: experiment_tracker
node_id: exp:chronotransport-r2-stop-chain
idea: idea:chronotransport
protocol: CT-P3R-3S-r2
updated: 2026-07-16
implementation_status: repair_in_progress
registration_status: NOT_READY
experiment_running: false
---

# ChronoTransport r2 execution tracker

| ID | Dependency | Purpose | Status | Immutable evidence |
|---|---|---|---|---|
| W0 | approved spec | current plan and source classification | TESTED | plan `4dcc98d`; classification SHA-256 `7FBD6FC4...72445D`; exact 47-file inventory / 21 tests |
| W1 | W0 | A1 registration/random controls | TESTED | remote cumulative 43 prior passes + 9 repaired contract passes + 36 control/manifest passes |
| W2 | W1 | A2 Slurm/environment identity | IN_PROGRESS | no formal GPU check before valid I/R |
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
- W0 explicitly classifies all 47 tracked ChronoTransport test/tool/script paths. Eighteen of the 21 tests
  and all current r2 producers/validators are `REQUIRED`; three legacy tests are
  `TEST_ONLY_NON_FORMAL`; legacy entrypoints are `OUT_OF_SCOPE`. Any unclassified addition or source-vector
  drift now fails closed.
- W1 binds the approved `537f692` spec, freezes all unsuffixed random controls to integer seed 3407,
  recomputes every random action hash from the actual registered window order and rejects missing, string or
  alternate seeds. The first remote run exposed and retained a real generator RED (43 passed, 1 xfailed,
  1 failed); the corrected key contracts then passed 9/9 and the control/manifest matrix passed 36/36.
- These are implementation tests only. Registration is still `NOT_READY`; no I/R, PRECHECK, GPU producer,
  Gate, training run or scientific result exists.

## Status semantics

- `LOCKED`: dependency or registration contract is not met; execution forbidden.
- `IN_PROGRESS`: implementation work only unless the row is E0--E5 and has a recorded Slurm job.
- `TESTED`: exact listed implementation bytes passed the recorded verification; not scientific evidence.
- `PASS`/`FAIL`: only a validated formal terminal artifact may assign this status to E1/E3/E5.
- `INVALID_IMPLEMENTATION`: isolate the run, repair, re-review and rerun; do not convert it to science FAIL.
