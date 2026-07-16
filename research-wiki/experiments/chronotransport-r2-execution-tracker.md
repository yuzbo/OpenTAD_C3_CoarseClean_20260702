---
type: experiment_tracker
node_id: exp:chronotransport-r2-stop-chain
idea: idea:chronotransport
protocol: CT-P3R-3S-r2
updated: 2026-07-16
implementation_status: implemented_and_cpu_tested_pending_exact_byte_review
registration_status: NOT_READY
experiment_running: false
---

# ChronoTransport r2 execution tracker

| ID | Dependency | Purpose | Status | Immutable evidence |
|---|---|---|---|---|
| W0 | approved spec | current plan and source classification | TESTED | classification SHA-256 `2713C84D...08D2F`; exact 65-path inventory / 25 matching tests |
| W1 | W0 | A1 registration/random controls | TESTED | remote cumulative 43 prior passes + 9 repaired contract passes + 36 control/manifest passes |
| W2 | W1 | A2 Slurm/environment identity | TESTED | clean `c585ae5`: 78 passed + `bash -n`; Stage-B/G2-3 81 passed then repaired fixture 1/1 |
| W3 | W2 | filesystem/source/import integrity | TESTED | same-descriptor/import/entrypoint/TOCTOU contracts in candidate `6c3606c` |
| W4 | W3 | real ActionFormer per-window loss | TESTED | same-head batch-two aggregate identity and negative contracts |
| W5 | W4 | Stage-C A3/A4 transaction | TESTED | paired CT/matched transaction, retry, state, normalizer and checkpoint contracts |
| W6 | W5 | formal Stage-C/matched/Gate-4 workflows | TESTED | 4,200-success runners, post-Gate3 and official Gate4 producer/finalizer contracts |
| W7 | W6 | remote verification, I and R | IN_PROGRESS | exact CPU 441/1 + C3 20/20; full Pro prompt exceeded one-turn thinking time and is split into same-SHA Part 1/Part 2; re-review and R-bound CUDA checks pending |
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
- No implementation I, registration R, PRECHECK, ChronoTransport Job ID, Gate artifact or experiment
  result exists. Candidate `6c3606cc5161d415909a42741b3bc402278bf332` is review-only.
- The 2026-07-16 read-only Slurm check showed unrelated DUCA/S1 jobs only; none is reused or modified.
- W0 explicitly classifies all 65 tracked ChronoTransport test/tool/script paths. Twenty-two of the 25
  matching tests and all current r2 producers/validators are `REQUIRED`; three legacy tests are
  `TEST_ONLY_NON_FORMAL`; legacy entrypoints are `OUT_OF_SCOPE`. Any unclassified addition or source-vector
  drift now fails closed.
- W1 binds the approved `537f692` spec, freezes all unsuffixed random controls to integer seed 3407,
  recomputes every random action hash from the actual registered window order and rejects missing, string or
  alternate seeds. The first remote run exposed and retained a real generator RED (43 passed, 1 xfailed,
  1 failed); the corrected key contracts then passed 9/9 and the control/manifest matrix passed 36/36.
- W2 separates the registration-time required model/software contract from producer-observed allocation
  identity. The formal observer records raw Slurm visibility/allocation fields, maps the current CUDA PID to
  one full GPU UUID, requires exactly one torch-visible device and logical `cuda:0`, and rejects model,
  driver, CUDA, PyTorch, cuDNN, precision, allocation-hash or MIG mismatches. Gate-1 profile/replay/result,
  precheck and Stage-B provenance bind the observed identity; the launcher no longer assigns
  `CUDA_VISIBLE_DEVICES`, and the latency claim no longer names physical GPU1. Clean remote CPU contracts at
  `c585ae5` passed 78/78 plus `bash -n`; the Stage-B/Gates-2/3 group passed 81 before one stale mock failure,
  then its exact repair passed 1/1. No CUDA action was taken; live allocation validation remains E0.
- W3--W6 are now present in production entrypoints and validators. The final exact remote CPU matrix was
  `441 passed, 1 skipped, 2 warnings in 968.62s`; targeted Gate4 was 32/32, remote py-compile and four r2
  launchers passed syntax, and C3 compatibility was 20/20. Eighteen changed/new files matched the local
  candidate by SHA-256. The protected CUDA skip remains a W7 task after review and valid I/R.
- These are implementation tests only. Registration is still `NOT_READY`; no I/R, PRECHECK, GPU producer,
  Gate, training run or scientific result exists. E0--E5 remain locked in strict stop-chain order.
- The first Pro call against `92a18be` independently established its single-parent, strict post-floor ancestry
  and docs-only diff but returned `GITHUB_SNAPSHOT_INCOMPLETE` because the reviewer interface lacked tree SHA
  and split Git-object timestamps required by the prompt. The user approved a strict equivalent-certificate
  fallback; this changes only the review transport contract and does not approve implementation or unlock E0.
- The 24,050-byte unified Pro prompt then exceeded the model's one-turn thinking duration. The user approved a
  dependency split: Part 1 audits snapshot/registration/Gates 1--3 and emits a complete `PART1_AUDIT_PACKET`;
  Part 2 must bind the same SHA, consume the verbatim output, audit Stage C/Gate 4/Slurm and issue the sole
  overall verdict. This is review orchestration only; W7 remains `IN_PROGRESS` and E0--E5 remain locked.

## Status semantics

- `LOCKED`: dependency or registration contract is not met; execution forbidden.
- `IN_PROGRESS`: implementation work only unless the row is E0--E5 and has a recorded Slurm job.
- `TESTED`: exact listed implementation bytes passed the recorded verification; not scientific evidence.
- `PASS`/`FAIL`: only a validated formal terminal artifact may assign this status to E1/E3/E5.
- `INVALID_IMPLEMENTATION`: isolate the run, repair, re-review and rerun; do not convert it to science FAIL.
