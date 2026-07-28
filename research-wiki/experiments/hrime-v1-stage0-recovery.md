# H-RIME v1 Stage-0 Recovery and Deterministic Core

## State

- User authorization: `approved`
- Design: `designed`
- Stage-0 code: `implemented`
- Deterministic H-RIME core: `implemented`
- Focused pure-CPU verification: `tested`
- Torch-dependent verification: `remote_pending`
- Slurm recovery transaction: `not_yet_submitted`
- Same-total-cost oracle: `not_yet_run`
- Learned H-RIME: `not_yet_implemented`
- Empirical support: `not_yet_empirically_supported`
- Paper status: `not_yet_paper_ready`

## Implemented scope

Stage 0 repairs the true short-window execution/ledger contract, clean-cwd
checkpoint compaction, hash-bound dense-checkpoint salvage, recovery-DAG
selection and the default Phase-4 seal. Salvage never writes into the failed
`d9d454cd` root and never reclassifies jobs `1198115` or `1198116` as
successful.

The deterministic H-RIME core implements:

1. canonical nominal/effective-K aliases at execution quantum 16;
2. exact reachable-total enumeration and cap projection;
3. exact-equality multiple-choice knapsack with frozen integer score protocol;
4. stable video-window grouping and shared-scan input/summary receipts;
5. video budget-plan, solver-input and assignment hashes;
6. existing-selector replay using canonical nominal aliases;
7. homogeneous effective-K bucket ordering and inverse restoration.

This is an algorithm and execution contract. It is not yet the connected
learned model and does not establish a performance result.

## Verification

Local checks passed:

- Python compilation of Stage-0 salvage and H-RIME core;
- Bash syntax for the recovery submitter, salvage launcher and code gate;
- the expanded 64-test focused non-Torch suite, including brute-force solver
  agreement, explicit Stage-0 budget-truth sealing and
  provenance-tamper rejection.

The first mandatory clean-commit MAX audit found one deployment blocker:
Stage-0 fixed-window ledgers were not bound to the physical-protocol hash and
did not expose raw/reachable/realized/projection-unused/solver-unused budget
truth. This was corrected before any Slurm submission. The finalizer now seals
and aggregates these fields or fails closed. The review must be rerun on the
corrected exact commit.

The local Windows process cannot load the CUDA-linked `torch` DLL. Therefore
selector/backbone/detector checks must pass remotely before the state can become
`tested` for the complete Stage-0 implementation.

## Next gate

1. commit and push the exact code snapshot;
2. create the frozen salvage manifest bound to that recovery commit and the two
   verified epoch-59 raw checkpoints;
3. run Slurm precheck/code gate;
4. release a fresh recovery DAG with `DUCA_RIME_DENSE_RECOVERY_MODE=salvage`
   and `DUCA_RIME_ENABLE_PHASE4=0`;
5. require Phase-1, dense recovery, Phase-2 and Phase-3 development receipts;
6. run the held-out same-total-cost H-RIME oracle before learned planner
   training.

Correct empirical statement:

`No paper-admissible empirical conclusion is available yet.`
