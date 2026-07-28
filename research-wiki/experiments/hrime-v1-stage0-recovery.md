# H-RIME v1 Stage-0 Recovery and Deterministic Core

## State

- User authorization: `approved`
- Design: `designed`
- Stage-0 code: `implemented`
- Deterministic H-RIME core: `implemented`
- Focused pure-CPU verification: `tested`
- Torch-dependent verification: `remote_unit_tested / launchers_prechecked`
- Slurm recovery transaction: `experiment_running / code_gate_scheduler_pending`
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
- the expanded 66-test focused non-Torch suite, including brute-force solver
  agreement, explicit Stage-0 budget-truth sealing and
  provenance-tamper rejection.

The first mandatory clean-commit MAX audit found one deployment blocker:
Stage-0 fixed-window ledgers were not bound to the physical-protocol hash and
did not expose raw/reachable/realized/projection-unused/solver-unused budget
truth. This was corrected before any Slurm submission. The finalizer now seals
and aggregates these fields or fails closed. The review must be rerun on the
corrected exact commit.

The clean-commit re-audit correctly found that the first fix was fail-open at
the launcher/finalizer boundary. The Phase-1 uniform launcher now requires and
hash-verifies the physical protocol, passes its SHA-256 to the finalizer, and
sets `--require-explicit-budget-truth`; the parent pipeline establishes these
variables before launching either uniform budget. Legacy ledger compatibility
remains available only when a caller does not request this strict Stage-0 mode.

The local Windows process cannot load the CUDA-linked `torch` DLL. Therefore
selector/backbone/detector checks must pass remotely before the state can become
`tested` for the complete Stage-0 implementation.

The first recovery submission attempt passed all immutable-input checks but
encountered Slurm `AssocMaxSubmitJobLimit` while building the held DAG. It
created four held jobs before the fifth `sbatch` failed, exposing that the
submitter's `ERR` trap was not inherited inside `submit_job`. Jobs
`1199974`--`1199977` were canceled without release; stale route-local jobs
`1198117` and `1198118` were also canceled because their dependencies can never
be satisfied. No unrelated job was touched. All three transactional RIME
submitters now enable Bash `errtrace` so a nested submission failure cancels
the complete held prefix before exit. The incomplete fresh root is retained as
deployment evidence and is not reusable.

The corrected exact commit
`902168a12bc92babd62b6cb1877ce7137f56cea0` passed a fresh independent MAX
deployment audit. Its commit-bound physical protocol has SHA-256
`1823826b36df3e6bc038743c173fd16c6990e3c2080665dd4f5b77bf88e7e34e`; the
immutable salvage manifest has SHA-256
`b4f5b7fdfe8a491d7fc14d8ffbfbb6f76742de0a458ebcfbc31cc0de53d85a0e`.
The fresh recovery transaction was released at
`/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_902168a1_20260728_183709`.
Its submission-manifest SHA-256 is
`fd6fef65ac01e7830c6b5e337684b19a3bad65c1432f819cfecb32e83dfefb85`.
Jobs `1199978`--`1199983` are the code gate, Phase 1, ActionFormer salvage,
TriDet salvage, Phase 2 and Phase-3 controller respectively. The code gate
completed, but this transaction then failed closed for two deployment reasons:

- Phase 1 retained the base config's repository-relative VideoMAE initialization
  path and failed on the GPU node before evaluation;
- both salvage wrappers directly executed
  `scripts/run_duca_rime_dense_salvage.sh`, whose Git mode was `100644`, and
  exited with code 126 before Python.

Phase 2/controller consequently became `DependencyNeverSatisfied` and were
canceled by exact job IDs `1199982`/`1199983`. The old root, logs and failed
states remain immutable. Only the code-gate and submission receipts exist; no
Phase-1, dense-recovery, Phase-2 or Phase-3 terminal receipt exists.

The source repair now requires and hash-checks the absolute VideoMAE path in the
Phase-1 dense evaluator, passes it through `model.backbone.custom.pretrain`,
records its hash, invokes both salvage arms with explicit Bash, and restores the
script's executable bit. A fresh independent review also required the salvage
terminal evidence to bind that initialization directly; both the source evidence
and recovery receipt now record the resolved path and SHA-256 after a second
in-process hash check. Local Bash syntax and focused launcher/salvage tests pass.
Exact clean commit `0ab242f31be8de7b7da806b645d3aa60d02d8d88` passed an independent
deployment audit, 82 targeted remote Linux/Torch tests, and explicit remote
prechecks for Phase-1 dense, exact-uniform, paired cost, ActionFormer salvage
and TriDet salvage.

The commit-bound physical protocol has SHA-256
`2f11c12d62451c7ec41b54ac889058617f56f889e6f289cfe865a47eb03ff9f9`.
The new immutable salvage manifest has SHA-256
`faab636144d0855f2d8f26d6c7298459302b3c84508bdc2da24b1b864013772d`.
The fail-closed transaction was atomically released at
`/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_0ab242f3_20260728_201613`;
its submission-manifest SHA-256 is
`b996543dfe57bc3678799591f38f0e96e76da971eb8d5a4f7a4edbb15aa3d04d`.
Jobs `1200135`--`1200140` are respectively the code gate, Phase 1,
ActionFormer salvage, TriDet salvage, Phase 2 and Phase-3 controller. The first
snapshot has the code gate priority-pending and every child protected by its
registered `afterok` dependency. Phase 4 remains disabled and official-final
remains sealed.

## Next gate

1. require the exact code-gate receipt on commit `0ab242f3`;
2. require Phase-1, dense recovery, Phase-2 and Phase-3 development receipts;
3. run the held-out same-total-cost H-RIME oracle before learned planner
   training.

Correct empirical statement:

`No paper-admissible empirical conclusion is available yet.`
