# H-RIME v1 Stage-0 Recovery and Deterministic Core

## State

- User authorization: `approved`
- Design: `designed`
- Stage-0 code: `recovery_v4_deployed / uniform_runtime_contract_gap_found`
- Deterministic H-RIME core: `implemented`
- Focused pure-CPU verification: `tested`
- Torch-dependent verification: `remote_unit_tested / launchers_prechecked`
- Slurm recovery transaction: `recovery_v4_failed_closed / scheduler_terminalization_pending`
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

The terminal snapshot at `2026-07-28 21:02 CST` is:

- code gate `1200135`: `COMPLETED`, exit `0:0`;
- Phase 1 `1200136`: `FAILED`, exit `1:0`; its actual exact-uniform inference
  retained the base config's repository-relative VideoMAE initialization path
  because the uniform launcher omitted the absolute runtime override;
- ActionFormer/TriDet salvage `1200137`/`1200138`: `FAILED`, exit `1:0`; both
  produced raw compacted EMA checkpoints and ran their engineering evaluators,
  but structured evidence finalization rejected the frozen `training` subset
  because the salvage role fell through to a `validation` expectation;
- Phase 2/controller `1200139`/`1200140`: `DependencyNeverSatisfied`, then
  canceled by exact ID without touching unrelated jobs.

The exact source remains clean at
`0ab242f31be8de7b7da806b645d3aa60d02d8d88`, and all three registered manifest
hashes still match. Nevertheless, both dense `checkpoint_evidence.json` files,
the Phase-1 and Phase-2 pipeline receipts, and the Phase-3 receipt are absent.
Partial salvage/checkpoint/evaluation sidecars are not terminal evidence. The
root is immutable failed engineering evidence; Phase 4 was never opened and
official-final remains sealed.

## Recovery v3 implementation

The user authorized the exact contract repair and a fresh redeployment. The
local implementation now:

1. binds the absolute VideoMAE path/SHA in exact-uniform precheck, actual
   inference and terminal receipt;
2. introduces a dedicated dense-reference protocol predicate and evaluator
   contract instead of incorrectly adding dense protocols to trainable-RIME
   formal routing;
3. validates both dense backends, selector absence, 768-window dense execution,
   `with_cp=False`, identical development block lists, registered `training`
   subset, saved predictions, initialization identity and official-final
   exclusion;
4. emits `duca_rime_dense_reference_terminal_evaluation_v1` as explicitly
   engineering-only evidence.

Python compilation, Bash syntax and the expanded focused suite passed locally:
`96 passed`.

The exact implementation commit
`bbf051410839f7bec36b0f2cc085de0cd5041cad` passed an independent MAX
deployment review (`GO`). Slurm preflight `1200405` completed 193 authoritative
remote Linux/Torch contract tests. Full launcher preflight `1200462` then passed
Phase-1 dense, uniform K384, uniform K192, paired-cost, ActionFormer salvage and
TriDet salvage contracts. Both salvage checks reported `output_written=false`,
kept source jobs `1198115`/`1198116` failed, and did not consume official-final.

## Recovery v3 deployment

The fresh transaction was atomically released at
`/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_bbf05141_20260728_215335`.
Its immutable identities are:

- physical protocol:
  `69a9cc0b85aaa647a5641f3c00eadd9b8405e8435d3ed5820aae3949df210f4c`;
- production salvage manifest:
  `f7c09b017a4e973211c0f816f55de506d68046801886066dfff3555f15942aef`;
- submission manifest:
  `53a633c162dd69ec3bdfd291e8df97d8e79619d9b688808d0dfad36127abc265`.

Jobs `1200483`--`1200488` are respectively code gate, Phase 1, ActionFormer
salvage, TriDet salvage, Phase 2 and Phase-3 controller. The terminal recovery-v3
state is:

- code gate `1200483`: `COMPLETED`, exit `0:0`;
- ActionFormer/TriDet salvage `1200485`/`1200486`: `FAILED`, exit `1:0`;
- Phase 1 `1200484`, Phase 2 `1200487`, and controller `1200488`: canceled by
  exact ID after the dense failures made full transaction closure impossible.

Both dense jobs had successfully copied and hash-verified their immutable EMA
state, but `tools/test.py` stopped before inference because the launcher did not
export its `DUCA_RIME_EXPECTED_COMMIT` under the formal evaluator's canonical
name `DUCA_EXPECTED_COMMIT`. No dense checkpoint/recovery, Phase-1, Phase-2 or
Phase-3 terminal receipt exists. The original source transactions remain
immutable. Phase 4 was never opened and official-final is sealed.

Recovery-v4 design is frozen in
`docs/superpowers/specs/2026-07-28-stage0-recovery-v4-evaluator-commit-bridge-design.md`.
It adds only the explicit canonical environment bridge and a precheck that
executes the same environment lookup and Git comparison as the evaluator. It
does not change model or experiment semantics. The regression failed before the
bridge was implemented and passed afterward; compilation, Bash syntax and the
expanded focused suite completed with `97 passed`.

## Recovery v4 deployment

Exact clean implementation commit
`1b44fe3a35042d28c55b9e838f69107bd1461810` passed an independent deployment
audit (`GO`). Remote Slurm code preflight `1200583` completed 194 authoritative
Linux/Torch contract tests. Full launcher/runtime preflight `1200601` also
completed after deliberately injecting a stale canonical evaluator commit before
each salvage precheck; the launcher overwrite-bridge restored and verified the
required exact commit. No production output was written by precheck.

The production identities are:

- physical protocol:
  `2d416cddd923aa46693ad5361979558e845252947fcb50491cd5cc6c6e70be8c`;
- salvage manifest:
  `2fb3f9c1a7623e059f855227c34d7614ef2fb6c9e29ee5461e29b4cf5f107d11`;
- transaction root:
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_1b44fe3a_20260728_221502`;
- submission manifest:
  `ca72b350ccd7227671554e6e413281cd7059c97f5c3161e2ed93c7a087549767`;
- released submission receipt:
  `eca9e24a06ad7ff2a187066f2f255eb4d764a2f3c5362734444463fa6c128449`.

Jobs `1200627`--`1200632` are respectively code gate, Phase 1, ActionFormer
salvage, TriDet salvage, Phase 2 and Phase-3 controller. At the validated
`2026-07-28 22:25 CST` snapshot:

- code gate `1200627` completed with exit `0:0`;
- both salvage jobs `1200629`/`1200630` completed with exit `0:0` and emitted
  their checkpoint evidence plus passing recovery receipts;
- both recovery receipts preserve the source training-job state as `FAILED` and
  restrict claims to
  `engineering_dense_reference_recovery_not_method_evidence`;
- Phase 1 `1200628` is running; Phase 2 `1200631` and the Phase-3 controller
  `1200632` remain protected by dependencies;
- Phase 4 remains disabled and official-final remains sealed.

This proves only that the repaired deployment contract now executes through the
actual formal evaluator and closes both dense recovery arms. It is not a model
performance result.

## Recovery v4 Phase-1 failure

At `2026-07-28 22:32:51 CST`, Phase 1 job `1200628` failed with exit `1:0`
during the first actual exact-uniform K384 forward. The exact terminal exception
was:

`ValueError: dynamic RIME backbone requires an aligned [B,K] mask`.

The traceback identifies the contract gap precisely:

1. the exact-uniform baseline has no `duca_rime_physical` selector;
2. `ActionFormer.forward_test` therefore took its ordinary branch and called
   `self.backbone(inputs)` without `masks`;
3. the same config enables `BackboneWrapper.dynamic_temporal_bucket`, whose
   `_prepare_dynamic_temporal_bucket` requires an aligned `[B,K]` mask;
4. the uniform launcher precheck validated only config/protocol/pretrain fields
   and never exercised model construction plus a tensor forward.

This is not a K384 budget-ledger failure: the emitted engineering ledger bound
the attempted backbone input to K384. It is the detector-to-backbone mask handoff
that failed before the first prediction completed. No Phase-1 terminal receipt
exists. Phase 2 job `1200631` is `DependencyNeverSatisfied`; Phase-3 controller
`1200632` remains dependency-held. Phase 4 remains disabled and official-final
remains sealed.

## Next gate

1. repair the exact-uniform detector/backbone handoff so a dynamic temporal
   bucket receives the already aligned dataset mask even without a physical
   selector;
2. add focused forward tests for aligned, missing, mismatched and inactive-tail
   masks and make launcher precheck exercise the real model-forward contract;
3. repeat independent review and remote runtime precheck on a fresh exact commit
   before deploying another fresh transaction root;
4. require Phase-1, both dense recovery, Phase-2 and Phase-3 development
   receipts before running the held-out same-total-cost H-RIME oracle or learned
   planner training.

Correct empirical statement:

`No paper-admissible empirical conclusion is available yet.`
