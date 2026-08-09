# DUCA paper-feasibility Stage A

## Status

- Decision: `user_approved`
- Design: `designed`
- Short-window corrigendum: `user_approved / U-PRO-STAGEA-SHORT-K-CORRIGENDUM-1`
- Implementation: `split_negative_control_source_06103e34 / four_gate_release_chain_passed`
- Local focused verification: `py_compile_and_bash_passed / direct_CPU_numeric_oracles_passed / release_contract_17_passed_1_skipped / mandatory_C3_23_passed`
- Independent read-only audit: `minimal_repair_GO / FP64_dtype_caveat_resolved`
- Authoritative Linux/Slurm verification: `exact_source_7e893569 / code_gate_1215388_passed / real_short_gate_1215389_passed`
- Experiment: `stagea_1233471_to_1233476_repeated_runtime_binding_failed_closed / seal_1233477_cancelled / metrics_never_opened`
- Empirical support: `not_yet_empirically_supported`
- Paper status: `not_yet_paper_ready`

## Latest split-negative-control transaction

Exact experiment source `06103e347343ec87969c84d3706c87081d23149d`
passed code gate `1233465` (152 tests), real natural-short gate `1233466`, and
two-rank numeric plus exact-211 release gate `1233467`, all with exit `0:0`.
Their terminal engineering receipts were hash-validated before release.

The frozen Stage-A root is
`/data/run01/sczc063/yuzibo/rime_runs/duca_paper_stage_a_06103e34_20260809_233131`.
Protocol/submission/released-receipt SHA-256 values are respectively
`67fe5bbdc22e4d98f4b4c8fbea8fc56627406d1232e34363c4f2934067f34a90`,
`16ed0c869924a86be9861f68a4247a889fb81852d4bb9976f67da85a82bcec40`
and `5dfacff6b72c9b84d6abd4c9d5f6b75e1cb8e0cca08c092675dece97a9cd15f0`.

Jobs `1233471`--`1233476` all failed `1:0` before creating any cell root.
Every log is byte-identical with SHA-256
`4d4e83310b96792e8cffc0e3667fd9b29f3c93280390239ac446b5d4da0976aa`
and terminates with `runtime annotation binding drift`; seal `1233477` was
cancelled. The exact checkout contains neither `data` nor `data/thumos-14`, so
the signature is the previously registered
`missing_runtime_thumos_relative_bindings`. Its bounded automatic recovery was
already consumed under the `00f54dfe` transaction. This recurrence is terminal
fail-closed pending explicit reauthorization of a source-level pre-release
binding fix. No checkpoint, prediction, evaluation, cell receipt or metric was
created.

## Question

At an exact heavy input of K=384, does jointly learned, task-aware temporal
position selection improve official THUMOS14 localization over exact-uniform
selection under the same ActionFormer, initialization, training exposure,
checkpoint and evaluator contract?

## Frozen Stage-A matrix

Four arms use seeds `5801`, `8123`, and `12011`:

1. dense ActionFormer at T=768;
2. exact-uniform fixed K384;
3. stateless mixed-K training with exposure counts `(8,12,16,24)` over
   requested budgets `(192,256,384,512)`, evaluated with fixed requested K384;
4. DUCA jointly optimized ASFormer evidence with learned fixed K384 positions.

Every cell trains on all 200 `training` videos with two-process DDP, global
batch two, 60 epochs, 100 successful updates per epoch and terminal epoch-59
EMA. Training has no validation loader. Evaluation uses exactly the complete
211-video OpenTAD `validation` set, standard sliding-window merge, NMS and mAP.

The mixed-K arm is a detector-robustness training control. It is not a dynamic
inference method. The DUCA ASFormer frontend is jointly trained from the 200
training videos and is not a frozen external checkpoint; its scan cost belongs
to full-stack DUCA cost.

The approved natural-short-window rule separates `K_req`, `K_eff`, `K_unique`
and `K_backbone`. For valid length `L`,
`K_eff=min(K_req,floor(L/16)*16)` and physical execution must satisfy
`K_backbone=K_unique=K_eff<=K_req`. The schedule and its nominal mean 384 refer
to `K_req`; realized cost is measured and reported separately. No frame
repetition, tail padding, video exclusion or length-conditioned request is
allowed. Fixed requested-K384 evaluation with this common cap remains a fixed
policy, not dynamic inference. A sub-quantum `L<16` window fails closed.

## Implemented evidence chain

- source and resolved-config SHA-256 per arm;
- exact Git commit and clean checkout;
- VideoMAE, annotation and class-map SHA-256;
- exact 200/211 annotation identities;
- two-rank full-train exposure replay for all 60 epochs;
- exactly 6000 optimizer/scheduler/EMA updates;
- terminal EMA compaction and training receipt;
- exact 211 prediction keys plus executed merge/NMS/evaluator receipt;
- transactional held submission for 12 cells and one dependent matrix seal.

The corrected scheduler representation is seven jobs: three per-seed control
jobs sequentially execute dense, fixed-uniform and mixed-uniform cells; three
independent per-seed jobs execute DUCA; one dependent job seals all twelve
logical cells. This grouping satisfies `MaxSubmitJobs=16`, does not share
weights, RNG state, work directories or receipts, and prevents a mixed-control
failure from suppressing the learned DUCA arm.

The exact deployed source is commit
`2df0103ec1c26ff7cff7ed15f399e78e640df211`. Authoritative gate job `1213711`
completed with 37 Linux/PyTorch tests. The production transaction root is
`/data/run01/sczc063/yuzibo/rime_runs/duca_paper_stage_a_2df0103e_20260802_120351`.
Seed jobs `1213712`, `1213713`, and `1213714` contain the twelve independent
logical cells; matrix seal job `1213715` depends on all three. At release, the
seed jobs were scheduler-pending on `AssocGrpGRES` and the seal was dependency-
pending. The first monitored state change placed all three seed jobs in
`RUNNING`, while the seal remained dependency-pending. No metric has been
opened.

## Terminal failure

All three grouped seed jobs failed in the third logical arm,
`uniform_mixed_train_k384_eval`, before the DUCA learned-position arm could run:

- seed 5801, job `1213712`: `FAILED 1:0`;
- seed 8123, job `1213713`: `FAILED 1:0`;
- seed 12011, job `1213714`: `FAILED 1:0`;
- matrix seal `1213715`: dependency-impossible and cancelled by exact ID.

The shared exception is
`ValueError: uniform_mixed_k forbids effective-K shrinkage on a short window`.
Register the unique failure signature as
`paper_full200_uniform_mixed_k_short_window_exact_requested_k_infeasible`.
The immutable failed-log SHA-256 values for seeds 5801, 8123, and 12011 are,
respectively, `9ed49fa701b13c99960c0ef5fa88e597021120fe16bc3d810ad60c6293ff0879`,
`dae2a78d35157b4d6efdc93c31e9f7452789ae69c263b819ac1b3fe404c6e0da`, and
`604aa86707635f00c93de7d8af526fa9b1356e94371ca2eca3a07d66f513217a`.

At the time of failure this was not eligible for bounded automatic repair. The
then-frozen mixed-K design
required every requested K to execute exactly and rejected both short-window
effective-K shrinkage and padding. The paper protocol simultaneously requires
all 200 training videos, which necessarily includes short windows. Allowing
quantum-aligned effective-K aliases would change the registered actual exposure
and mean-heavy-K semantics; excluding short videos or padding would change the
data or no-padding contract. A scientific protocol decision was therefore
required before a fresh transaction. Six earlier dense/uniform cell receipts
exist, but the matrix is incomplete and no metric from them has been opened.

That scientific decision is now supplied by
`U-PRO-STAGEA-SHORT-K-CORRIGENDUM-1`. The repository implementation removes the
contradictory exact-requested guard, preserves all four K meanings, commits
budget rows only after successful optimizer steps, and binds formal accounting
to actual heavy-backbone tensors. A clean-commit Linux test gate and a separate
real-data natural-short-window selector-to-heavy-backbone Slurm gate are required
before a fresh hash-bound seven-job transaction can be released. No artifact of
the old transaction may satisfy either prerequisite or any new matrix cell.

The clean Linux code gate deliberately does not create a matrix manifest. It
runs the focused Torch/contract suite and emits a code-only receipt declaring
the real short-window gate pending. Only after that second Slurm gate passes may
the grouped submitter create the gate-bound manifest and release cells. This
ordering prevents a circular prerequisite and matches the approved sequence.
The short-window runner must consume and validate the exact code-gate receipt;
both prerequisite hashes are then frozen into the manifest and propagated into
training, cell, submission and matrix receipts, so neither gate can be skipped.

The first clean corrected checkout used source
`75b9ba3d2053675ef83902e03dd4ff705c235244`. Code-gate job `1215366` failed
before creating its root because the external Slurm wrapper used POSIX `sh`
with `source`. The explicit-Bash retry `1215367` reached the full suite and
reported `90 passed / 1 failed`: `L=15` correctly failed closed, but the
exception text did not satisfy the focused regex. These are engineering-only
failures with signatures `slurm_wrap_posix_sh_source_not_found` and
`subquantum_failclosed_exception_message_contract_mismatch`; neither produced a
passing receipt, manifest, model run, or metric. The second repair is restricted
to the exception wording and requires a new commit, clean checkout, full code
gate and real short-window gate before any Stage-A release.

That repair is source `00f54dfecb6a536224958b1cd64d2daa5b8ca982`.
Code gate `1215368` and real short-window gate `1215369` completed `0:0`; their
receipt SHA-256 values are
`5cb16630ea07f38db6dc9a14d9bcd18efa2a0c9ab8f408dd0c9fde3610c26185` and
`2eec808e36d9eb92a8f22eee67d5a00588e4f114e6b27dbaca9a56a495b29d89`.
The real gate covered all 200 training videos, selected a real natural short
sample, proved zero `L<16` samples, and completed all four heavy-backbone paths
without padding/repetition. It is not method-performance evidence.

The first corrected release at root
`/data/run01/sczc063/yuzibo/rime_runs/duca_paper_stage_a_00f54dfe_20260803_023009`
failed before training because the transport-clean checkout omitted ignored
runtime `data/thumos-14` links. Jobs `1215370`–`1215375` share signature
`missing_runtime_thumos_relative_bindings`; seal `1215376` was cancelled. After
hash-verifying and restoring exactly the formal runtime bindings, all four config
prechecks passed while Git remained clean.

The unique active recovery root is
`/data/run01/sczc063/yuzibo/rime_runs/duca_paper_stage_a_00f54dfe_retry1_20260803_023358`.
Protocol manifest SHA-256 is
`b4baa6b60954c00dc906740d801a170cc079021192cc2c7c2c81f7f5bc209366`;
submission manifest SHA-256 is
`8def0bce9a0447b9a3d25f6a171452ba9e15ad2f5fdf937571052d2915f3e19e`;
released receipt SHA-256 is
`c49d8f3f1b017ec11ef7ad1ca3c246e2798fb7892af183544809421bb658c97c`.
Control/DUCA jobs are `1215377/1215378` (5801), `1215379/1215380` (8123),
and `1215381/1215382` (12011); dependent seal is `1215383`. This identity was
released as `experiment_running` only; no metric was authorized or opened.

This recovery is now terminally failed closed. Learned-DUCA jobs
`1215378/1215380/1215382` all stopped in epoch 0 on
`RuntimeError: physical exact-K slot marginals do not sum to one`; register
`physical_exactk_long_chain_fp32_slot_mass_loss`. Their log SHA-256 values are
`5371743766d85d7df461682e9b498ffbcd25c332b6021fd50a646e6f234b4b1b`,
`7db05504b28713b0d8a19ffe840d042de7d0af2b36da7ebb1502965b46cddad2`, and
`5d688a5b2171f6a4e24d66c428ff7db60c9016f7c47d0803501cf1d1b429a780`.
The three isolated controls were still running but were cancelled because a new
selector source commit makes them ineligible for the replacement matrix; seal
`1215383` was cancelled. No partial metric was inspected.

Read-only mathematical/code/failure audits and GPU scale diagnostic `1215387`
showed that graph reachability and partition were valid, while long FP32
forward/backward chains lost slot-wise mass after direct exponentiation. The
same `T=768,K=384` graph failed at score scales 16/32/64. Per-slot log-domain
normalization passed with finite gradients and roughly `2.38e-7` maximum row
error; FP64 also passed but is rejected as the paper path because it changes
selector cost. The diagnostic log SHA-256 is
`c2200fc76264e1d3d42d89bf6e5b2ac1fee305751cf84adc9ba217714e57ef9b`.

The implemented narrow repair performs the categorical slot normalization in
log space before exponentiation. It first rejects pre-normalization log-mass
drift outside a conservative FP32 accumulation envelope, preventing the
projection from hiding graph/recurrence faults. It leaves the legal graph, log
partition, Viterbi hard path, K, loss and architecture unchanged and retains
fail-closed column-occupancy and ordering checks. New regressions verify small-graph
brute-force Gibbs equivalence plus long-chain high-dynamic-range marginals and
finite backward gradients. The formal code gate now includes the physical
structured-selection test module. A fresh commit, Linux/PyTorch gate, real
short-window gate, runtime path precheck, root and seven-job matrix are required;
no artifact from either `00f54dfe` root can be reused.

The exact repair is source
`7e8935692b732f2958ba3c20787ae19c86f7b15c`. Authoritative code gate
`1215388` passed 140 Linux/PyTorch tests and emitted receipt SHA-256
`92b31ab9a9d33580c3b27a367a3866d5ffdf5cf2f9e34a186384ef42e6e0bbf0`.
Real short-window gate `1215389` passed with receipt SHA-256
`904dacf22d6d88cba8206367c6b824da002f717e7227ad1f310afa08a4a89b16`.

The fresh formal transaction root is
`/data/run01/sczc063/yuzibo/rime_runs/duca_paper_stage_a_7e893569_20260803_030306`.
Protocol, submission and released-receipt SHA-256 values are
`33903a7eee1089202295161368edf4e5e5b3470c88b072c9fa00f82f15fefaf3`,
`1f4b14569f45977d6b606febe7c42898d401258ee5f73b918b56d197d65bb105`, and
`cea74e4bd231e8687e583f2fa98ff5dacec3c41e2ec5352396e097e7abfb01f7`.
Jobs `1215390/1215391`, `1215392/1215393`, and `1215394/1215395` are the
control/DUCA seed pairs; seal `1215396` is exact afterok all six. All six groups
entered `RUNNING`, but the transaction is now terminally failed closed.

Control jobs `1215390/1215392/1215394` each produced a dense terminal cell and a
fixed-uniform training receipt, then failed during fixed-uniform evaluation
budget validation. The exact violated predicate is duplicate key identity:
ledger rows 721 and 722 are byte-identical and both use
`(video_test_0001431, window_start_frame=7680)`. Their canonical row SHA-256 is
`400d197499d056b2874aa1646cccd56cbc83378ea24e2b76e3bf47a75d2b2fb6`.
All requested/effective/unique/backbone/padded-K and selected-index predicates on
that row pass. Register `paper_exact_uniform_eval_duplicate_window_ledger_key`;
the cause still has to be separated into duplicate dataset sample versus
duplicate ledger emission before a protocol-neutral repair can be designed.

Learned-DUCA jobs `1215391/1215393/1215395` failed on raw physical exact-K slot
mass outside the registered FP32 normalization envelope. This is classified as a
repeat of `physical_exactk_long_chain_fp32_slot_mass_loss`, whose one bounded
recovery has already been consumed. It is not eligible for automatic threshold
relaxation, invariant suppression, FP64 substitution or redeployment. Exact seal
`1215396` was cancelled after its dependencies became impossible. Mixed-K never
started, no learned-DUCA training receipt exists, and only three of twelve cell
receipts exist.

No single cell, seed, intermediate checkpoint or incomplete matrix may support a
performance statement. No metric was opened from this transaction. Its status is
terminal `ENGINEERING_STATUS`; it yields no official mAP result and no comparison
with AdaTAD mAP=65.

## Minimal solver repair decision and implementation

`U-PRO-STAGEA-MINIMAL-SOLVER-REPAIR-1` adjudicated the terminal state as
`GO_MINIMAL_SOLVER_REPAIR`. Repository-level cross-checks accepted both root
causes:

- the control duplicate is deterministic dataset enumeration. For
  `N=2688,W=768,S=384`, the old loop emitted terminal start `1920` regularly and
  then back-shifted the next overflow to `1920` again;
- learned-DUCA narrows an internally FP32 coverage log distribution back to the
  AMP scorer dtype and then accumulates an unnormalized long FP32
  forward/backward chain. The raw mass guard depends on message gauge and is not
  a valid structural invariant.

The implemented minimal repair enumerates unique canonical sliding starts at
the dataset source; it does not post-hoc deduplicate ledgers or change the
physical window key. AMP/FP32 coverage distributions now remain FP32, while an
explicit FP64 input remains FP64 for oracle diagnostics. Physical exact-K
forward/backward subtracts one global exact-K gauge, normalizes every slot
message, carries the corresponding alpha/beta additive scales, restores
`logZ + K*gauge`, and verifies independently reconstructed forward/backward
partitions. Marginals retain row, column, ordering, finite-gradient and hard-path
checks. This changes no graph, hard decoder, architecture, loss, budget,
hyperparameter, data split, seed, checkpoint or evaluator.

The external proposal for a generic persistent pre-backbone execution journal
is not part of this repair: the dataset root cause is closed, current per-rank
ledgers already commit only after successful backbone execution, and the
finalizer still rejects duplicate `(video_id,window_start_frame)` identities.
Likewise, proposed numeric thresholds were not treated as established facts.
The accepted tests are structural: exact terminal enumeration, small-graph
brute-force marginals/logZ/gradients, additive-gauge invariance, FP32 versus FP64
long-chain oracle, AMP dtype preservation, and the production-shaped
`T=768,K=384` finite backward stress.

Local compilation and direct CPU numerical checks pass, including the full
`T=768,K=384` stress; the normal pytest modules remain skipped on Windows by the
repository's c10.dll guard. Status is `implemented / local_direct_tested /
authoritative_Linux_gate_pending`, not `tested` on the production environment.
The exact implementation commit is
`cb077a77d48d9776028fa4d88fcf5b3ca1d9e357`.
No Stage-A redeployment is authorized until a clean commit, the full Linux code
gate, the real short-window heavy-backbone gate, exact-211 identity dry-run and
fresh hash-bound transaction prerequisites all pass. Old roots, receipts and
partial cells remain immutable and ineligible.

The release prerequisite is now strengthened to four exact-commit/hash-bound
gates: clean Linux/PyTorch code, real natural-short-window heavy execution,
production-like learned `T=768,K=384` DDP/AMP numerical stability, and exact-211
metadata/physical-window identity. The numeric gate uses a real full-model
forward/backward/optimizer path, reproduces the superseded raw-message failure
within a bounded 100-update window, and evaluates the repaired solver against
the frozen FP64/flow/gradient/hard-path predicates. The exact-211 gate enumerates
official validation metadata only and checks every `(video,start)` and physical
UID exactly once; it performs no decode, model call, prediction or metric access.
All four receipts are now required by the manifest, cell, submission and seal
paths. This is `implemented / local_contract_tested / authoritative_Slurm_pending`;
it is not an mAP result and does not authorize Stage B.

## Conditional Stage B

Stage B remains blocked until Stage A is repaired, complete and full-200, training-only
out-of-fold per-K utility/risk targets exist. It will add three dynamic
mean-K384 runs and the corresponding evaluation-only exact same-realized-K
uniform replays. H-RIME, TriDet and K192 remain deferred.
