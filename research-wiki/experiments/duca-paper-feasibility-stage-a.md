# DUCA paper-feasibility Stage A

## Status

- Decision: `user_approved`
- Design: `designed`
- Short-window corrigendum: `user_approved / U-PRO-STAGEA-SHORT-K-CORRIGENDUM-1`
- Implementation: `corrigendum_implemented_locally`
- Local focused verification: `15_passed / 1_Linux_loader_test_skipped_on_Windows / Torch_tests_blocked_by_local_c10_DLL`
- Independent read-only audit: `selector_tensor_chain_GO / enforced_two_gate_dependency_chain_GO / no_P0_or_P1`
- Authoritative Linux/Slurm verification: `corrected_source_00f54dfe / code_gate_1215368_passed / real_short_gate_1215369_passed`
- Experiment: `recovery1_failed_closed_on_learned_exactk_numeric_invariant / controls_and_seal_cancelled / log_domain_repair_local / rerun_pending / metrics_sealed`
- Empirical support: `not_yet_empirically_supported`
- Paper status: `not_yet_paper_ready`

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

No single cell, seed, intermediate checkpoint or incomplete matrix may support a
performance statement. Until all twelve terminal receipts pass, the status is
only `ENGINEERING_STATUS`.

## Conditional Stage B

Stage B remains blocked until Stage A is repaired, complete and full-200, training-only
out-of-fold per-K utility/risk targets exist. It will add three dynamic
mean-K384 runs and the corresponding evaluation-only exact same-realized-K
uniform replays. H-RIME, TriDet and K192 remain deferred.
