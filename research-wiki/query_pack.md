# DUCA-RIME Current Query Pack

Last updated: `2026-08-04`

## Current decision

The user supplied and approved the final narrow repair
`U-PRO-V21-FINAL-REPAIR-1`; its candidate-free statistical core remains
`implemented / focused-tested`. On `2026-08-02`, the user explicitly
reprioritized the project: do not build the missing full-simulation runner, do
not execute the 52-by-500 or 24-by-200 MC grids, and do not treat Admission
engineering as a prerequisite for testing DUCA's model value. The simulation
code and its unexecuted design are preserved as non-authorizing historical
provenance. Production Admission remains `NO_GO`, but it no longer blocks the
paper-facing DUCA feasibility route. The current first priority is an official
full-data experiment that can determine whether DUCA's selected-axis acquisition
improves the accuracy--cost frontier against matched controls. The paper
mainline remains an
**offline TAD pure selected-axis pre-backbone acquisition plugin**.
Physical-time head injection is a separately named integration/diagnostic
route and cannot support the pure-plugin claim. This is not Online TAD and is
not yet the paper's final empirically supported method.

Current evidence level:

| Item | State |
|---|---|
| Scientific route | `user_approved` |
| Pro architecture adjudication | `U-PRO-PURE-PLUGIN-1 + U-PRO-ADMISSION-V21-1 + U-PRO-V21-FINAL-1 + U-PRO-V21-CORRIGENDUM-1 + U-PRO-V21-FINAL-REPAIR-1 / final_repair_core_accepted` |
| Pure selected-axis coordinate refactor | `implemented / authoritative_Slurm_code_gate_passed_1204067` |
| Acquisition admission-v2 | `formal_path_disabled / historical_read_only_or_engineering_fixture_only` |
| Acquisition admission-v2.1 | `historical_nonblocking / core_implemented / full_simulation_cancelled_by_user_priority / production_NO_GO` |
| Official full-data DUCA feasibility | `exactk_repair_source_7e893569_double_gate_passed / stagea_1215390_to_1215396_terminal_failed_closed / metrics_never_opened / no_empirical_conclusion` |
| V2.1 data feasibility | `failed_on_immutable_training_metadata / 100_pool_70_full_only_30_short_only_0_both` |
| Phase-1 v2 closure | `not_authorized / requires_verified_admission_v2_1` |
| Four-stage implementation | `implemented` |
| Focused local checks | `49_passed_2_POSIX_skipped_on_Windows / broad_nonTorch_247_passed_5_skipped / Ubuntu_POSIX_runtime-policy_smoke_passed` |
| Remote authoritative code gate | `acquisition_v2_job_1204067_completed_0_0 / receipt_sha256_d664a619007f1cafbd4e52f2fd6a053fb0e3b5336dcb2be2b16302912286e5c8` |
| Dense reference training | `recovery_v6_salvage_completed / engineering_only` |
| Phase 1 closure | `recovery_v6_contract_gate_failed_1201417 / universal_loss_equivalence_premise_invalid / observed_mismatch_component_unresolved / no_terminal_receipt` |
| Phase 2/3/4 | `recovery_v6_children_cancelled / phase4_never_opened` |
| Phase-4 official comparability | `blocked / current trainer uses 100-of-200 development role / entry hard-disabled pending full-train refit` |
| Latest four-stage transaction | `recovery_v6_failed_closed / terminal` |
| H-RIME scientific route | `user_approved / designed / deferred_until_window_local_DUCA_feasibility` |
| Stage-0 repair implementation | `recovery_v6_exact_commit_5a599e90 / authoritative_preflight_passed` |
| H-RIME deterministic core | `implemented / local_non_torch_tested` |
| H-RIME Stage-1 oracle/evaluation surface | `implemented / local_non_torch_tested / remote_torch_tested` |
| H-RIME shared-scan/model integration | `not_yet_implemented` |
| H-RIME same-total-cost oracle | `not_yet_run` |
| H-RIME Stage-0 recovery transaction | `recovery_v6_failed_closed / diagnostic_remediation_local_tested / scientific_contract_review_required` |
| Paper evidence contract | `user_frozen` |
| DUCA-RIME empirical superiority | `not_yet_empirically_supported` |
| Paper-ready method | `not_yet_paper_ready` |

The first acquisition-v2 preflight used exact source commit
`70cf49de82a9d0ed889ed94af9604edd61070e55`, clean remote checkout
`/data/run01/sczc063/yuzibo/OpenTAD_DUCA_ACQUISITION_70cf49de`, and Slurm job
`1204048`. It failed before any calibration or experiment because
`tests/test_duca_rime_training_contract.py` retained the superseded
`duca_protected_physical_v1` expectation for the Phase-1 uniform config.
Failure signature:
`phase1_uniform_test_expected_superseded_physical_protocol`.
The correction changes only the test contract and does not change the model,
loss, budget, threshold, data, checkpoint, evaluator, or scientific gate.
The failed root is immutable and has no gate receipt:
`/data/run01/sczc063/yuzibo/rime_preflight/duca_acquisition_v2_70cf49de_20260729_202923`.

Remediation commit `119db2f83756281729506632a18bfed607794d13`
subsequently passed Slurm code-gate job `1204067` with exit `0:0`; the
content-bound gate receipt SHA-256 is
`d664a619007f1cafbd4e52f2fd6a053fb0e3b5336dcb2be2b16302912286e5c8`.
The workflow nevertheless stops before numeric calibration because the current
producer uses synthetic head features while labeling rows
`train_only_calibration`. It does not consume registered role-scoped videos,
form video/process-grouped null distributions, launch independent repeated
processes, or freeze the prescribed normalized statistic and denominator
floor. Remote inventory also found no admissible non-fixture training-target
JSONL, acquisition data manifest, or NI-margin source. This is a P0 protocol
implementation gap, not model-performance evidence.

External audit `U-PRO-ADMISSION-V21-1` confirmed those six P0s and added six
more: cloned-head tautology, no full-model train/backward path, no immutable raw
distribution, self-declared schemas, unverifiable development non-access, and
no immutable failure receipt. Its architecture-preserving, real-video v2.1
direction is accepted, but four frozen details are not.

The exact split-manifest SHA-256 is
`41349cd39a6a550b6e1613de968577b1605c93902edd52a88309121b9e90c057`.
Its `detector_selector_train` pool has 100 videos, so three disjoint 32-video
roles are count-feasible. The production sliding enumerator nevertheless
back-shifts a long video's terminal window to 768. Immutable annotation
metadata therefore yields 70 full-only videos, 30 short-only videos and zero
videos with both; short-bin counts are 7/13/10 for `1-256`, `257-512` and
`513-767`. The proposed requirement that every video contain both a natural
full and short window is impossible, and the first fixed short bin also misses
the proposed minimum of eight. No synthetic crop/padding workaround is
permitted.

The Stage-0 response is implemented: old v2 formal calibration/admission fails
closed, its random-head path is engineering-fixture-only, old v2 receipts are
historical-read-only and cannot authorize production, and a metadata-only
v2.1 feasibility auditor emits typed content-bound failure. Full v2.1 work
remains blocked until the role-level window contract, exact crossed bootstrap
and catastrophic bound, scientifically justified NI margin, and enforceable
versus observational isolation claims are frozen.

The number 100 has a strictly limited meaning: it is the
`detector_selector_train` development role, not the official training split.
The registered THUMOS training set contains 200 videos. Read-only code audit
found that the current Phase-4 pipeline exports the 100-video role block list
to both RIME and matched controls. Its nominal `official60` schedule matches 60
epochs and 6000 optimizer updates only by using batch size one on 100 videos;
it is therefore not a full-data OpenTAD/AdaTAD training recipe.

Paper-facing training must instead refit every candidate and trainable control
on all 200 videos after the method is frozen. To match the upstream AdaTAD
effective batch and update schedule, use global batch two, 60 epochs, 100
optimizer updates per epoch and 6000 total updates. The selected method also
requires out-of-fold utility/risk targets covering all 200 training videos.
The current Phase-4 entrypoint is hard-disabled until that hash-bound refit
contract exists.

The evaluation side is different: the Phase-4 evaluator already requires the
complete registered `validation` key set with no missing or extra predictions.
OpenTAD deliberately removes two malformed/empty THUMOS test videos and uses
the remaining 211 for fair comparison. Thus 211 is the complete
OpenTAD-comparable evaluation set, not an arbitrary test subset.

The fail-closed correction and official-comparability guard were published on
`codex/duca-rime-20260727` as commit
`505cc4e48b2511a13ad4936c295dfcb084d8d7fd`. Local focused checks passed, but
the exact-commit Linux/PyTorch Slurm code gate remains pending.

The bounded Pro response `U-PRO-V21-FINAL-1` returned
`GO_IMPLEMENT_V2_1`, but independent mathematical and repository audit gives
`CONDITIONAL_ACCEPT_WITH_CORRIGENDUM`, not verbatim acceptance. The 32/32/32
role-level coverage, `NO-GO_FOR_NI` Admission policy, full-200 five-by-forty
OOF refit, two-GPU DDP global batch two, exact-211 transaction, H-RIME stage
order and AdapTok naming are frozen. Three statistical details remain unsafe
to encode: sparse multinomial `D_v=0` handling changes the equal-video
estimand without a coverage proof; the fixed-scale maxT is not strictly
studentized and binary two-stream agreement is not a calibrated Monte Carlo
criterion; and the catastrophic holdout-max/calibration-max ratio depends on
sample counts and can be unstable. The additional mandatory administrator
network/mount attestation also needs an explicit threat model and N16R4
feasibility decision.

That audit requested a narrow statistical-protocol corrigendum, not another
architecture/innovation round. Its input contract is retained in
`docs/superpowers/plans/2026-07-30-duca-v2-1-pro-response-adjudication.md` as
the provenance for the returned response below.

The returned final repair `U-PRO-V21-FINAL-REPAIR-1` resolves those five P0s and
is accepted for protocol implementation. Canonical ranks no longer depend on
input order; triplets are allocation-only while video remains the statistical
unit; `kappa=1`; the exact 52-scenario non-vacuity/power and 24-scenario MC
calibration registries are closed; and runtime receipts now require closed
evidence, independent verifiers and hard authorization invariants. The complete
record is
`docs/superpowers/plans/2026-08-01-duca-admission-v2-1-final-repair-implementation.md`.

This closes implementation blocker 2 only. Complete 52-by-500 execution,
24-by-200 independent MC calibration and authoritative exact-clean Linux/Slurm
runtime receipts remain outstanding. Real-video production workers, scale-fit,
calibration, holdout opening, Phase 1+, learned H-RIME, full-200 refit and
official-final remain unauthorized. No paper-admissible empirical result exists.

The next execution audit is recorded in
`docs/superpowers/specs/2026-08-02-duca-admission-v2-1-full-simulation-execution-design.md`.
It found a new P0 boundary between a frozen statistical registry and a runnable
distributed experiment. `run_simulation_outer` has a caller-selected default
stream zero, regenerates the same deterministic streams for five shift profiles,
and has no full task/receipt runner. The MC gate has evaluators but no producer
for the 4,800 independent operational streams or the two independent 2M halves
and their true concatenated 4M reference. Those choices affect random dependence
or binary64 results and must not be invented in a launcher.

Read-only N16R4 capability inquiry reported `MaxArraySize=1001`, account
`MaxSubmitJobs=16`, a GPU-only partition, and a default eight CPUs per requested
GPU. The registered OpenTAD environment is Python 3.10.20/NumPy 1.23.5, while
the frozen simulation runtime is Python 3.11.7/NumPy 1.23.5/Philox golden. A
one-task-per-outer DAG and the existing environment are therefore invalid.
These are `ENGINEERING_STATUS` facts, not model or paper performance evidence.

## Paper responsibility

This section is a `user_frozen` reporting and claim contract. All research,
implementation, monitoring, analysis, and writing must be accountable to the
final paper. A real number is not scientific evidence merely because a run
produced it.

Every future statement must belong to exactly one class:

1. `ENGINEERING_STATUS`: prechecks, smoke tests, running epochs, small or
   training-domain subsets, ledger failures, single-seed pilots, and incomplete
   matrices. These may diagnose execution, but they must not explain model
   performance, support a method claim, compare with an official number, or
   enter the paper.
2. `THEORETICAL_ANALYSIS`: states assumptions, objective, derivation or
   proposition, falsifiable implications, and limitations. It may not turn an
   unverified assumption into an empirical claim.
3. `COMPLETE_DEVELOPMENT_EXPERIMENT`: the entire pre-registered held-out
   development matrix, matched baselines, terminal checkpoint rule, registered
   seeds, paired statistics, full-stack cost, and immutable provenance are
   complete. It may select or kill a route but is not automatically a paper
   performance result.
4. `PAPER_ADMISSIBLE_RESULT`: the official dataset/split/evaluator, proof of
   training exclusion, matched training and inference contracts, strongest
   registered baselines, all seeds/backends/budget panels, paired statistics,
   full-stack cost, and hash-bound receipts are complete.

A paper-admissible comparison additionally requires:

- a precise hypothesis and experimental unit;
- identical detector, initialization policy, data exposure, successful update
  count, effective batch, terminal checkpoint rule, and post-processing;
- fixed, exact same-cost, direct-transfer, and causal baselines;
- no threshold changes, intermediate-checkpoint selection, extra training, or
  test replacement after observing results;
- all registered accuracy, high-IoU, short-action, transfer, and cost gates.

Internal diagnostics, intermediate epochs, convenient subsets, possibly seen
videos, unmatched checkpoints, partial matrices, single seeds, proxy metrics,
and missing receipts are prohibited substitutes for a complete experiment.

When complete comparable evidence does not exist, the required statement is
`No paper-admissible empirical conclusion is available yet`. The only
alternative is a self-contained theoretical analysis with explicit assumptions
and limits.

The active DUCA-RIME transaction currently has no `PAPER_ADMISSIBLE_RESULT`.
Phase-1 20-video training-domain mAP and running dense-reference epochs are
withdrawn from all performance explanations and retained only as
`ENGINEERING_STATUS`. Phase 3 is development evidence only if its entire frozen
arm matrix passes; Phase 4 is paper-facing only if all 12 cells, statistics,
cost evidence, and provenance receipts are complete.

## Active transaction monitor

At `2026-07-28 14:47 CST`, immutable transaction `d9d454cd` had reached its
terminal fail-closed state:

- code gate `1198113`: `COMPLETED`, 158 tests passed;
- Phase 1 `1198114`: `FAILED` after 12m44s because uniform-K384 ledger
  line 64 recorded a short window with `dense_valid_len=231`,
  `effective_k=unique_k=231`, but `backbone_input_k=padded_k=384`;
- dense ActionFormer `1198115`: `FAILED` after the 60-epoch training loop
  printed `Training Over`; its raw `epoch_59.pth` exists, but
  `compact_duca_rime_checkpoint.py` failed with
  `ModuleNotFoundError: No module named 'tools'`;
- dense TriDet `1198116`: the same terminal failure after its 60-epoch
  training loop; its raw `epoch_59.pth` exists;
- neither dense arm produced `terminal_ema.pth`, `training_receipt.json`,
  evaluation evidence, or checkpoint binding;
- Phase 2 `1198117`: `DependencyNeverSatisfied`;
- Phase-3/4 controller `1198118`: dependency-pending and unauthorized.

No registered job or artifact is literally named `4B`; in the current
conversation, “previous 4B experiment” is interpreted as this latest
four-stage transaction. Its Phase-1, Phase-2, Phase-3, and Phase-4 terminal
receipts are absent, so it did not complete successfully.

The uniform-K384 localization JSON exists on the 20-video development subset,
but its evaluation receipt is absent, so it is not a passing Phase-1 artifact.
The failure is a valid cost-contract stop, not evidence about model accuracy.

The first repaired Stage-0 recovery transaction is terminally failed closed:

- exact deployment commit:
  `902168a12bc92babd62b6cb1877ce7137f56cea0`;
- root:
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_902168a1_20260728_183709`;
- submission-manifest SHA-256:
  `fd6fef65ac01e7830c6b5e337684b19a3bad65c1432f819cfecb32e83dfefb85`;
- jobs: code gate `1199978`, Phase 1 `1199979`, dense ActionFormer salvage
  `1199980`, dense TriDet salvage `1199981`, Phase 2 `1199982`, and Phase-3
  controller `1199983`;
- terminal scheduler state: code gate `1199978` passed; Phase 1 `1199979`
  failed because the dense evaluator retained the repository-relative VideoMAE
  initialization path instead of the already hash-checked absolute path;
  ActionFormer/TriDet salvage `1199980`/`1199981` failed before Python because
  the submit wrapper directly executed a tracked non-executable script;
  Phase 2/controller `1199982`/`1199983` became
  `DependencyNeverSatisfied` and were canceled by exact ID;
- only the code-gate receipt and released submission receipt exist; no Phase-1,
  dense-recovery, Phase-2 or Phase-3 terminal receipt exists;
- dense recovery claim scope:
  `engineering_dense_reference_recovery_not_method_evidence`;
- Phase 4 disabled; official-final sealed.

The repair binds and hash-checks the absolute VideoMAE initialization in the
actual `tools/test.py` command and invokes salvage through explicit Bash while
also restoring its executable bit. Exact commit
`0ab242f31be8de7b7da806b645d3aa60d02d8d88` passed local tests, 82 remote
Linux/Torch tests, three Phase-1 launcher prechecks, both dense-salvage
prechecks, and an independent clean-commit audit.

The second repaired Stage-0 recovery transaction is terminally failed closed:

- root:
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_0ab242f3_20260728_201613`;
- physical-protocol SHA-256:
  `2f11c12d62451c7ec41b54ac889058617f56f889e6f289cfe865a47eb03ff9f9`;
- salvage-manifest SHA-256:
  `faab636144d0855f2d8f26d6c7298459302b3c84508bdc2da24b1b864013772d`;
- submission-manifest SHA-256:
  `b996543dfe57bc3678799591f38f0e96e76da971eb8d5a4f7a4edbb15aa3d04d`;
- jobs: code gate `1200135`, Phase 1 `1200136`, ActionFormer salvage
  `1200137`, TriDet salvage `1200138`, Phase 2 `1200139`, and Phase-3
  controller `1200140`;
- terminal scheduler state: code gate `1200135` completed; Phase 1 `1200136`
  failed; ActionFormer/TriDet salvage `1200137`/`1200138` failed; Phase 2 and
  the Phase-3 controller `1200139`/`1200140` became
  `DependencyNeverSatisfied` and were canceled by exact ID;
- Phase 1 completed its registered dense development controls but the
  exact-uniform launcher failed at actual inference because it did not override
  the base config's repository-relative VideoMAE initialization path; its
  precheck had not exercised that runtime binding;
- both salvage arms created raw compacted EMA checkpoints and salvage sidecars,
  then failed while finalizing structured evaluation evidence because
  `tools/test.py` expected `validation` for an engineering salvage role whose
  frozen evaluator subset was `training`;
- no dense `checkpoint_evidence.json`, Phase-1 `pipeline_receipt.json`,
  Phase-2 `pipeline_receipt.json`, or Phase-3 terminal receipt exists;
- recovery remains engineering-only, original jobs remain `FAILED`, Phase 4 is
  disabled, and official-final is sealed.

This status is `ENGINEERING_STATUS`, not an empirical result.

The user approved direct execution of the compact paper-feasibility route.
Exact source commit `2df0103ec1c26ff7cff7ed15f399e78e640df211` passed Slurm gate
`1213711`, but the production transaction is now terminally failed closed.
Seed jobs `1213712/1213713/1213714` all failed with exit `1:0` in the mixed-K
training arm on the identical exception `uniform_mixed_k forbids effective-K
shrinkage on a short window`; seal `1213715` was cancelled after its dependency
became impossible. The frozen old mixed-K contract requires exact requested-K
execution and forbids both shrinkage and padding, while paper-facing Stage A
requires all 200 videos and therefore natural short windows. Every available
resolution changes actual budget semantics, data inclusion, or no-padding
execution, so this is a scientific protocol inconsistency rather than an
automatically repairable engineering defect. Six earlier dense/uniform cell
receipts remain immutable but partial and unopened. DUCA learned fixed-K384 did
not run; no complete matrix, empirical comparison, or paper-admissible result
exists. Stage B remains blocked.

On `2026-08-03`, the user supplied final short-window adjudication
`U-PRO-STAGEA-SHORT-K-CORRIGENDUM-1` with verdict `IMPLEMENT`. It is accepted.
The `(8,12,16,24)` cycle is now explicitly the requested-K exposure, with
nominal requested mean 384. Natural windows execute the deterministic,
label-free quantum cap `K_eff=min(K_req,floor(L/16)*16)` while preserving
`K_req` as a separate ledger field. The physical invariant is
`K_backbone=K_unique=K_eff<=K_req`; padding, repetition, video exclusion and
length-conditioned requests are prohibited. Because the repository's VideoMAE
path requires positive q=16 buckets, any `L<16` window fails closed and the real
full-200 preflight must prove that none exists.

The correction is implemented locally with retry-safe per-successful-update
accounting and actual BackboneWrapper/inner-VideoMAE tensor evidence. A new
Slurm prerequisite runs the real decoder, selector, physical gather and heavy
backbone for requests 192/256/384/512 including a natural short window; its
immutable receipt is required by the Stage-A manifest. The fresh matrix uses
three sequential control jobs, three independent DUCA jobs and one seal, so a
mixed-control failure cannot suppress the learned DUCA arm and the seven-job DAG
fits `MaxSubmitJobs=16`. This state is `implemented`, not yet authoritative
Linux-tested, experiment-running, empirically supported, or paper-ready. No old
metric or partial receipt has been opened or reused; Stage B remains sealed.
Local syntax/Shell/contract verification is `15 passed / 1 Linux-only skip`;
the local Windows Torch suite is unavailable because `c10.dll` fails to load,
so no Torch-tested status is claimed before Slurm. Independent read-only review
accepted the selector-to-backbone accounting and the scoped real-data gate. A
final deployment-order audit found and repaired one P0 circularity: the clean
Linux code gate no longer attempts to build the gate-dependent matrix manifest.
The enforced order is now clean code gate → real short-window gate → new
manifest/root → seven-job Stage A. The short-window gate directly consumes the
content-bound code-gate receipt, and both hashes are required by the grouped
submitter, every cell and the matrix seal; this is an enforced dependency rather
than an operator convention.

The first corrected remote source was clean commit
`75b9ba3d2053675ef83902e03dd4ff705c235244`. Its initial code-gate submission
`1215366` failed before creating a gate root because Slurm `--wrap` entered
POSIX `sh` and rejected `source`; signature
`slurm_wrap_posix_sh_source_not_found`. A one-time launcher-only retry used
explicit `/bin/bash -lc` and reached the full suite, where job `1215367` failed
with `90 passed / 1 failed`: the q=16 sub-quantum path correctly raised
`ValueError`, but its message did not match the frozen focused-test wording.
Signature: `subquantum_failclosed_exception_message_contract_mismatch`. No model
experiment, performance metric, manifest, or passing receipt was produced. The
repair changes only the fail-closed exception text; it does not change the
model, loss, K rule, data, seed, checkpoint, evaluator, or scientific question.

The narrow repair is exact source
`00f54dfecb6a536224958b1cd64d2daa5b8ca982`. Slurm code gate `1215368`
completed `0:0` with 91 Linux/PyTorch tests; its receipt SHA-256 is
`5cb16630ea07f38db6dc9a14d9bcd18efa2a0c9ab8f408dd0c9fde3610c26185`.
The dependent real-data heavy-backbone gate `1215369` completed `0:0`; receipt
SHA-256 `2eec808e36d9eb92a8f22eee67d5a00588e4f114e6b27dbaca9a56a495b29d89`.
It enumerated all 200 training videos, found 43 natural short samples and zero
sub-quantum samples, then completed real selector → unique gather → VideoMAE
execution for all four requested budgets with no padding or repetition. This is
engineering evidence only and consumed no validation/test data or metric.

The first seven-job release under this source used root
`/data/run01/sczc063/yuzibo/rime_runs/duca_paper_stage_a_00f54dfe_20260803_023009`.
Jobs `1215370`–`1215375` all failed before training on the same
`runtime annotation binding drift`; seal `1215376` was cancelled. The clean
transport checkout lacked OpenTAD's ignored `data/thumos-14` runtime symlinks.
Register `missing_runtime_thumos_relative_bindings`; no cell root, checkpoint,
evaluation, or metric was produced. The one-time protocol-neutral recovery
restored the same symlink targets used by the earlier formal checkout only after
their annotation/class-map hashes matched the registered assets, verified all
four runtime configs, and preserved a clean exact-commit Git status.

The recovery transaction was
`/data/run01/sczc063/yuzibo/rime_runs/duca_paper_stage_a_00f54dfe_retry1_20260803_023358`.
Protocol manifest SHA-256 is
`b4baa6b60954c00dc906740d801a170cc079021192cc2c7c2c81f7f5bc209366`,
submission manifest SHA-256 is
`8def0bce9a0447b9a3d25f6a171452ba9e15ad2f5fdf937571052d2915f3e19e`,
and released-receipt SHA-256 is
`c49d8f3f1b017ec11ef7ad1ca3c246e2798fb7892af183544809421bb658c97c`.
Control/DUCA job pairs are `1215377/1215378` (5801), `1215379/1215380`
(8123), and `1215381/1215382` (12011); seal is `1215383`. All three DUCA jobs
failed in epoch 0 at `_physical_row_forward_backward` with the exact invariant
error `physical exact-K slot marginals do not sum to one`. Their immutable log
SHA-256 values are `5371743766d85d7df461682e9b498ffbcd25c332b6021fd50a646e6f234b4b1b`,
`7db05504b28713b0d8a19ffe840d042de7d0af2b36da7ebb1502965b46cddad2`,
and `5d688a5b2171f6a4e24d66c428ff7db60c9016f7c47d0803501cf1d1b429a780`.
Because a new selector fix requires a new commit, the three running control jobs
and seal were cancelled by exact ID; no partial metric was opened. Register
failure signature `physical_exactk_long_chain_fp32_slot_mass_loss`.

Independent code, mathematical and failure-identity audits agree that legal
path reachability and finite partition checks had already passed. The production
implementation then accumulated a long exact-K forward/backward chain in FP32
and directly exponentiated `alpha+beta-logZ`; tests covered physical selection
only at `T=6,K=3`, and the short-window gate used a no-gradient single-sample
path. GPU diagnostic `1215387` deterministically reproduced the same failure at
`T=768,K=384` for score scales 16/32/64. On identical graphs and inputs,
per-slot log-domain normalization retained finite gradients and reduced maximum
row-mass error to about `2.38e-7`; FP64 also worked but would distort the
paper-facing full-stack cost. Diagnostic log SHA-256 is
`c2200fc76264e1d3d42d89bf6e5b2ac1fee305751cf84adc9ba217714e57ef9b`.

The narrow repair is implemented locally: normalize each slot's log marginal
before exponentiation, but first reject any pre-normalization log-mass drift
outside a conservative FP32 accumulation envelope so the projection cannot hide
a graph/recurrence defect. Retain column-occupancy and ordered-expectation
fail-closed checks, leave the graph, partition, Viterbi path, K, loss and model
architecture unchanged, and add both a small brute-force Gibbs equivalence test
and a `T=768,K=384` high-dynamic-range finite-gradient regression. The formal
code gate now includes `tests/test_duca_structured_selection.py`. This state is
`implemented / authoritative_Linux_Slurm_tested`. Independent post-patch audit
accepted the mathematical categorical normalization but identified that an
unbounded projection could hide a uniformly scaled DP error; the final code
therefore rejects raw log-mass drift outside an FP32 accumulation envelope
before normalizing.

Exact repair source `7e8935692b732f2958ba3c20787ae19c86f7b15c` was transported
with bundle SHA-256
`e95b36a53f661c7b98063aeda7dabcc45d2254d5decf353e44578ec928e01476`.
Code gate `1215388` completed `0:0` with 140 Linux/PyTorch tests, including the
long exact-K backward regression; receipt SHA-256 is
`92b31ab9a9d33580c3b27a367a3866d5ffdf5cf2f9e34a186384ef42e6e0bbf0`.
Real short-window gate `1215389` completed `0:0`; receipt SHA-256 is
`904dacf22d6d88cba8206367c6b824da002f717e7227ad1f310afa08a4a89b16`.

Fresh Stage-A root is
`/data/run01/sczc063/yuzibo/rime_runs/duca_paper_stage_a_7e893569_20260803_030306`.
Protocol, submission and released-receipt SHA-256 values are respectively
`33903a7eee1089202295161368edf4e5e5b3470c88b072c9fa00f82f15fefaf3`,
`1f4b14569f45977d6b606febe7c42898d401258ee5f73b918b56d197d65bb105`, and
`cea74e4bd231e8687e583f2fa98ff5dacec3c41e2ec5352396e097e7abfb01f7`.
Control/DUCA pairs are `1215390/1215391`, `1215392/1215393`, and
`1215394/1215395`; dependent seal is `1215396`. All six groups entered
`RUNNING`; the DUCA jobs crossed the previous immediate failure boundary with no
registered error signature. This remains `ENGINEERING_STATUS`; no loss,
checkpoint or partial metric was opened. Stage B and all metric access remain
sealed.

Terminal update on `2026-08-04`: all six execution jobs failed and seal
`1215396` was cancelled by exact ID after becoming dependency-impossible.
Controls `1215390/1215392/1215394` completed their dense cell and fixed-uniform
training, then failed fixed-uniform evaluation-budget validation. A read-only
predicate replay found the first and only violated predicate at ledger line 722:
`(video_test_0001431, window_start_frame=7680)` was written twice as two
byte-identical rows (the same canonical row SHA-256
`400d197499d056b2874aa1646cccd56cbc83378ea24e2b76e3bf47a75d2b2fb6`). Register
`paper_exact_uniform_eval_duplicate_window_ledger_key`. Requested/effective/
unique/backbone K were all contract-consistent for that row; this is not a
short-window or K384 feasibility failure.

Learned-DUCA jobs `1215391/1215393/1215395` failed when the raw physical exact-K
slot mass exceeded the registered FP32 normalization envelope. This is a repeat
of `physical_exactk_long_chain_fp32_slot_mass_loss`, not a new auto-repairable
engineering signature. Anti-repetition rule 139 therefore requires fail-closed:
do not widen the envelope, suppress the invariant, switch the paper path to FP64
or launch another matrix automatically. Only three dense terminal cell receipts
exist; fixed-uniform has training receipts but no cell receipts, mixed-K never
started, learned DUCA has no training/cell receipt, and no matrix seal exists.
No loss, checkpoint, partial/single-seed metric or mAP was opened. Stage A has no
performance result and cannot be compared with AdaTAD or support Stage B.

The user authorized recovery-v3 implementation and redeployment. The local
repair is now implemented:

- both exact-uniform Phase-1 budgets require and hash-check the absolute
  VideoMAE initialization, resolve it during precheck, pass it in the actual
  `tools/test.py` override, and bind its SHA-256 in the terminal receipt;
- dense ActionFormer/TriDet protocols now use a separate dense-reference
  evaluation predicate and validator. They remain outside trainable-RIME formal
  routing, use their registered `training` development subset, emit a dedicated
  engineering-only terminal schema, and validate the absolute pretrain,
  backend, selector absence, block list, checkpointing, saved predictions and
  official-final exclusion;
- local Python/Bash checks and an expanded 96-test focused suite passed.

Recovery v3 is now deployed as `ENGINEERING_STATUS`:

- exact clean source:
  `bbf051410839f7bec36b0f2cc085de0cd5041cad`;
- independent MAX deployment review: `GO`;
- authoritative Slurm code preflight `1200405`: `COMPLETED`, with 193 remote
  Linux/Torch contract tests passing;
- full launcher preflight `1200462`: `COMPLETED`; Phase-1 dense, uniform K384,
  uniform K192, paired cost and both immutable dense-salvage arms passed. The
  salvage prechecks wrote no output and left source jobs `1198115`/`1198116`
  classified as `FAILED`;
- production physical-protocol SHA-256:
  `69a9cc0b85aaa647a5641f3c00eadd9b8405e8435d3ed5820aae3949df210f4c`;
- production salvage-manifest SHA-256:
  `f7c09b017a4e973211c0f816f55de506d68046801886066dfff3555f15942aef`;
- fresh transaction:
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_bbf05141_20260728_215335`;
- submission-manifest SHA-256:
  `53a633c162dd69ec3bdfd291e8df97d8e79619d9b688808d0dfad36127abc265`;
- jobs: code gate `1200483`, Phase 1 `1200484`, ActionFormer salvage
  `1200485`, TriDet salvage `1200486`, Phase 2 `1200487`, and Phase-3
  controller `1200488`.

Terminal update at `2026-07-28 22:06 CST`:

- code gate `1200483`: `COMPLETED`, exit `0:0`;
- ActionFormer/TriDet salvage `1200485`/`1200486`: `FAILED`, exit `1:0`, after
  immutable EMA copying but before inference because the launcher did not bridge
  `DUCA_RIME_EXPECTED_COMMIT` to the evaluator's canonical
  `DUCA_EXPECTED_COMMIT`;
- Phase 1 `1200484`, Phase 2 `1200487`, and controller `1200488` were canceled
  by exact ID once the transaction could no longer complete;
- no dense checkpoint/recovery, Phase-1, Phase-2, or Phase-3 terminal receipt
  exists. Phase 4 was never opened and official-final remains sealed.

Recovery v4 is limited to the explicit evaluator commit-environment bridge and
a precheck that exercises the same environment lookup and Git comparison as the
formal evaluator. It does not change the model or scientific protocol. The new
regression failed before implementation and passed afterward; Python
compilation, Bash syntax and the expanded focused suite completed with
`97 passed`.

Recovery v4 is now remotely verified and deployed as `ENGINEERING_STATUS`:

- exact clean source:
  `1b44fe3a35042d28c55b9e838f69107bd1461810`;
- independent clean-commit deployment audit: `GO`;
- authoritative Slurm code preflight `1200583`: `COMPLETED`, with 194 remote
  Linux/Torch contract tests passing;
- full launcher/runtime preflight `1200601`: `COMPLETED`; it deliberately
  injected a stale canonical evaluator commit before both salvage checks, and
  the launcher overwrite-bridge restored and verified the exact required
  identity without writing production output;
- production physical-protocol SHA-256:
  `2d416cddd923aa46693ad5361979558e845252947fcb50491cd5cc6c6e70be8c`;
- production salvage-manifest SHA-256:
  `2fb3f9c1a7623e059f855227c34d7614ef2fb6c9e29ee5461e29b4cf5f107d11`;
- fresh transaction:
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_1b44fe3a_20260728_221502`;
- submission-manifest SHA-256:
  `ca72b350ccd7227671554e6e413281cd7059c97f5c3161e2ed93c7a087549767`;
- released submission-receipt SHA-256:
  `eca9e24a06ad7ff2a187066f2f255eb4d764a2f3c5362734444463fa6c128449`;
- jobs: code gate `1200627`, Phase 1 `1200628`, ActionFormer salvage
  `1200629`, TriDet salvage `1200630`, Phase 2 `1200631`, and Phase-3
  controller `1200632`.

Validated deployment snapshot at `2026-07-28 22:25 CST`:

- code gate `1200627`: `COMPLETED`, exit `0:0`;
- ActionFormer/TriDet salvage `1200629`/`1200630`: `COMPLETED`, exit `0:0`;
  each produced its registered checkpoint evidence and passing recovery receipt,
  while retaining the source training job as `FAILED` and the claim scope as
  `engineering_dense_reference_recovery_not_method_evidence`;
- Phase 1 `1200628`: `RUNNING`;
- Phase 2 `1200631` and Phase-3 controller `1200632`: dependency-held;
- Phase 4 remains disabled and official-final remains sealed.

The commit bridge has therefore passed the actual formal-evaluator boundary
that failed recovery v3. The transaction as a whole is not yet complete, and
`No paper-admissible empirical conclusion is available yet`.

Failure update at `2026-07-28 22:32:51 CST`:

- Phase 1 job `1200628` failed with exit `1:0` during the first actual
  exact-uniform K384 forward;
- the exact exception is
  `ValueError: dynamic RIME backbone requires an aligned [B,K] mask` at
  `BackboneWrapper._prepare_dynamic_temporal_bucket`;
- the trace proves that `ActionFormer.forward_test` took its non-physical-selector
  branch and called `self.backbone(inputs)` without the dataset mask, while the
  configured dynamic temporal bucket requires the aligned mask even for the
  no-selector exact-uniform baseline;
- the launcher precheck only loaded and asserted config/protocol fields; it did
  not build the model or execute a real tensor forward, so it could not detect
  this handoff gap;
- no Phase-1 terminal receipt exists. Phase 2 job `1200631` is
  `DependencyNeverSatisfied`, and Phase-3 controller `1200632` remains
  dependency-held;
- the already completed dense recovery receipts remain valid engineering
  recovery evidence. Phase 4 remains disabled and official-final remains sealed.

This is an execution-contract failure, not a performance result. No reported or
intermediate metric was used in this diagnosis.

Recovery-v5 is now `implemented / static_checked / independently_reviewed_GO /
targeted_remote_torch_tested / authoritative_preflight_passed /
production_transaction_running`. It introduces one shared detector-to-backbone
handoff: ActionFormer and TriDet pass the exact aligned mask whenever
`dynamic_temporal_bucket=True`, ordinary backbones retain their legacy mask-free
call, and a physical RIME selector paired with a non-dynamic backbone fails
closed. Both detector train/test call sites are AST-guarded, the Slurm code gate
includes the focused runtime contract test, and the uniform precheck freezes the
dynamic bucket and 16-frame quantum. Windows PyTorch execution remains blocked
by the pre-existing `c10.dll` initialization failure, so no runtime test is
claimed locally. No protocol, model objective, budget, split, checkpoint,
hyperparameter, metric, or paper claim changed.

Operational update:

- exact code source and GitHub branch tip at deployment freeze:
  `74de620d8fafc365694aa1f400318a401add3ecc`;
- clean remote checkout:
  `/data/run01/sczc063/yuzibo/OpenTAD_DUCA_HRIME_74de620d`;
- focused Linux/PyTorch mask-contract suite: `10 passed`;
- recovery-v4 impossible children `1200631`/`1200632`: canceled by exact ID;
- authoritative Slurm preflight `1201057`: `COMPLETED`, exit `0:0`; its
  commit-bound receipt has SHA-256
  `740bc46cff9db814dc8e6c1ae5ad9051db6c6bc9503979969515268462cf0af3`;
- the attempted CPU-only replacement was rejected before job creation because
  this cluster requires a GPU request on its only submission partition;
- the first invocation of the frozen deployment script
  `/data/run01/sczc063/yuzibo/rime_prerequisites/deploy_duca_rime_recovery_v5_74de620d.sh`,
  SHA-256
  `780cd27f36a68d307a4fd90168a96dfe1db3a34e530c9f332a594e78a3b769a1`,
  failed before creating a protocol, manifest or production root because the
  clean checkout lacked the ignored runtime data symlinks and therefore could
  not resolve `data/thumos-14/annotations/thumos_14_anno.json`;
- the exact failure is a protocol-preserving deployment-environment signature,
  `missing_runtime_data_symlinks_before_protocol_freeze`. Its failed freeze log
  is preserved and the unused target root ending in `20260728_233000` remains
  absent;
- the remote checkout's runtime annotation/video symlinks were restored to the
  established immutable datasets and Git cleanliness was reverified;
- the single permitted retry used
  `/data/run01/sczc063/yuzibo/rime_prerequisites/deploy_duca_rime_recovery_v5_74de620d_retry1.sh`,
  SHA-256
  `22698937ff31b4fc696df2a73b0e737c2eeca5fdcd660b8bfcd5c9b46faba635`,
  and released the fresh root
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_74de620d_20260728_235100`;
- production physical-protocol SHA-256:
  `65db63c4b3ebb7f407099efe0f3a97670c19359b0a6f680cb44114645cb3b244`;
- production salvage-manifest SHA-256:
  `af3c466e6d2f61ea9284de540e5b353bbecb1609c0fb40face172c7d1e642acf`;
- submission-manifest SHA-256:
  `6290a5f0bbe15128f8313fec1aaf3003b306e1e7b7c6a3259a3fd21b46beaeb5`;
- released-receipt SHA-256:
  `81d5ce399a0568df744908f7068dc53746b00b7b9f5b4df08d4a5d429c54c95e`;
- jobs: production code gate `1201169`, Phase 1 `1201170`, ActionFormer
  salvage `1201171`, TriDet salvage `1201172`, Phase 2 `1201173`, and
  Phase-3 controller `1201174`.

Production code gate `1201169` subsequently completed with exit `0:0`; its
receipt binds the exact source commit and has SHA-256
`7de03703c23ae79772b8598bea7de3fbaa0db85bffc58d71f467e9f7294045e4`.
Both salvage jobs have now completed with exit `0:0` and terminal engineering
receipts:

- ActionFormer `1201171`: checkpoint-evidence SHA-256
  `72699b01de350c36a2fa6243215aad0bc0294c6c21cf68c07565e1e4d6df9832`,
  recovery-receipt SHA-256
  `2a245ad1209fe8986da612754fbd47c68656e9c136ecd0e448798319232cf5bf`;
- TriDet `1201172`: checkpoint-evidence SHA-256
  `5549264c89dccfc7adec06e7ea14c41c1650d07879a138be8779efab96a5689c`,
  recovery-receipt SHA-256
  `37e6980daecc3b77ae406d3be0b5cfaca43fc5fb39e3f389f095b0ec2246f3a1`.

Both receipts preserve source jobs `1198115`/`1198116` as `FAILED`, do not
reclassify them, use no official-final data, and restrict their claim scope to
`engineering_dense_reference_recovery_not_method_evidence`.

Phase 1 `1201170` then failed with exit `1:0` during the exact-uniform K192
short-window path. The dynamic bucket correctly supplied 16-frame chunks, which
produce eight temporal tokens after tubelet embedding, but
`vit_adapter.Adapter.forward` attempted to reshape each runtime chunk with its
nominal configuration-time `temporal_size=192`. The exact exception was
`RuntimeError: shape '[-1, 192, 10, 10, 96]' is invalid for input of size
1075200`. No Phase-1 terminal receipt exists. Dependency-impossible jobs
`1201173`/`1201174` were canceled by exact ID; Phase 4 was never opened.

This unique engineering failure is registered as
`vit_adapter_static_temporal_axis_on_dynamic_k_bucket`. Recovery-v6 implements
the bounded repair: derive the runtime temporal token count from
`N / (h * w)`, reject non-integral geometry, never mutate the configured
temporal size, and exercise both the valid dynamic-bucket path and invalid
geometry in the focused Slurm-gated test. Static compilation, Bash syntax, and
`git diff --check` pass. An independent read-only audit returned `GO` for this
bounded commit and confirmed that the route uses `VisionTransformerAdapter`,
not the untouched ladder adapter.

The exact repair commit is
`5a599e909aca58751711979e8c9e5b68ab6cab72`. It is pushed to GitHub and
installed in the clean remote checkout
`/data/run01/sczc063/yuzibo/OpenTAD_DUCA_HRIME_5a599e90` from a verified bundle
with SHA-256
`6e4052a5ae4f8e74a2cbfa12303415712b5b41b84906ff1b5c27fd8853edca48`.
The required annotation/video runtime symlinks resolve to the established
immutable datasets while Git status remains clean. Slurm preflight script
`/data/run01/sczc063/yuzibo/rime_prerequisites/duca_rime_submit_v6_code_gate.sh`
has SHA-256
`3e02fabb176d93d5dc125992c55bd80e0188fd85519a2dcd2b0be240e7903a35`;
it submitted authoritative gate `1201390` with root
`/data/run01/sczc063/yuzibo/rime_preflight/duca_rime_recovery_v6_5a599e90_20260729_003219`.
Its first snapshot was queued on scheduler priority. No scientific protocol or
paper claim changed. This is only
`ENGINEERING_STATUS`; no model-quality or paper-admissible empirical conclusion
is available.

Slurm preflight `1201390` completed with exit `0:0`; its exact-commit receipt
has SHA-256
`bef1f6446ceab601b910bfee0f21d0d0d95a297e426455bf682a064f3f4fb2be`.
The frozen Recovery-v6 deployment script
`/data/run01/sczc063/yuzibo/rime_prerequisites/deploy_duca_rime_recovery_v6_5a599e90.sh`
has SHA-256
`f44ff20e8a7acf134581fb460c1eb1188da02070c09aff7bf2bb9cb20e89c8f9`
and released fresh root
`/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_5a599e90_20260729_003600`.
Its immutable identities are:

- physical-protocol SHA-256:
  `94ebe87782e5375afe71ed1506f13e3c812d105f018a3ccdf24eea450f0a35f9`;
- salvage-manifest SHA-256:
  `61f7cfec47b0a467b1f8e616487686937b51bc96098ca15e776c31ff024fa7f0`;
- submission-manifest SHA-256:
  `759fe6e97b10edf03128b6b2244dbab6cbc3e5009d7fdf1d8d9f5319d5d3375a`;
- released-receipt SHA-256:
  `007cee9134ebdba67563681b6bbc3a5e1cecbcf7ad998c688d1cd131bcdbd691`.

Jobs are production code gate `1201416`, Phase 1 `1201417`, ActionFormer
salvage `1201418`, TriDet salvage `1201419`, Phase 2 `1201420`, and Phase-3
controller `1201421`. The dependency map and exact commit passed validation.
At the first snapshot, `1201416` was running and every child was
dependency-held. `phase4_submission_enabled=false` and
`official_final_sealed=true`.

Production code gate `1201416` subsequently completed with exit `0:0`; its
exact-commit receipt SHA-256 is
`34152cfe1fb6c008f4cd20d11f3ed1c6dd19f980caf45d2b1069a029a065146d`.
Phase 1 `1201417` and both salvage jobs `1201418`/`1201419` subsequently
entered `RUNNING`; Phase 2 `1201420` and controller `1201421` remain
dependency-held.

Both salvage jobs then completed with exit `0:0` and exact terminal engineering
evidence:

- ActionFormer `1201418`: checkpoint-evidence SHA-256
  `f5b4f231686fe9aec9e79545ee2eba010d4004e07d285dae05830bb2ede8d7a3`,
  recovery-receipt SHA-256
  `45590ba3a02a06526cf1ad16d217c33c98e77d2c24aeea7509a8a1bee2adcbf1`;
- TriDet `1201419`: checkpoint-evidence SHA-256
  `d979e854a3f75f49f58c5d168bcee5eb5716bcdcb1af6cb5f2595b9a21669327`,
  recovery-receipt SHA-256
  `ba3e7ddaa310bdf36a78723738545de2b99c76560f28d97742c257ee7538257a`.

Both receipts bind Recovery-v6 commit, preserve source jobs
`1198115`/`1198116` as `FAILED`, do not reclassify them, use no official-final
data, and retain claim scope
`engineering_dense_reference_recovery_not_method_evidence`.

Phase 1 `1201417` subsequently failed with exit `1:0` at the protected
full-model admission gate. The terminal contract was
`ProtectedPhysicalGateFailure` with status
`p1_p2_full_model_gate_failed` and exact error
`protected physical full-model gate failed: exact-uniform physical and
selected-axis detector losses disagree`. Register the unique signature as
`protected_physical_exact_uniform_selected_axis_loss_equivalence_gate_failed`.
The immutable log is
`/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_5a599e90_20260729_003600/logs/rime-phase1-1201417.out`
with SHA-256
`0b9aedc943139e024939fa16bf5cf3007c7ae387e74f04bdae823551e3baee29`.
No Phase-1 pipeline receipt exists.

This is a failure of the frozen scientific/admission contract, not a launcher,
path, environment, or tensor-runtime exception. The bounded monitor therefore
correctly failed closed, made no code or protocol change, and did not retry.
That operational classification does **not** prove that the model implementation
is wrong: the gate's own equivalence premise now requires revalidation.

An independent code-and-math audit found:

- exact-uniform positions are integer round-half-to-even anchors; unless
  `(valid_len - 1)` is divisible by `(K - 1)`, selected-to-physical time is
  piecewise linear rather than globally affine;
- the physical head maps centers and approximates a local physical stride,
  while target assignment uses center sampling, regression ranges and
  stride-normalized offsets; IoU/GIoU loss is invariant to one global positive
  affine transform, not to a general piecewise-linear warp;
- therefore exact physical-axis versus selected-axis detector-loss equality is
  not a theorem for the implemented representation, even under uniform
  selection;
- the `1e-4` loss/proposal and `1e-6` target/score tolerances were introduced as
  hard-coded engineering tolerances in commit `ce5d03ebf`; no derivation,
  repeated-null calibration, or FP16/FP32 error study is registered;
- the failure path records neither full versus short-padded window, offending
  loss key, error magnitude, threshold, nor FP32 replay. The current artifact
  therefore cannot distinguish a minor numeric miss, a true semantic bug, or an
  over-strong equivalence premise.

The corrected verdict is `gate_failed_closed /
universal_loss_equivalence_premise_invalid /
observed_mismatch_component_unresolved`. Exact dependency-impossible children
`1201420`/`1201421` were canceled. The production gate and two engineering-only
dense-salvage receipts remain valid and immutable; Phase 4 was never opened and
official-final remains sealed. This terminal state is `ENGINEERING_STATUS` only
and is not a model-quality or paper-admissible performance conclusion.

The apparently high Phase-1 terminal mAP values are also not official-final
performance. The split manifest selects 20 of the 200 `training` videos by
blocking the other 180. The historical checkpoints use the THUMOS `training`
subset as their standard training domain, and no checkpoint-specific manifest
proves that these 20 videos were excluded. They are therefore high-confidence
in-sample sanity controls. Recomputing pooled mAP from the immutable prediction
files reproduced the terminal values exactly, so score normalization,
`top_k=None`, or the per-video diagnostic aggregation is not the explanation.
Do not compare these values with the upstream 69.03 official validation result.
These values are withdrawn from future performance explanations and remain only
as `ENGINEERING_STATUS`.

## Central research question

Given an offline video, can a cheap full-video scan predict a
length-normalized total heavy-compute budget and allocate it across the video's
overlapping 768-candidate AdaTAD windows, while each window still chooses exact
physical positions, so that high-IoU/short-action localization is protected
against exact realized-cost controls and measured full-stack cost falls?

## Current and proposed decision, statistical, and cost units

The earlier phrase “video-level risk decides K” was semantically wrong for the
**current implementation**, but a true video-level budget is a valid and now
preferred next-model design. These facts must not be conflated.

1. **Current implementation:** one dataset row is one 768-candidate crop/window.
   `RimeBudgetController` summarizes cheap `[B,T,D]` evidence for that row and
   predicts a window-level `[B,M]` utility/risk panel. No cross-window
   aggregation, total-video budget, or joint allocation exists.
2. **Proposed planning unit:** one complete offline video. A cheap first pass
   obtains all window summaries plus a video summary and predicts a normalized
   budget density, not a raw duration-blind scalar K. The resulting `B_v` is the
   total heavy-compute quota for that video.
3. **Proposed allocation unit:** the set of 768-candidate windows belonging to
   the video. A hard discrete allocator chooses `K_vw` jointly under
   `sum_w c_vw(K_vw) <= B_v`; it does not force every window to use the same K.
4. **Proposed selection/execution unit:** one 768-candidate window. The existing
   exact-K physical decoder chooses positions inside the assigned `K_vw`, and
   AdaTAD still performs detection and physical-time remapping window by window.
5. **Statistical unit:** video. Cross-fitting, calibration folds, confidence
   intervals, and paired bootstrap keep all windows from one video together;
   windows are not independent experimental units.
6. **Cost unit without a cross-window cache:** the sum of actual heavy execution
   over all windows, including repeated work in the default 25%-overlap region.
   `E_v=sum_w K_vw` is the execution count. The unique physical-frame union
   `U_v` and duplication ratio `E_v/U_v` are diagnostics only; they become
   compute units only after a real shared-feature/cache implementation.
7. **Current evidence:** the window controller and exact-K decoder exist.
   H-RIME now also has a locally tested deterministic contract/core for
   canonical effective-K aliases, reachable-budget projection, exact-equality
   MCKP, stable video grouping, hash-bound replay, and homogeneous-K dispatch
   planning. The Stage-1 development-oracle surface now connects replay to the
   full detector and emits machine-verifiable window-coverage, merge, NMS,
   saved-prediction and official-evaluator receipts. Fourteen targeted
   Torch/Linux tests on the exact clean Stage-1 commit passed, including strict
   source-checkpoint compatibility, short-window replay, and actual
   merge/NMS/evaluator receipt construction. It does **not** yet have a connected
   learned video-budget head, an executing shared-video scan, the grouped
   training path, or calibration evidence.

For working candidate `H-RIME`, let `q=16`, let `W_v` be the video's windows,
and let each feasible `K_vw` be quantum-aligned and bounded by the valid window
length. A video head predicts utility/risk over a registered budget-density
panel. After a frozen training-only price selects `B_v`, the allocator solves

`argmax_{K_v1...K_vW} sum_w [u_vw(K_vw)-beta*r_vw(K_vw)]`

subject to the measured hard cost and feasible-K constraints. The minimal first
version uses an exact multiple-choice knapsack and charges overlap twice because
the current heavy backbone recomputes it. An overlap-interaction term or
cross-window feature cache is a separately falsified extension.

The user correction on `2026-07-28` promoted this hierarchical route from a
deferred alternative to the preferred next-model design. The user subsequently
approved the audited H-RIME specification and authorized implementation. Its
literal state is now `user_approved`, `designed`,
`core_and_stage1_oracle_surface_implemented`, `local_non_torch_tested`, and
`remote_torch_tested`; the learned/shared-scan detector path is not yet
implemented or empirically supported and must not be silently attributed to the
current four-stage RIME code.

## Frozen method semantics

1. The external detector grid is 768 candidate positions.
2. Candidate heavy budgets are `K=(192,256,384,512)`, quantum 16.
3. The heavy VideoMAE backbone receives exactly the selected effective K; no
   Kmax padding is allowed.
4. The detector backbone, projection, adapter, head, losses, and NMS remain the
   registered ActionFormer or TriDet backend.
5. Selection decisions may use only cheap inference-visible evidence. GT,
   teacher outputs, validation/test labels, raw-prediction caches, and
   counterfactual ledgers are forbidden at inference.
6. Predictions are mapped from the selected axis back to physical time before
   official evaluation and NMS.

## Four stages and what they produce

### Phase 1 — execution and geometry closure

Produces exact-K physical execution, dense/uniform/no-probe/probe controls,
coordinate round-trip audits, inference ledgers, and real cost instrumentation.
This is an algorithmic/evidence foundation, not a new final model.

### Phase 2 — trainable baseline and causal admission

Produces the probe-free `U-mixed-K` detector, whose per-training-sample
60-epoch exposure histogram is exactly `(8,12,16,24)` over
`(192,256,384,512)`, hence mean K=384. The stateless sample is a video-associated
random crop/window, not an inference-time whole-video decision. Phase 2 also
produces video-grouped cross-fitted targets, counterfactual measurements,
O1–O4 causal gates, and two frozen budget protocols. This is a new trainable
baseline and decision protocol, not the final DUCA-RIME model.

### Phase 3 — first DUCA-RIME candidate

Produces the first trainable candidate (`RIME-full`) and its causal arm matrix:
`U-fixed`, `F-bound`, `D-no-risk`, `AdapTok-TAD`, `D-shuffle`, plus evaluation-
only `U-same-K`. Every train arm has exactly 6000 successful detector updates.
Only a passing development receipt authorizes Phase 4.

### Phase 4 — frozen publication validation

Retrains and evaluates the frozen candidate over:

- detector: ActionFormer, TriDet;
- panel: K384, K192;
- fresh seed: 5801, 8123, 12011.

This produces 12 formal cells and a fail-closed matrix receipt. It does not
invent a fourth model; it determines whether the Phase-3 candidate is
empirically supportable and transferable.

## Budget-panel correction

- `K384`: `frozen_price_dynamic_budget`; content-conditioned dynamic allocation
  is allowed and must realize at least two requested K values.
- `K192`: `fixed_floor_budget_position_only`; all requested budgets are exactly
  192. Risk predictions may still supervise learned positions, but they do not
  allocate K. No dynamic-budget claim is allowed for this panel.

Reason: when 192 is the minimum candidate budget, a risk-triggered fallback to
larger K makes a mean-192 dynamic policy mathematically infeasible.

## Cost correction

Variable-K RIME is cost-matched against `U-same-K`, which replays the exact
per-window realized K map keyed by `(video_id, window_start_frame)`. Costs are
then aggregated per video. `U-fixed` remains the fixed-budget accuracy
comparator. The profiler reads `effective_k` before legacy `effective_budget`;
otherwise RIME would be incorrectly reported as dense K=768.

## Approved next-model design

The current priority candidate is `H-RIME`: a whole-video total-budget planner,
joint per-window K allocator, and existing within-window exact-K physical
selector. Whole-video budget means one total quota `B_v`, not one identical K
for every window. The current window-local RIME remains a necessary baseline.

The governing design is
`docs/superpowers/specs/2026-07-28-hrime-v1-budget-conserving-design.md`.
External review `U-PRO-HRIME-1` is conditionally accepted, not copied verbatim:

- Approach C and Stage-0 → oracle → factorized H-RIME → publication-admission
  sequencing are accepted;
- suggested numeric gates are proposals until frozen from training/calibration
  roles in a pre-committed manifest;
- raw video caps are projected to reachable effective-K totals, and receipts
  expose both projection and solver unused budget;
- the allocation surrogate must be validated by full official merge/NMS/
  evaluator replay;
- calibration includes Brier, reliability, risk-coverage and worst-group
  behavior rather than ECE alone;
- primary endpoint, multiplicity and noninferiority rules are pre-registered;
- MCKP dtype, score quantization, tie-break, version and assignment hash are
  frozen;
- shared scanning, grouping and homogeneous-K dispatch are implemented as
  separate audited interfaces rather than assumed from the current flat-window
  detector.

The external `U-PRO-CBCG-1` review refined `Pair-Risk Graph RIME` into the
working candidate `CBCG-RIME`: place calibrated boundary-coverage failure on
consecutive physical-selection edges for same-K position choice. CBCG-RIME is
now an optional within-window extension of H-RIME, not the first implementation
priority.

Its first gate is a complete, genuinely held-out, same-K oracle. Before learned
code, the oracle must resolve edge-target identifiability, define normalized
source/internal/sink gap masses, hard-cap graph bandwidth, preserve one
hard/soft energy, and separate gap-only effects from content-conditioned edge
risk. External sandbox patch hashes and reported synthetic tests are
`PARTNER_CLAIM`; those artifacts are absent from this repository.

The current direct-transfer arm must be described as an
`AdapTok-inspired TAD budget allocation baseline`, not an official AdapTok
reproduction. Conformal risk remains a possible calibration fallback, while
two-round sequential acquisition is deferred because of latency, cache, and
AdapTok-overlap risk.

## Claim gate

A positive paper claim requires:

1. development Phase 3 passes before the official-final set is opened;
2. all 12 Phase-4 cells are present and hash-bound;
3. RIME beats both best fixed and uniform same-K under paired video-cluster
   bootstrap;
4. high-IoU, short-action, and pair-support non-degradation gates pass;
5. measured full-stack latency is below dense;
6. seed directions are positive for every detector/budget panel.

Until those artifacts exist, the correct status is `implemented/tested` or
`experiment_running`, never `empirically_supported` or `paper_ready`.

## Immediate execution

Current Stage-A recovery decision:

1. `U-PRO-STAGEA-MINIMAL-SOLVER-REPAIR-1` is accepted as
   `GO_MINIMAL_SOLVER_REPAIR`, with two bounded corrections. Do not copy its
   solver pseudocode without carrying both alpha and beta scales, and do not add
   the proposed generic execution journal after the duplicate was proven to
   originate in dataset window enumeration.
2. Canonical sliding-window starts must be unique and retain the existing
   snippet-center endpoint/annotation slicing semantics. Ledger finalization
   continues to fail closed on duplicate physical identities; no post-hoc
   deduplication is allowed.
3. AMP coverage log probabilities cross into the physical exact-K solver at
   FP32. The solver uses a per-slot normalized log-semiring with explicit scale
   recovery and a global additive exact-K gauge; an explicit FP64 path is oracle
   evidence only, never the paper execution path.
4. Replace the gauge-dependent raw slot-mass envelope with structural
   identities: forward/backward logZ agreement, brute-force small graphs,
   additive-shift invariance, FP32/FP64 oracle agreement, row/column/order
   predicates, finite gradients and unchanged hard path.
5. Before any new Stage-A release, require a clean implementation commit,
   authoritative Linux/PyTorch gate, real natural-short-window heavy-backbone
   gate, a production-like learned `T=768,K=384` DDP/AMP numeric gate, an
   exact-211 metadata/physical-UID dry-run and fresh commit/hash-bound
   manifests/root. All four prerequisite receipts must propagate through the
   submission, training, cell and matrix-seal chain. No old receipt, checkpoint
   choice, partial cell or metric can enter the replacement matrix.
6. The four-gate release chain is implemented and locally contract-tested. The
   numeric gate must reproduce the superseded raw-message failure within at most
   100 real full-model updates, then pass frozen FP32/FP64, dual-logZ, edge-flow,
   gradient and hard-path checks on the captured production tensor. This is
   engineering stability evidence only. Linux/Slurm receipts and a fresh Stage-A
   transaction remain absent, so no model-performance conclusion exists. The
   implementation through `3a0563fb` is pushed and independently audited `GO`.
   The apparent remote block was corrected on `2026-08-05`: `N16R4` is not a
   public DNS host but a local alias for gateway
   `ssh.cn-zhongwei-1.paracloud.com`, using ParaCloud login identity
   `sczc063@BSCC-N16R4`; the Slurm cluster reached after login is `n16r4`.
   A local SSH config now records this mapping, and a read-only probe returned
   host `ln01`, user `sczc063`, `/usr/bin/sbatch`, `/usr/bin/squeue`, and
   `ClusterName=n16r4`. Remote release is therefore no longer connection-blocked;
   no new gate or Stage-A job is claimed until an exact commit-bound submission
   actually returns its scheduler identity and receipt.

   The first authoritative release attempt on exact source `e0a58ab2` submitted
   code-gate job `1222939` at root
   `/data/run01/sczc063/yuzibo/rime_preflight/duca_paper_code_gate_e0a58ab2_20260805_235718`.
   It failed after `146 passed / 1 failed`: the new small-graph FP64 oracle test
   compared a `Double` slot-mass tensor with a default-`Float` `torch.ones`
   reference, and the installed PyTorch rejects cross-dtype `allclose`. Register
   `linux_fp64_oracle_expected_ones_dtype_mismatch`. The selector output had
   already matched the FP64 brute-force distribution; this is a deterministic
   Linux-only test-construction defect, not a model/numeric/scientific failure.
   The bounded repair uses `ones_like` on the actual slot-mass tensor and changes
   no model, solver, loss, budget, data, seed, evaluator, threshold or metric.
   No downstream gate or Stage-A job was submitted and no metric was opened.

   The repaired source `dfe787f1` subsequently passed authoritative code gate
   `1222944` and real short-window gate `1222951`. Combined release-gate job
   `1222954` then failed before exact-211 with signature
   `numeric_gate_per_update_t768_k384_capture_requirement`: its implementation
   contradicted the frozen “within at most 100 updates” contract by requiring an
   exact `T=768,K=384` rank-local capture after every successful update. Natural
   short rows can validly produce smaller effective K, and the two DDP ranks can
   differ. The authorized narrow correction permits non-target search updates,
   performs the same `MAX` trigger collective on every rank at every update, and
   retains the terminal fail-closed owner check within 100. This is gate control
   flow only; all scientific/model settings and thresholds remain frozen. No
   Stage-A job or metric exists from this attempt.

   Exact repair source `751ce695` passed code gate `1223013` (148 tests) and
   natural-short gate `1223116`, but release-gate `1223142` failed on both ranks
   before `scaler.step` with non-finite unscaled gradients. Register
   `numeric_gate_pre_step_gradient_finiteness_bypasses_formal_amp_replay`. The
   formal config and production engine already freeze up to eight AMP retries per
   batch with RNG/model-buffer/custom-state restoration and successful-update-only
   optimizer/scheduler/selector accounting. The numeric gate had bypassed that
   semantics by rejecting overflow before GradScaler could skip and reduce its
   scale. The authorized correction mirrors the production bounded replay,
   synchronizes step success across ranks, requires finite gradients on the
   successful attempt, and still fails after eight retries. No model, loss,
   budget, threshold, data, seed, checkpoint or evaluator change is permitted.
   Fresh four-gate verification remains mandatory; no result exists yet.

   Exact AMP-replay source `4b766457` passed code gate `1223270` (149 tests) and
   natural-short gate `1223282`. Its two-rank release gate `1223308` then crossed
   both earlier gate-control-flow failures and reached the terminal bounded-search
   predicate, but failed because `old production FP32 guard did not trigger within
   100 updates`. No numeric receipt was written, so capture/replay counts and the
   largest non-triggering legacy statistic are not persisted. Exact-211 and
   Stage-A were not started. This failure cannot be auto-repaired: mandatory
   reproduction of the superseded gauge-dependent raw-message guard is now a
   scientific gate-premise question. Do not remove it, change the envelope or
   extend/retry the search without adjudication. The next action is one bounded
   Pro review of whether actual T768/K384 production-tensor coverage plus the
   current structural oracles is sufficient, or whether an immutable historical
   failure fixture must remain as a separate negative control.

Approved compact feasibility design, superseding the simulation-first order
below while retaining it as negative-history context:

1. Preserve old Admission v2/v2.1 paths fail-closed and historical. Do not build
   their missing distributed runner or execute candidate-free MC grids.
2. Create one paper-facing full-data DUCA feasibility protocol from the upstream
   AdaTAD ActionFormer recipe: all 200 training videos, two-GPU global batch two,
   60 epochs/6000 successful updates, fixed epoch-59 EMA, and exact complete
   211-video OpenTAD evaluation.
3. Close only experiment-enabling P0s: jointly trained train-only ASFormer
   evidence, exact VideoMAE/protocol/split hashes, per-rank batch one for
   homogeneous execution, and an evaluator that rejects missing or extra video
   keys. These checks are part of the paper experiment, not a separate evidence
   claim.
4. Run Stage A on ActionFormer/K384 with seeds 5801, 8123 and 12011: dense
   AdaTAD, uniform fixed-K, uniform mixed-K-training/exact-K384-evaluation, and
   DUCA learned fixed-K384 positions. Only after full-200 train-only OOF targets
   exist may Stage B add DUCA dynamic-K and evaluation-only exact realized-cost
   `U-same-K` replay.
5. Report only the completed official matrix: official average and per-IoU mAP,
   high-IoU and short-action behavior, realized effective K, full-stack latency,
   throughput and peak memory, with paired video-cluster bootstrap and every seed.
6. The core feasibility question is whether DUCA improves the official
   accuracy--cost frontier over both fixed and exact realized-cost uniform
   controls. If it does not, stop H-RIME and diagnose selector/controller
   learning. If it does, use the complete result to design the smallest model
   improvement and only then extend to H-RIME, TriDet and K192.
7. The exact-211 paper-feasibility evaluation is now explicitly authorized under
   this hash-bound Stage-A protocol. It does not reopen or reclassify the old
   failed Phase-4 transaction. Recovery-v6 remains immutable failed history.

Retained earlier recovery instructions:

1. Treat transaction `d9d454cd` as terminally failed closed. Do not release its
   dependency-blocked Phase 2/controller or reuse missing receipts as evidence.
2. Preserve both raw epoch-59 dense checkpoints immutably. A future repair may
   compact and evaluate them only through a new hash-bound post-processing
   transaction that names the original training jobs and source checkpoint
   hashes; it may not overwrite the failed root or pretend the original jobs
   passed.
3. Correct the checkpoint invocation/import surface and add a clean-repository
   runtime test that exercises the same launcher command.
4. Repair the short-window exact-uniform execution path. The actual VideoMAE
   input must be the quantum-aligned feasible `K_eff`, with
   `backbone_input_k=unique_k=effective_k`; duplicated K384 tail frames cannot
   be labeled as K231 savings.
5. Run focused checks, a `PRECHECK_ONLY=1` launcher, and a new immutable
   Phase-1 transaction after the implementation commit is clean. Recover the
   window-local Phase-2/3
   development path as a required baseline, but do not spend the official-final
   matrix on it merely because the old DAG named it the candidate.
6. Before implementing the video planner, run a held-out,
   same-total-heavy-cost allocation oracle. It must compare uniform allocation,
   independent window RIME, and joint video-level allocation using exact
   per-window replay and video-grouped statistics. Stop if cross-window
   redistribution has no material high-IoU/short-action headroom.
7. If the oracle passes, implement the two-pass `H-RIME` path: full-video cheap
   scan, normalized `B_v` prediction, exact joint `K_vw` allocation, K-bucketed
   window execution, and cross-window ledger. Only a complete development
   matrix can freeze the final candidate for Phase 4.
8. Keep official-final evaluation sealed unless a future complete development
   receipt authorizes Phase 4.
9. Do not apply the external Patch A/B artifacts to production. Patch A may be
   specified as a held-out same-K oracle only after the Phase-1 execution/split
   prerequisites close; Patch B stays on hold until the complete oracle,
   cross-fit calibration, causal, and full-stack cost gates pass.
10. Implement only the H-RIME core/interfaces and held-out oracle surface before
    the oracle receipt. Do not launch large learned-H-RIME training merely
    because implementation has started.
11. Treat recovery transaction `0ab242f3` as immutable failed engineering
    evidence. Do not promote its partial EMA/salvage/evaluation sidecars into
    terminal dense evidence or reuse the root in place.
12. Before another recovery submission, bind and hash-check the absolute
    VideoMAE initialization in every actual Phase-1 evaluator, including both
    uniform controls, and make precheck execute the same resolved override.
    Give dense salvage an explicit training-subset engineering-evaluation role
    in the structured evidence path and test receipt finalization end to end.
