# Research Log

## 2026-07-29 — Official full-train boundary corrected

- Audited the complete Phase-4 training and evaluation path in response to the
  user's official-comparability correction.
- Confirmed that the registered training set has 200 videos, while the current
  Phase-4 trainer exports the `detector_selector_train` block list and trains
  RIME plus matched controls on only 100. This is a valid development partition
  but cannot support the paper main table or direct published-number claims.
- Confirmed that the evaluation path requires `block_list=None`, the exact
  registered official-final video-key set, and rejects missing or extra
  predictions. OpenTAD's data contract intentionally excludes two
  malformed/empty THUMOS test videos, leaving a complete comparable set of 211.
- Identified the required refit contract: frozen method, leakage-safe OOF
  targets for all 200 training videos, global effective batch two, 60 epochs,
  100 optimizer updates per epoch, 6000 total updates, matched controls and the
  unchanged upstream evaluator/NMS.
- Hard-disabled the current Phase-4 cell entrypoint. No official-final
  experiment may start until the full-200 refit producer, receipts and code
  gate exist. No current result is paper-admissible.
- Current focused checks pass (`66 passed`); the broad non-Torch contract suite
  passes (`219 passed, 3 skipped`). Full Torch collection remains unavailable
  on the Windows host due the known `c10.dll` WinError 1114, so an exact
  clean-commit Linux/PyTorch Slurm gate is still required.

## 2026-07-29 — Admission v2.1 audit adjudicated; impossible window contract found

- Fully read and registered `U-PRO-ADMISSION-V21-1`. Accepted its current
  formal-v2 `NO-GO`, all twelve P0 findings, preservation of the pure
  selected-axis model, and the real-video/full-model/disjoint-holdout direction.
- Did not accept the proposed protocol verbatim. Independently identified four
  blockers: the fixed natural-window coverage is physically infeasible, the
  sparse video/process bootstrap and catastrophic bound are underdefined,
  `2 * reporting quantum` is not a scientific NI-margin justification, and
  repository-level Slurm code cannot prove hard network/mount/object-lock
  isolation without cluster support.
- Read-only remote verification bound split manifest SHA-256
  `41349cd39a6a550b6e1613de968577b1605c93902edd52a88309121b9e90c057`.
  Its `detector_selector_train` pool has 100 videos. Production-compatible
  enumeration from immutable annotation metadata gives 70 full-only, 30
  short-only, zero with both, and fixed short-bin counts 7/13/10. Thus `3 x 32`
  is count-feasible, but the proposed per-video full-plus-short rule and
  minimum-eight first short bin are not.
- Implemented the noncontroversial Stage-0 response. Old Admission v2 formal
  calibration/admission now fails closed; the random-head path is
  engineering-fixture-only; old v2 receipts require explicit
  historical-read-only parsing and cannot authorize production entrypoints.
- Added a pure metadata v2.1 feasibility auditor and focused tests. It mirrors
  `SlidingWindowDataset.split_video_to_windows`, emits a finite exclusive
  content-bound typed failure, consumes no decoded frames or candidate output,
  and never authorizes Phase 1.
- Full v2.1 calibration and Phase 1 remain unstarted. The next decision must
  freeze role-level window coverage, exact crossed statistics, a justified NI
  margin and enforceable isolation claims. Phase 4 and official-final remain
  sealed; no model-performance or paper-admissible empirical conclusion exists.

## 2026-07-29 — DUCA acquisition-v2 selected-axis implementation

- Accepted the core Pro adjudication with bounded implementation corrections:
  the paper mainline is the pure pre-backbone selected-axis acquisition plugin;
  physical-time head injection remains a separately named integration route.
- Implemented and locally audited the selected-axis coordinate contract,
  standard-head restoration, exact-K/no-padding execution, GT remapping,
  inverse mapping before NMS, content-bound Admission v2, numeric-null
  calibration, and pre-candidate scientific-protocol freeze.
- Commit `70cf49de82a9d0ed889ed94af9604edd61070e55` was pushed and transported by
  SHA-bound Git bundle to a clean N16R4 checkout. Bundle SHA-256 is
  `5347836c57564a151f10818908d36e7488211ca66a72e9bf4356c70405c6e9af`.
- Authoritative Slurm code-gate job `1204048` failed closed with exit `1:0`;
  247 tests passed and one stale assertion still expected the superseded
  `duca_protected_physical_v1` Phase-1 protocol instead of
  `duca_rime_selected_axis_plugin_v2`. No gate receipt was produced.
- Failure signature:
  `phase1_uniform_test_expected_superseded_physical_protocol`.
  The correction updates only the regression expectation and adds assertions
  that the selected-axis plugin uses the standard detector head. The focused
  local regression passes. No model, loss, data, threshold, checkpoint,
  metric, or scientific protocol changed.
- The failed preflight root is immutable:
  `/data/run01/sczc063/yuzibo/rime_preflight/duca_acquisition_v2_70cf49de_20260729_202923`.
  Numeric calibration, scientific admission, Phase 1, Phase 4, and
  official-final were not started.
- Remediation commit `119db2f83756281729506632a18bfed607794d13`
  passed authoritative Slurm code-gate job `1204067` with exit `0:0`.
  Gate-receipt SHA-256 is
  `d664a619007f1cafbd4e52f2fd6a053fb0e3b5336dcb2be2b16302912286e5c8`.
- A post-gate evidence audit stopped the workflow before numeric calibration.
  No non-fixture training-target JSONL, acquisition data manifest, or
  `duca_acquisition_ni_margin_source_v1` exists in the registered remote
  assets. Code-gate fixture targets and block lists are forbidden substitutes.
- An independent specification-to-code audit found a P0 protocol gap in the
  current numeric calibrator: it runs synthetic head-feature fixtures while
  declaring `train_only_calibration`, does not consume real role-scoped videos,
  has no video/process-grouped null distribution or independent process
  launches, and does not freeze the registered normalized statistic and
  denominator floor. Therefore the calibrator is an implemented draft, not a
  valid Admission-v2 producer.
- Admission v2, Phase 1, Phase 4, and official-final remain unopened. The next
  authorized work is to repair and preregister the real-data numeric
  calibration protocol and generate its train/calibration-only input assets;
  no performance claim is available.

## 2026-07-27 — DUCA-RIME four-stage implementation

- Recorded user approval for direct four-stage execution.
- Marked the earlier total-60 bounded-density plan as superseded by the
  dynamic-budget RIME adjudication.
- Implemented Phase 1 exact-K/geometry/cost evidence, Phase 2 `U-mixed-K` and
  causal gates, Phase 3 candidate/ablation matrix, and Phase 4 formal matrix.
- Corrected the mixed-K schedule to a per-video stateless 60-entry exposure
  with exact mean K=384.
- Corrected K192 to `fixed_floor_budget_position_only`; it cannot support a
  dynamic-budget claim.
- Corrected full-stack cost matching to exact `U-same-K` replay.
- Corrected the profiler to consume `effective_k`.
- Added an explicit Phase-4 authorization → Phase-2 receipt → budget protocol
  path/hash → checkpoint audit → terminal identity binding.
- Independent MAX audit found and corrected a deployment-blocking dense
  evidence SHA variable-name mismatch between the Phase-3 controller and
  Phase-3/4 submitters.
- Deployment preflight also corrected the one legacy Phase-1 gate invocation
  to run through `bash`, because that retained script is intentionally tracked
  without an executable bit.
- Rejected fabrication of a trained commit for the historical exact-uniform
  checkpoint whose surviving log records `git_head=unknown`. The Phase-1
  no-probe/probe cost pair now uses one byte-identical, SHA-bound checkpoint
  trained at `cb89586a92b8b0a8349ecc9551bc50aa97982360`; the launcher and seal
  require identical checkpoint SHA, trained commit, epoch, and EMA state.
  The no-probe arm drops only the registered probe/transition state that its
  configuration does not build, so the common heavy path is weight-identical.
- Retrieved the official released AdaTAD VideoMAE-S/ActionFormer checkpoint
  from the source-linked Google Drive file through its direct user-content
  endpoint. Its size is `200938640` bytes and SHA-256 is
  `21dbb9efe9f62d3089696c3c535edd27e8b8d9c14a06a21aac5738ec82bfab97`.
- Pre-registered the exact K/cost ladder, K384/K192 panel semantics,
  `weak_overlap` decoder, risk rule, O4 calibration gates, and 2 s / 8 s
  duration strata in both the submitting commit and submission manifest.
- The first released transaction at commit `57965bec` failed at the code gate
  before any experiment work because Slurm `--wrap` executes under `/bin/sh`
  and therefore rejected the Bash-only `source` builtin. All downstream jobs
  remained dependency-blocked. The submitter now explicitly enters
  `/bin/bash -lc` before the CUDA/Miniforge bootstrap; the failed root and
  scheduler records are retained as negative deployment evidence.
- The next code gate at commit `c0e7e036` reached the full remote Torch suite
  and stopped on two stale test fixtures: a `protected_e2e` RIME construction
  requested bridge scale `0.0` despite the registered `1.0` contract, and a
  no-padding ledger fixture omitted `irregular_dense_valid_len`. The fixtures
  were corrected to exercise, rather than violate, their production contracts;
  no model behavior or gate threshold was relaxed.
- The diagnostic rerun at commit `8667d057` cleared those failures and exposed
  one final stale floor-protocol fixture that constructed non-uniform RIME
  without the required official trainable ASFormer evidence source. The fixture
  now supplies that registered source contract; production behavior and every
  scientific gate remain unchanged.
- At commit `3f8e3ca1`, the exact remote suite passed all 158 tests. Its first
  unobstructed config-matrix run then exposed an over-broad generic assertion:
  the two evaluation-only Phase-1 paired cost profilers were being checked as
  train/evaluation-result configs. The matrix now keeps batch-size-one and saved
  predictions mandatory for trainable/formal-evaluation configs, while applying
  the stronger relevant contract to those two cost-only configs: test batch
  size one, zero loader workers, no saved accuracy predictions, no accuracy
  claim, and a byte-identical paired checkpoint identity.
- Current state: `implemented/tested`; remote code gate and Slurm deployment
  remain pending. No empirical or paper-ready claim has been made.

## 2026-07-28 — DUCA-RIME final transaction released

- Froze the physical protocol on implementation commit
  `f510741b32075c5c4e729d4207a549886a6dd064`; manifest SHA-256 is
  `5d28d1d37e698b5f17156245f55da62a82dc5b537c32fe70104f1be231e605d8`.
- Released the immutable fail-closed transaction at
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_four_phase_f510741b_20260728_094811`.
  Submission-manifest SHA-256 is
  `c74c351bad04dd7bfc6701ca5205e419116694d21736419776cec1f3cdb7ada6`.
- Code-gate job `1197889` completed successfully on the exact clean commit:
  all 158 focused tests passed and all 24 registered configs passed their
  stage-specific matrix.
- Slurm released Phase 1 (`1197890`), dense ActionFormer (`1197891`), and dense
  TriDet (`1197892`) into `RUNNING`. Phase 2 (`1197893`) and the Phase-3/4
  controller (`1197894`) remain dependency-gated as registered.
- Current scientific state is `experiment_running`, not
  `empirically_supported` or `paper_ready`.
- Early monitoring then found both dense jobs failed before their first
  optimizer update: their configs correctly set `dataset.val=None` and disabled
  validation intervals, but omitted the generic trainer's explicit
  `seal_eval_dataloaders_during_training=True` switch. The Phase-3 dependency
  failed closed. Phase 1 and all remaining jobs in this now-uncompletable
  transaction were canceled by exact job ID; the root and logs are retained.
- Added the missing seal to both dense configs and bound it in the launcher
  precheck, config matrix, and focused tests. This is an orchestration fix, not
  a model or scientific-protocol change.
- The next immutable transaction at commit `1ff54baf` passed its code gate.
  Phase 1 and dense ActionFormer ran normally, and ActionFormer reached update
  50. Dense TriDet then failed on its first backward pass because it alone still
  inherited reentrant VideoMAE gradient checkpointing; DDP reported a parameter
  marked ready twice. The transaction again failed closed, and jobs `1197975`,
  `1197976`, `1197979`, and `1197980` were canceled by exact ID.
- Dense TriDet now explicitly sets `with_cp=False`, as the already-working dense
  ActionFormer and protected/RIME bases do. The launcher and code gate reject
  any future dense reference with checkpointing re-enabled. This changes memory
  use, not the detector, objective, data, update count, or publication claim.
- The corrected dense precheck passed in Slurm job `1198049`. Diagnostic TriDet
  job `1198059` then completed 50 stable updates and was intentionally canceled;
  it is smoke evidence only.
- Released the active immutable transaction on commit
  `d9d454cd49a3e7a87694fc948601d00ff4043cb0` at
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_four_phase_d9d454cd_20260728_101256`.
  Protocol SHA-256 is
  `c4dfc31a64b56a93366c43443883df535e572eed38df63878fe11d3e00193a70`;
  submission-manifest SHA-256 is
  `ed374ae81991ca8241c0b01ab6588f13ea292b967b18a58115ec3f735440b038`.
- Code gate `1198113` passed. Phase 1 `1198114`, dense ActionFormer `1198115`,
  and dense TriDet `1198116` are running; both dense references passed update
  50. Phase 2 `1198117` and controller `1198118` remain correctly
  dependency-gated.
- Recorded `Pair-Risk Graph RIME` as a `discussed` post-v1 candidate. It is not
  designed, implemented, tested, or authorized to alter the frozen transaction.
- Monitoring snapshot `2026-07-28 10:40 CST`: Phase 1 job `1198114` failed
  closed after 12m44s in `finalize_duca_rime_inference_ledger.py`. Uniform-K384
  ledger line 64 for `video_validation_0000686` recorded
  `dense_valid_len=231`, `effective_k=unique_k=231`, but
  `backbone_input_k=padded_k=384`, violating the frozen no-padding cost
  contract. Its development localization JSON exists, but no evaluation
  receipt was emitted. Phase 2 job `1198117` is
  `DependencyNeverSatisfied`; controller `1198118` remains dependency-pending.
  Dense ActionFormer `1198115` and dense TriDet `1198116` continued into epoch
  9 with zero observed Traceback, OOM, non-finite-loss, or gradient-skip
  matches. No official-final data were opened and no downstream stage is
  authorized.
- Audited the unexpectedly high Phase-1 terminal mAP. The split manifest has
  200 `training` videos and uses a 180-ID block list to retain only 20
  certification-development videos. Historical checkpoint configs train on
  the same THUMOS `training` subset, and no checkpoint-specific exclusion
  manifest was found, so the 20-video measurements are high-confidence
  in-sample sanity controls. Pooled mAP was independently recomputed from the
  immutable predictions and exactly matched the terminal JSON; no
  self-normalization or `top_k=None` inflation was found. These numbers must
  not be compared with the upstream 69.03 official validation result.
- User froze a paper-responsibility contract: partial, training-domain,
  intermediate, small-subset, single-seed, unmatched, or missing-receipt
  results may be retained only as engineering status and must never be used to
  explain model performance or support the paper. Froze the full contract in
  mandatory-read `research-wiki/query_pack.md`. Until a complete comparable
  experiment exists, the correct empirical statement is that no paper-
  admissible conclusion is available; the only alternative is a self-contained
  theoretical analysis with explicit assumptions and limits.

## 2026-07-28 — CBCG-RIME external review absorbed and adjudicated

- Fully read and registered external review `U-PRO-CBCG-1`. Its overall
  freeze-v1 → same-K oracle → learned-head hold → causal/full-stack gates route
  was conditionally accepted.
- Corrected three stale execution claims in the review: the code gate has
  already passed; Phase 1 has failed closed on the no-padding ledger; Phase
  2/3/4 are dependency-blocked even though the two dense references were still
  running at the last verified snapshot. “Wait for the four-stage run to finish”
  and “immediately apply Patch A” are therefore not the current execution plan.
- Retained `CBCG-RIME` only as a working refinement of the `discussed`
  Pair-Risk Graph idea. It narrows generic pair risk to calibrated
  boundary-coverage failure on consecutive physical-selection edges, while
  preserving video-level risk for K selection.
- Recorded hard design blockers before implementation: path-to-edge regret
  attribution is underidentified without balanced perturbations and stability
  evidence; source/sink gap masses are not yet mathematically normalized;
  sparse complexity requires an enforced span cap; gap-only confounding,
  hard/soft energy equality, cross-fit calibration, and bit-exact risk-off
  behavior need explicit tests.
- The report's linked sandbox patches, hashes, and reported test count are
  unavailable in the repository and remain `PARTNER_CLAIM`; no code from them
  was applied and no implementation/test status was promoted.
- Standardized the comparison name to
  `AdapTok-inspired TAD budget allocation baseline`. The official AdapTok paper
  and repository still require direct provenance registration before
  publication.
- No empirical performance conclusion was added. The current project still has
  no `PAPER_ADMISSIBLE_RESULT`.

## 2026-07-28 — Risk granularity corrected and four-stage terminal state verified

- Resolved the user annotation about AdaTAD's 768-point input. The current
  controller does not compute whole-video risk: it pools cheap `[B,T,D]`
  evidence per training crop or inference sliding window and returns a
  window-level `[B,M]` utility/risk panel.
- Froze the unit distinction for future designs and writing: model decisions are
  per 768-candidate window; cross-fitting/statistics are grouped by video; costs
  are the per-video sum over actual heavy inputs for all windows, including
  overlap. `U-same-K` must replay each `(video_id, window_start_frame)` before
  aggregation.
- Compared three abstractions. Whole-video scalar K was rejected for v1 because
  it is not implemented and can wash out local short-action/boundary demand.
  Window-local K is the current recommended route. A hierarchical video prior
  plus window residual is deferred until window-local v1 establishes genuine
  headroom and cross-window correlation.
- Independently queried Slurm at `2026-07-28 14:47 CST`. Code gate `1198113`
  completed; Phase 1 `1198114` failed the no-padding ledger; dense ActionFormer
  `1198115` and dense TriDet `1198116` both failed after their 60-epoch training
  loops during checkpoint compaction; Phase 2 `1198117` and controller
  `1198118` remain `DependencyNeverSatisfied`.
- Both dense raw epoch-59 checkpoints exist, but neither terminal EMA,
  training/evaluation receipt, checkpoint binding, nor any Phase-1/2/3/4
  terminal receipt exists. The latest transaction is therefore terminally
  failed closed, not complete.
- Identified the dense post-training failure surface:
  `python tools/bata/compact_duca_rime_checkpoint.py` cannot resolve
  `from tools.bata import duca_p0_training` in the released environment.
  No remote state was changed.
- Preserved the raw checkpoints as possible inputs to a future, separately
  hash-bound salvage transaction. They are engineering artifacts, not positive
  experimental evidence.
- No model code or launcher was modified because the corrected risk/execution
  design is awaiting user approval under the brainstorming gate. No empirical
  performance conclusion was added.

## 2026-07-28 — Whole-video budget correction and H-RIME proposal

- The user clarified that offline cheap scanning can plan over the complete
  video even though AdaTAD continues to execute and detect on 768-candidate
  windows. Accepted this as a substantive correction to the earlier
  window-only final-model recommendation.
- Distinguished a total video quota `B_v` from a uniform per-window K. The
  proposed hierarchy is video-level budget prediction, joint window-level
  `K_vw` allocation, and existing within-window exact-K physical selection.
- Audited the current code contract: videos are flattened into overlapping
  window rows; the current controller, target/replay path, and ledger are
  per-window; no whole-video joint decision exists. H-RIME is recorded as
  `discussed/designed/awaiting_user_approval`, not implemented or tested.
- Froze truthful overlap accounting. Under the current no-cache backend, heavy
  work in overlapping windows is recomputed, so formal cost is
  `sum_w K_vw`; unique physical frames and the duplicate ratio are diagnostics,
  not compute savings.
- Promoted a held-out same-total-heavy-cost allocation oracle ahead of learned
  implementation. It must compare uniform, independent-window, and joint
  video-level allocation and stop the route if cross-window redistribution has
  no material high-IoU/short-action headroom.
- Reframed the failed four-stage RIME path as a required window-local baseline
  and infrastructure source, not automatically the final publication model.
  Phase-1 execution and dense checkpoint closure remain prerequisites.
- No model/launcher code or remote scheduler state was changed. No empirical
  performance claim was made.

## 2026-07-28 — Pro H-RIME report absorbed and implementation authorized

- Fully read and registered `U-PRO-HRIME-1`. The user approved the adjudicated
  route and authorized implementation.
- Accepted the main Approach-C architecture: shared full-video cheap scan,
  normalized total-video budget, exact MCKP per-window allocation, reuse of the
  existing exact-K selector, homogeneous-K heavy dispatch, and unchanged
  AdaTAD/NMS.
- Froze the corrected design at
  `docs/superpowers/specs/2026-07-28-hrime-v1-budget-conserving-design.md` and its
  implementation plan at
  `docs/superpowers/plans/2026-07-28-hrime-v1-implementation.md`.
- Did not accept the report verbatim. Numeric accuracy/calibration/cost gates
  remain proposals until a training/calibration-only pre-registration; raw caps
  are projected to reachable effective-K totals; official merge/NMS replay must
  validate the additive oracle; risk evaluation extends beyond ECE; endpoints,
  multiplicity, seed interpretation and deterministic MCKP ties are explicit.
- Independently audited the current repository. It has a reusable exact-K
  decoder and per-window replay, but flat window datasets, no whole-video
  planner/allocator, no grouped two-pass dispatch and no shared video scan.
  H-RIME was therefore `implementation_started`, not implemented/tested at that
  audit point.
- Independently verified the failed dense raw checkpoints:
  ActionFormer source job `1198115`, size `623799387`, SHA-256
  `cd92f3d499360c834f7ddd6ccfd5cba172c870bf6922de566b2b7e3878680e11`;
  TriDet source job `1198116`, size `411540059`, SHA-256
  `8940dbe756e8abfa3f7c8b042f3c658b26898d5c805d2876011a4e7510d11e12`.
  Both are epoch-59 raw checkpoints with EMA state but lack complete embedded
  commit/variant/seed provenance.
- Froze a new immutable salvage requirement. The failed root remains unchanged,
  and recovered artifacts must bind source job/path/size/hash plus explicitly
  external provenance.
- Scheduler energy fields were unavailable/zero and no trusted GPU monitor was
  active, so no energy claim is permitted.
- Directly registered official AdapTok and EVATok paper/code sources. Per-video
  adaptive allocation/routing is prior-art context, not H-RIME's novelty claim.
- No model performance claim was added. The project still has no
  `PAPER_ADMISSIBLE_RESULT`.

## 2026-07-28 — Stage-0 repair and H-RIME deterministic core implemented

- Repaired the short-window execution contract. Candidate requests now map to a
  homogeneous, quantum-aligned effective K before heavy execution; a 231-valid
  window maps `(192,256,384,512)` to `(192,224,224,224)`, and the heavy tensor
  width/ledger use 192 or 224 without replicated inactive tail.
- Changed all dense checkpoint compactors to clean-cwd module invocation and
  added a manifest-driven recovery tool/launcher. It validates the failed root,
  source job IDs, exact epoch-59 path/size/SHA/schema/EMA keys, writes only to a
  fresh recovery root, records missing embedded provenance honestly, keeps the
  original job state `FAILED`, and makes no energy claim.
- Added a dual-mode recovery DAG. `fresh_train` remains the default;
  `salvage` requires a frozen manifest/hash and only redirects the standard
  downstream checkpoint-evidence pointers. The failed transaction is never
  modified. Phase 4 is forced sealed in this recovery DAG.
- Implemented `hrime_exact_equality_mckp_v1`: canonical effective-K alias
  deduplication, reachable-cap projection, exact equality DP, frozen int64
  score quantization, deterministic objective/risk/lexicographic tie-break,
  and solver-input/assignment hashes.
- Implemented stable complete-video window groups, shared-scan receipt
  contracts, video budget-plan hashes, exact-K replay rows for the existing
  selector, and homogeneous-K dispatch/inverse-restoration plans. These are
  deterministic contracts, not yet a connected learned/shared-scan runtime.
- Independent code audit caught and corrected non-16-aligned K acceptance,
  fractional integer truncation, and insufficient feasible-set/plan/replay hash
  binding before merge.
- Local Python compilation, Bash syntax and 46 focused no-Torch tests passed.
  Torch-dependent tests are not claimed locally because the Windows host cannot
  load the CUDA-linked `torch` DLL; authoritative Slurm verification remains
  pending.
- No experiment performance claim was made. The same-total-cost oracle has not
  run, learned H-RIME is not implemented, and no paper-admissible empirical
  conclusion is available.
- The mandatory zero-context MAX deployment audit returned an initial `NO-GO`:
  the repaired fixed-window execution ledger still wrote a null physical
  protocol hash and did not distinguish raw/requested, reachable, realized,
  projection-unused and solver-unused budgets. No Slurm job was submitted.
- Corrected the blocker by requiring the exact physical-protocol SHA-256 in
  every Stage-0 ledger row, adding all five budget fields with explicit
  `window_fixed_request` / `stage0_engineering_window_execution` scope, and
  making the finalizer aggregate and fail closed on inconsistent budget truth.
  The expanded focused non-Torch suite then passed 64 tests. A fresh audit on the
  corrected clean commit remains mandatory before deployment.
- The clean-commit re-audit returned a second `NO-GO`: the selector producer was
  strict, but `run_duca_rime_phase1_uniform_eval.sh` did not pass the protocol
  hash or demand explicit budget truth from the finalizer, so a legacy row could
  still be sealed. Corrected the consumer chain: the launcher now requires and
  hash-checks the protocol manifest, invokes the finalizer as a module with the
  expected hash plus `--require-explicit-budget-truth`, and the parent pipeline
  exports the binding before K384/K192. The expanded focused non-Torch suite
  now passes 66 tests. Slurm remains unsubmitted pending another clean audit.
- A third zero-context MAX audit on commit `534da568` returned `GO`. A
  commit-bound physical protocol and immutable two-backend salvage manifest
  were frozen, but the first recovery submission met the account-level
  `AssocMaxSubmitJobLimit` after four held jobs had been created. The nested
  `sbatch` failure did not trigger the parent-only `ERR` cleanup trap.
- Canceled unreleased jobs `1199974`--`1199977` and the route-local stale
  `DependencyNeverSatisfied` jobs `1198117`/`1198118`; no unrelated job was
  changed. Enabled Bash `errtrace` in the four-phase, Phase-3 and Phase-4
  transactional submitters so future nested failures cancel their held prefix.
  The partial submission root remains non-reusable evidence; no experiment or
  performance claim was created.
- Fresh MAX audit returned `GO` on exact clean commit
  `902168a12bc92babd62b6cb1877ce7137f56cea0`. Froze the new commit-bound
  physical protocol (`1823826b...e7e34e`) and salvage manifest
  (`b4f5b7fd...d85a0e`), then atomically released recovery jobs
  `1199978`--`1199983` under
  `duca_rime_recovery_902168a1_20260728_183709`.
- The submission manifest is
  `fd6fef65ac01e7830c6b5e337684b19a3bad65c1432f819cfecb32e83dfefb85`;
  the receipt is released, dense recovery is explicitly engineering-only, the
  original failed jobs remain failed, Phase 4 is disabled, and official-final
  remains sealed. The first snapshot is scheduler-pending with no terminal
  receipts, so no empirical claim is available.

## 2026-07-28 — H-RIME Stage-1 oracle and execution-proof surface implemented

- Implemented the complete development-only Stage-1 strategy matrix:
  same-total uniform, independent window RIME, joint GT oracle, joint allocation
  with uniform positions, and feasibility-preserving shuffled null. Every video
  uses the exact same reachable total effective K across strategies.
- Added an explicit pre-execution preregistration builder. It freezes one
  primary endpoint, video bootstrap, intersection-union multiplicity family,
  noninferiority/materiality gates, guardrails, surrogate thresholds, MCKP
  numeric/tie contracts, evaluator semantics and official-final exclusion.
  No scientific threshold has a result-derived default.
- Connected each Stage-1 replay to the existing exact-K selector while preserving
  strict RIME-full parameter keys and shapes. Oracle permission, decision role,
  GT provenance, requested/effective K, no-padding execution and video-total
  budget truth all fail closed.
- Extended the test runtime to produce a machine-verifiable execution receipt:
  exact sliding-window dataset coverage, model-forward batch count, window
  counts by video, distributed aggregation, pre/post-NMS proposal counts,
  per-video NMS call count, post-NMS prediction SHA-256, official evaluator
  call/success, resolved config identities and implementation source hashes.
  The terminal also records `strict_exact_v1` checkpoint compatibility.
- Rejected negative oracle risk weights at both freeze and validation entry
  points, and rejected prediction artifacts missing any expected development
  video instead of silently interpreting them as empty.
- Local Python compilation and Bash syntax passed. The pure/non-Torch focused
  suites passed 61 Stage-1/RIME tests plus 23 repository-mandated C3 regression
  tests. The new runtime-receipt and strict model-loading tests remain
  `remote_torch_pending` because the Windows host cannot load the CUDA-linked
  Torch DLL.
- A fresh independent diff review additionally required that the shuffled null
  not collapse to the joint oracle for an entire budget anchor. Planning now
  records every degenerate video and fails before writing its output root if no
  non-identity, histogram-preserving feasible allocation exists for that anchor.
- Stage-1 is `implemented/local_non_torch_tested`, not executed. It authorizes no
  learned H-RIME training until Stage-0 closes, a clean-commit remote test and
  independent deployment audit pass, the complete preregistration is frozen,
  and the full development oracle receipt passes. No paper-admissible empirical
  conclusion is available.

## 2026-07-28 — Stage-1 remote verification and Stage-0 recovery failure absorbed

- Pushed exact Stage-1 implementation commit
  `577e748ffb3fe452a57094d3d0bb5f022c32f739` and checked it out cleanly on the
  remote Linux/Torch environment.
- Fourteen targeted remote Torch tests passed in 48.62 seconds: strict
  RIME-full-to-Stage-1 architecture loading, short-window replay,
  cross-window aggregation/NMS/official-evaluator execution receipts, the
  Stage-1 oracle core, and exact expected prediction-video keys. Stage-1 is now
  `remote_torch_tested`, not `experiment_run`.
- Rechecked recovery jobs `1199978`--`1199983`. The code gate passed. Phase 1
  failed because `tools/test.py` received the base config's relative VideoMAE
  initialization path. Both dense salvage arms failed with exit code 126
  because the generated wrapper directly executed a `100644` shell script.
  These are deployment failures, not model or performance evidence.
- Phase 2/controller were verified as `DependencyNeverSatisfied` and canceled by
  exact IDs `1199982` and `1199983`. The failed root, logs, released submission
  receipt, and original failed job states remain untouched.
- Implemented the deployment repair: Phase-1 dense evaluation now requires,
  hash-checks, passes and records the exact absolute VideoMAE initialization;
  the recovery submitter invokes both salvage commands through explicit Bash;
  the salvage launcher mode is restored to `100755`. Bash syntax and 36 focused
  launcher/salvage tests pass locally.
- Independent repair review found that salvage checked and used the VideoMAE
  initialization but did not explicitly carry its SHA-256 into the terminal
  recovery receipt. Added a second in-process hash check plus explicit path/hash
  fields to the source/recovery evidence; this closes provenance rather than
  changing execution.
- This repair still requires a clean commit, remote prechecks, an independent
  audit, new commit-bound manifests and a fresh immutable Slurm transaction.
  No paper-admissible empirical conclusion is available.

## 2026-07-28 — Corrected Stage-0 recovery transaction released

- Committed and pushed the deployment repair as exact source commit
  `0ab242f31be8de7b7da806b645d3aa60d02d8d88`.
- Local compilation/Bash checks and 82 focused tests passed. A clean remote
  checkout then passed the same 82 Linux/Torch tests. An independent
  clean-commit review returned code-level `GO`.
- Remote `PRECHECK_ONLY` passed for Phase-1 dense, exact-uniform and paired-cost
  launchers. Both salvage prechecks reloaded the original raw checkpoints,
  verified 499 ActionFormer and 462 TriDet EMA/state keys, wrote nothing, kept
  source jobs `FAILED`, and confirmed official-final exclusion.
- Froze the commit-bound physical protocol with SHA-256
  `2f11c12d62451c7ec41b54ac889058617f56f889e6f289cfe865a47eb03ff9f9`
  and the fresh two-backend salvage manifest with SHA-256
  `faab636144d0855f2d8f26d6c7298459302b3c84508bdc2da24b1b864013772d`.
- Atomically released jobs `1200135`--`1200140` at
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_0ab242f3_20260728_201613`.
  Submission-manifest SHA-256 is
  `b996543dfe57bc3678799591f38f0e96e76da971eb8d5a4f7a4edbb15aa3d04d`.
  The first snapshot has the code gate priority-pending and every child under
  exact fail-closed dependencies.
- Dense salvage remains
  `engineering_dense_reference_recovery_not_method_evidence`; original jobs and
  roots remain failed/immutable. Phase 4 is disabled, official-final is sealed,
  and no paper-admissible empirical conclusion is available.

## 2026-07-28 — Corrected Stage-0 recovery transaction failed closed

- Rechecked exact transaction
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_0ab242f3_20260728_201613`
  at `21:02 CST`. The deployed checkout remains clean at exact commit
  `0ab242f31be8de7b7da806b645d3aa60d02d8d88`; the submission, physical-protocol
  and salvage-manifest hashes remain exactly registered.
- Code gate `1200135` completed. Phase 1 `1200136` failed in the exact-uniform
  evaluator because its actual `tools/test.py` command did not override the
  base config's repository-relative VideoMAE initialization, despite the
  absolute path being required elsewhere. This is a runtime/precheck coverage
  defect, not performance evidence.
- ActionFormer/TriDet salvage `1200137`/`1200138` compacted raw EMA checkpoints
  and ran their engineering evaluations, then failed structured evidence
  finalization: the evaluator subset was frozen as `training`, while
  `tools/test.py` classified neither salvage role and therefore expected
  `validation`. No terminal dense checkpoint evidence or recovery receipt was
  produced.
- Phase 2/controller `1200139`/`1200140` were verified as
  `DependencyNeverSatisfied` and canceled by exact ID. No unrelated job was
  changed.
- Required Phase-1, dense, Phase-2 and Phase-3 terminal receipts are absent.
  Partial salvage and evaluation artifacts remain engineering diagnostics only.
  Phase 4 stayed disabled, official-final stayed sealed, and no paper-admissible
  empirical conclusion is available.

## 2026-07-28 — Recovery-v3 evaluator contracts implemented locally

- Treated the user's repair/redeploy instruction as approval of the already
  recorded minimal design and froze the implementation addendum at
  `docs/superpowers/specs/2026-07-28-stage0-recovery-v3-contract-repair-design.md`.
- Repaired exact-uniform Phase-1 evaluation so both budgets require,
  hash-check, resolve, actually override and receipt-bind the absolute VideoMAE
  initialization.
- Added a separate dense-reference protocol predicate and validator. This avoids
  the invalid shortcut of adding dense protocols to ordinary RIME formal
  routing, whose later payload requires trainable selector contracts absent
  from dense references.
- Dense evaluation now validates its engineering role, backend, no-selector
  path, 768-window dense execution, `with_cp=False`, training development
  subset/block list, saved predictions, pretrain identity and official-final
  exclusion before emitting a dedicated terminal schema.
- The new regression tests failed before implementation and passed afterward.
  Local Python compilation, Bash syntax, launcher, salvage, RIME contract,
  H-RIME, prediction-metric and mandatory C3 regressions completed with
  `96 passed`.
- State remains `implemented/local_tested/remote_pending`. No new experiment was
  yet submitted, no result was interpreted, and official-final remains sealed.

## 2026-07-28 — Recovery-v3 remotely verified and redeployed

- Pushed exact implementation commit
  `bbf051410839f7bec36b0f2cc085de0cd5041cad`; an independent MAX deployment
  audit returned `GO`.
- Created a fresh clean remote checkout. Slurm code preflight `1200405`
  completed 193 authoritative Linux/Torch contract tests and emitted its gate
  receipt.
- The first combined launcher preflight `1200441` intentionally failed closed
  because the production salvage output parent must exist while the production
  transaction root must remain absent before submission. No production root or
  salvage output was created. A separate same-source, same-commit preflight
  manifest preserved both contracts.
- Full launcher preflight `1200462` completed Phase-1 dense, uniform K384,
  uniform K192, paired-cost and both salvage checks. The two salvage checks
  audited 499 ActionFormer and 462 TriDet EMA keys, wrote no output, preserved
  source jobs as `FAILED`, and excluded official-final.
- Froze production physical protocol
  `69a9cc0b85aaa647a5641f3c00eadd9b8405e8435d3ed5820aae3949df210f4c`
  and production salvage manifest
  `f7c09b017a4e973211c0f816f55de506d68046801886066dfff3555f15942aef`.
- Atomically released jobs `1200483`--`1200488` at
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_bbf05141_20260728_215335`.
  Submission-manifest SHA-256 is
  `53a633c162dd69ec3bdfd291e8df97d8e79619d9b688808d0dfad36127abc265`.
- At `22:04 CST`, production code gate `1200483` was complete; Phase 1 and both
  salvage arms were running; Phase 2/3 remained dependency-held. The deployment
  monitor was rebound to these exact identities. Phase 4 remains disabled,
  official-final remains sealed, and no paper-admissible empirical conclusion
  is available.

## 2026-07-28 — Recovery-v3 failed closed; recovery-v4 bridge designed

- Production code gate `1200483` completed. Dense salvage jobs
  `1200485`/`1200486` then both failed before inference at the exact
  `tools/test.py` guard `formal evaluation checkout differs from
  DUCA_EXPECTED_COMMIT`.
- The checkpoint salvage itself had passed and the immutable sources remained
  unchanged. The launcher required `DUCA_RIME_EXPECTED_COMMIT` but omitted the
  explicit bridge to the evaluator's canonical `DUCA_EXPECTED_COMMIT`.
- Because the transaction could no longer produce all required receipts, Phase
  1 `1200484`, Phase 2 `1200487`, and controller `1200488` were canceled by
  exact ID. No unrelated job was changed. Phase 4 was never opened.
- Froze the recovery-v4 design: explicitly export the canonical commit variable
  and make `PRECHECK_ONLY` execute the same environment lookup and Git identity
  comparison as formal evaluation. Silent evaluator fallback was rejected
  because it would weaken fail-closed launcher diagnostics.
- Recovery-v3 remains immutable failed engineering evidence. No performance
  result was read or interpreted, and no paper-admissible empirical conclusion
  is available.
- Implemented the recovery-v4 bridge in the salvage launcher. It overwrite-
  exports the canonical evaluator commit and runs an evaluator-equivalent
  environment/Git identity probe before both precheck and actual execution.
  The new regression was red before the fix and green afterward; local
  compilation, Bash syntax and the expanded focused suite passed `97` tests.

## 2026-07-28 — Recovery-v4 remotely verified and redeployed

- Pushed exact implementation commit
  `1b44fe3a35042d28c55b9e838f69107bd1461810`; an independent clean-commit
  deployment audit returned `GO`.
- Slurm code preflight `1200583` completed 194 authoritative Linux/Torch
  contract tests. Full launcher/runtime preflight `1200601` then passed all
  Phase-1 and dense-salvage checks, including a deliberate stale canonical
  evaluator-commit injection that the new overwrite-bridge corrected before the
  evaluator-equivalent identity probe. Precheck wrote no production output.
- Froze production physical protocol
  `2d416cddd923aa46693ad5361979558e845252947fcb50491cd5cc6c6e70be8c`
  and production salvage manifest
  `2fb3f9c1a7623e059f855227c34d7614ef2fb6c9e29ee5461e29b4cf5f107d11`.
- Atomically released jobs `1200627`--`1200632` at
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_1b44fe3a_20260728_221502`.
  Submission-manifest SHA-256 is
  `ca72b350ccd7227671554e6e413281cd7059c97f5c3161e2ed93c7a087549767`;
  released receipt SHA-256 is
  `eca9e24a06ad7ff2a187066f2f255eb4d764a2f3c5362734444463fa6c128449`.
- Code gate `1200627` and both salvage jobs `1200629`/`1200630` completed with
  exit `0:0`. Both salvage arms crossed the former commit-identity failure,
  completed formal engineering evaluation and emitted checkpoint evidence plus
  passing recovery receipts. The receipts retain their original source jobs as
  `FAILED` and restrict claims to engineering dense-reference recovery.
- At `22:25 CST`, Phase 1 `1200628` was running; Phase 2 `1200631` and Phase-3
  controller `1200632` remained dependency-held. Monitoring was rebound to these
  exact identities. Phase 4 remains disabled, official-final remains sealed,
  and no paper-admissible empirical conclusion is available.

## 2026-07-28 — Recovery-v4 Phase 1 failed closed on mask handoff

- Phase 1 job `1200628` failed at `22:32:51 CST`, exit `1:0`, during the first
  actual exact-uniform K384 forward. No Phase-1 terminal receipt was produced.
- Exact exception:
  `ValueError: dynamic RIME backbone requires an aligned [B,K] mask` from
  `BackboneWrapper._prepare_dynamic_temporal_bucket`.
- The traceback shows `ActionFormer.forward_test` called
  `self.backbone(inputs)` without the dataset mask because the exact-uniform
  baseline has no `duca_rime_physical` selector. Its backbone nevertheless has
  the dynamic temporal bucket enabled and requires that aligned mask.
- The existing uniform launcher precheck only parsed and asserted config,
  protocol, budget and pretrain identity. It did not construct the model or run
  the tensor-forward boundary, so it could not catch the mismatch.
- Phase 2 job `1200631` became `DependencyNeverSatisfied`; Phase-3 controller
  `1200632` remains dependency-held. The two dense salvage receipts remain
  passing engineering recovery evidence. Phase 4 remains disabled and
  official-final remains sealed.
- This diagnosis used only execution state, traceback and tensor-contract
  provenance. No intermediate or terminal performance value was reported or
  interpreted.

## 2026-07-28 — Recovery-v5 mask handoff implemented

- User authorized a complete correction, redeployment, and bounded self-healing
  monitor.
- Implemented one shared ActionFormer/TriDet backbone handoff. A dynamic
  temporal bucket now always receives the exact aligned detector mask; ordinary
  backbones preserve the legacy invocation; physical RIME paired with a
  non-dynamic backbone fails closed.
- Added a focused runtime contract test for both backends, AST coverage of all
  train/test call sites, Slurm code-gate inclusion, and frozen dynamic-bucket
  assertions in the Phase-1 uniform precheck.
- Python compilation, Bash syntax and `git diff --check` passed. Two independent
  read-only patch audits returned `GO`. Local Torch execution is explicitly not
  claimed because Windows failed loading `c10.dll`; the authoritative
  Linux/Torch gate is pending.
- The repair changes no model objective, budget, split, checkpoint,
  hyperparameter, metric or paper claim. Recovery-v4 remains immutable failed
  engineering evidence, Phase 4 is disabled, official-final is sealed, and no
  paper-admissible empirical conclusion is available.

## 2026-07-28 — Recovery-v5 remote gate queued; monitor upgraded

- Pushed and cleanly installed exact commit
  `74de620d8fafc365694aa1f400318a401add3ecc`.
- The focused Linux/PyTorch detector-backbone mask suite passed all 10 tests.
- Canceled only recovery-v4 dependency-impossible jobs `1200631` and `1200632`
  by exact ID; old roots, logs and valid dense salvage receipts remain
  immutable.
- Initial Slurm gate `1201029` waited on `AssocGrpGRES`. A CPU-only replacement
  was attempted to avoid unnecessary GPU use, but cluster submission policy
  rejected it before creating a job because the only partition requires an
  explicit GPU request. The standard 1-GPU gate was restored as `1201057`; it
  remains queued on `AssocGrpGRES`, not failed.
- Prepared a fresh deployment script at
  `/data/run01/sczc063/yuzibo/rime_prerequisites/deploy_duca_rime_recovery_v5_74de620d.sh`
  with SHA-256
  `780cd27f36a68d307a4fd90168a96dfe1db3a34e530c9f332a594e78a3b769a1`.
  It targets the fresh root
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_74de620d_20260728_233000`
  and cannot run before the commit-bound gate receipt passes.
- Upgraded automation `duca-rime` from 30-minute passive observation to a
  15-minute bounded self-healing contract: one retry per unique deterministic
  protocol-preserving engineering failure, new commit/manifests/root required,
  and fail-closed escalation for repeated, ambiguous, scientific, data,
  numerical or model-quality failures. Phase 4 and official-final remain sealed.

## 2026-07-28 — Recovery-v5 preflight passed; bounded retry released production DAG

- Slurm preflight `1201057` completed with exit `0:0`. Its receipt binds exact
  source commit `74de620d8fafc365694aa1f400318a401add3ecc` and has SHA-256
  `740bc46cff9db814dc8e6c1ae5ad9051db6c6bc9503979969515268462cf0af3`.
- The first hash-frozen deployment invocation failed before producing any
  protocol, manifest or production root. The exact traceback was
  `FileNotFoundError: data/thumos-14/annotations/thumos_14_anno.json`; a clean
  clone had no ignored runtime annotation/video symlinks. The failure log and
  unused root identity ending in `20260728_233000` remain preserved.
- Classified the unique failure as
  `missing_runtime_data_symlinks_before_protocol_freeze`. Restored only the
  established checkout-local symlinks to immutable datasets, verified both
  targets and reverified that the deployed source checkout remained Git-clean.
  This changed no model, protocol, loss, budget, split, checkpoint,
  hyperparameter or metric.
- Consumed the one allowed automatic retry for that signature. Retry script
  `/data/run01/sczc063/yuzibo/rime_prerequisites/deploy_duca_rime_recovery_v5_74de620d_retry1.sh`
  has SHA-256
  `22698937ff31b4fc696df2a73b0e737c2eeca5fdcd660b8bfcd5c9b46faba635`.
- Released fresh immutable root
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_74de620d_20260728_235100`
  with physical-protocol SHA-256
  `65db63c4b3ebb7f407099efe0f3a97670c19359b0a6f680cb44114645cb3b244`,
  salvage-manifest SHA-256
  `af3c466e6d2f61ea9284de540e5b353bbecb1609c0fb40face172c7d1e642acf`,
  submission-manifest SHA-256
  `6290a5f0bbe15128f8313fec1aaf3003b306e1e7b7c6a3259a3fd21b46beaeb5`,
  and released-receipt SHA-256
  `81d5ce399a0568df744908f7068dc53746b00b7b9f5b4df08d4a5d429c54c95e`.
- Atomically released jobs `1201169`--`1201174`: production code gate,
  Phase 1, ActionFormer salvage, TriDet salvage, Phase 2 and Phase-3 controller.
  Production code gate `1201169` completed with exit `0:0`; its exact-commit
  receipt SHA-256 is
  `7de03703c23ae79772b8598bea7de3fbaa0db85bffc58d71f467e9f7294045e4`.
  Phase 1 `1201170` and both salvage jobs `1201171`/`1201172` are running;
  Phase 2 `1201173` and controller `1201174` remain dependency-held. Phase 4
  remains disabled, official-final remains sealed, and no performance value or
  paper-admissible empirical conclusion was produced.

## 2026-07-29 — Recovery-v5 dense salvage arms completed

- ActionFormer salvage `1201171` completed with exit `0:0`. Checkpoint-evidence
  SHA-256 is
  `72699b01de350c36a2fa6243215aad0bc0294c6c21cf68c07565e1e4d6df9832`;
  terminal recovery-receipt SHA-256 is
  `2a245ad1209fe8986da612754fbd47c68656e9c136ecd0e448798319232cf5bf`.
- TriDet salvage `1201172` completed with exit `0:0`. Checkpoint-evidence
  SHA-256 is
  `5549264c89dccfc7adec06e7ea14c41c1650d07879a138be8779efab96a5689c`;
  terminal recovery-receipt SHA-256 is
  `37e6980daecc3b77ae406d3be0b5cfaca43fc5fb39e3f389f095b0ec2246f3a1`.
- Both terminal receipts bind exact recovery commit
  `74de620d8fafc365694aa1f400318a401add3ecc`, retain original source jobs
  `1198115`/`1198116` and source commit
  `d9d454cd49a3e7a87694fc948601d00ff4043cb0` as failed evidence, do not
  reclassify either source job, exclude official-final, and retain the claim
  scope `engineering_dense_reference_recovery_not_method_evidence`.
- Phase 1 `1201170` remains running. Phase 2 `1201173` and Phase-3 controller
  `1201174` remain dependency-held. This update is `ENGINEERING_STATUS`; no
  performance value was inspected or interpreted.

## 2026-07-29 — Recovery-v5 Phase 1 failed; Recovery-v6 engineering repair implemented

- Phase 1 `1201170` failed with exit `1:0`; no Phase-1 terminal receipt exists.
  The exact-uniform K192 short-window path reached the real VideoMAE adapter,
  where an eight-token runtime temporal axis was reshaped using nominal
  `temporal_size=192`. The exact terminal exception was
  `RuntimeError: shape '[-1, 192, 10, 10, 96]' is invalid for input of size
  1075200`.
- Registered unique failure signature
  `vit_adapter_static_temporal_axis_on_dynamic_k_bucket`. This is deterministic,
  reproducible and protocol-preserving engineering correctness, so it is
  eligible for one bounded fresh recovery.
- Canceled only dependency-impossible jobs `1201173`/`1201174` by exact ID.
  Recovery-v5 root, scheduler state, production gate and both valid dense
  salvage receipts remain immutable. Phase 4 was never opened.
- Recovery-v6 derives the adapter's runtime temporal count from
  `N / (h * w)`, rejects non-integral token geometry, and does not mutate the
  configured nominal temporal size. Added red-before/green-after runtime and
  failure-contract tests and added `vit_adapter.py` to the Slurm code-gate
  compilation surface.
- Static compilation, Bash syntax and `git diff --check` pass. Remote Torch
  verification, exact commit publication, commit-bound manifests and a new
  transaction root remain pending. An independent read-only audit returned
  `GO` for the bounded diff, confirmed that active dynamic configs use
  `VisionTransformerAdapter`, and found no scientific-protocol change. No
  performance value was inspected or interpreted.

## 2026-07-29 — Recovery-v6 exact commit published; Slurm gate queued

- Published exact repair commit
  `5a599e909aca58751711979e8c9e5b68ab6cab72`.
- A direct remote GitHub clone encountered a transient TLS termination before
  creating a checkout. Built a complete bundle from the already-pushed branch,
  SHA-256
  `6e4052a5ae4f8e74a2cbfa12303415712b5b41b84906ff1b5c27fd8853edca48`,
  and used it to create clean detached checkout
  `/data/run01/sczc063/yuzibo/OpenTAD_DUCA_HRIME_5a599e90`.
- Restored the required checkout-local annotation/video symlinks to the
  established immutable datasets, verified both targets, exact HEAD and clean
  Git status before submission.
- Preflight submission script SHA-256 is
  `3e02fabb176d93d5dc125992c55bd80e0188fd85519a2dcd2b0be240e7903a35`.
  It submitted Slurm code gate `1201390` at
  `/data/run01/sczc063/yuzibo/rime_preflight/duca_rime_recovery_v6_5a599e90_20260729_003219`;
  the first snapshot is scheduler-pending on priority.
- No Recovery-v6 production protocol, salvage manifest, submission manifest,
  root or DAG exists yet. Phase 4 and official-final remain sealed.

## 2026-07-29 — Recovery-v6 preflight passed; production DAG released

- Slurm preflight `1201390` completed with exit `0:0`; exact-commit receipt
  SHA-256 is
  `bef1f6446ceab601b910bfee0f21d0d0d95a297e426455bf682a064f3f4fb2be`.
- Deployment script SHA-256 is
  `f44ff20e8a7acf134581fb460c1eb1188da02070c09aff7bf2bb9cb20e89c8f9`.
  It generated fresh commit-bound physical protocol
  `94ebe87782e5375afe71ed1506f13e3c812d105f018a3ccdf24eea450f0a35f9`,
  production salvage manifest
  `61f7cfec47b0a467b1f8e616487686937b51bc96098ca15e776c31ff024fa7f0`,
  submission manifest
  `759fe6e97b10edf03128b6b2244dbab6cbc3e5009d7fdf1d8d9f5319d5d3375a`,
  and released receipt
  `007cee9134ebdba67563681b6bbc3a5e1cecbcf7ad998c688d1cd131bcdbd691`.
- Released jobs `1201416`--`1201421` at
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_5a599e90_20260729_003600`.
  They are production gate, Phase 1, ActionFormer salvage, TriDet salvage,
  Phase 2 and Phase-3 controller with the exact registered dependency map.
- The first scheduler snapshot had production gate `1201416` running and every
  child dependency-held. Phase 4 remains disabled, official-final remains
  sealed, and this is only `ENGINEERING_STATUS`.
- Production gate `1201416` subsequently completed with exit `0:0`; its
  exact-commit receipt SHA-256 is
  `34152cfe1fb6c008f4cd20d11f3ed1c6dd19f980caf45d2b1069a029a065146d`.
  Phase 1 `1201417` and salvage jobs `1201418`/`1201419` subsequently entered
  `RUNNING`; Phase 2 `1201420` and controller `1201421` remain dependency-held.

## 2026-07-29 — Recovery-v6 dense salvage arms completed

- ActionFormer salvage `1201418` completed with exit `0:0`; checkpoint-evidence
  SHA-256 is
  `f5b4f231686fe9aec9e79545ee2eba010d4004e07d285dae05830bb2ede8d7a3`
  and recovery-receipt SHA-256 is
  `45590ba3a02a06526cf1ad16d217c33c98e77d2c24aeea7509a8a1bee2adcbf1`.
- TriDet salvage `1201419` completed with exit `0:0`; checkpoint-evidence
  SHA-256 is
  `d979e854a3f75f49f58c5d168bcee5eb5716bcdcb1af6cb5f2595b9a21669327`
  and recovery-receipt SHA-256 is
  `ba3e7ddaa310bdf36a78723738545de2b99c76560f28d97742c257ee7538257a`.
- Both receipts bind exact Recovery-v6 commit, preserve original source jobs
  `1198115`/`1198116` as failed without reclassification, exclude
  official-final, and retain engineering-only claim scope.
- Phase 1 `1201417` remains running; Phase 2 `1201420` and controller `1201421`
  remain dependency-held. No performance value was inspected or interpreted.

## 2026-07-29 — Recovery-v6 failed closed at protected full-model gate

- Phase 1 `1201417` failed with exit `1:0`; no Phase-1 pipeline receipt exists.
  The terminal schema was `ProtectedPhysicalGateFailure`, status
  `p1_p2_full_model_gate_failed`, with exact error `protected physical
  full-model gate failed: exact-uniform physical and selected-axis detector
  losses disagree`.
- Registered failure signature
  `protected_physical_exact_uniform_selected_axis_loss_equivalence_gate_failed`.
  The immutable log path is
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_5a599e90_20260729_003600/logs/rime-phase1-1201417.out`;
  SHA-256 is
  `0b9aedc943139e024939fa16bf5cf3007c7ae387e74f04bdae823551e3baee29`.
- This is a frozen scientific/admission gate failure, not a deterministic
  protocol-preserving engineering defect. The bounded monitor made no code or
  protocol change and did not automatically retry.
- Canceled only exact dependency-impossible jobs `1201420`/`1201421`. Production
  gate `1201416` and both dense-salvage terminal receipts remain valid and
  immutable. Phase 4 was never opened and official-final remains sealed.
- This record is `ENGINEERING_STATUS` only. No model-quality or paper-admissible
  performance conclusion was drawn.

## 2026-07-29 — Recovery-v6 gate premise and thresholds independently audited

- Rechecked the exact gate implementation, ActionFormer physical head, integer
  exact-uniform anchors, commit history, tests and immutable failure log.
- The operational stop remains correct: the frozen gate rejected the run, so no
  child experiment or official-final access was allowed.
- The scientific root cause is not established. Integer round-half-to-even
  anchors generally induce a piecewise-linear, non-affine selected-to-physical
  map. The head's local stride approximation, center sampling, regression-range
  assignment, normalized offsets and IoU/GIoU objective do not guarantee exact
  loss equality with the legacy selected-axis parameterization.
- Commit `ce5d03ebf` introduced the `1e-4` loss/proposal and `1e-6` target/score
  tolerances without a registered derivation, null-repeat calibration or
  FP16/FP32 error study. They are engineering tolerances, not validated
  scientific thresholds.
- The failure artifact omits full versus short-padded provenance, offending loss
  key, error magnitude, applicable threshold and FP32 replay. It cannot
  distinguish numerical miss, semantic implementation bug and over-strong gate
  premise.
- Corrected verdict:
  `gate_failed_closed / root_cause_not_identified / gate_validity_under_review`.
  No performance number or paper claim is authorized.

## 2026-07-29 — Gate contract root-cause class established; diagnostics committed

- The result is in post-failure scientific-contract analysis, not model
  performance analysis. No paper-admissible empirical result exists.
- The universal gate premise is now rejected as a theorem for the implemented
  general case. At `T=768, K=384`, round-half-to-even integer anchors produce
  382 steps of two and one step of three, hence a non-affine coordinate warp.
  The physical head's target assignment and IoU/GIoU loss are not guaranteed to
  be scalar-loss conjugate to the selected-axis formulation under that warp.
- The exact observed component in job `1201417` remains unresolved. Its
  immutable log records no window role, loss key, error magnitude, threshold
  comparison or FP32 replay. The correct state is
  `universal_loss_equivalence_premise_invalid /
  observed_mismatch_component_unresolved`.
- Diagnostic-only commit
  `69136de3ed8d8f977c78cfe5258dae3d57f7e238` records affine applicability,
  per-loss errors, unchanged tolerance, AMP state and a separate diagnostic-only
  FP32 replay; failures produce exclusive JSON outside the Git worktree.
- The commit changes no model, loss, budget, threshold, data, checkpoint,
  metric, or admission outcome. Compilation, `git diff --check`, and the
  focused diagnostic/evidence suite passed with `32 passed`.
- The diagnostic commit and research analysis were pushed to
  `codex/duca-rime-20260727`; direct remote-ref verification confirmed the
  published branch identity before preparing the Pro review.
- One bounded Pro review is now warranted to select the paper architecture and
  replace the invalid universal equality premise with a justified scientific
  gate. No new transaction or Phase 4 access is authorized by this analysis.

## 2026-07-29 — Pure selected-axis architecture accepted and acquisition-v2 implemented

- Fully read and registered `U-PRO-PURE-PLUGIN-1`. Accepted its
  `CONDITIONAL GO`: organizational Route C, Route A as the only paper mainline,
  physical-time head as an isolated integration/diagnostic arm, and replacement
  of the invalid general loss-equivalence gate.
- Implemented selected-axis GT mapping, standard-head restoration,
  exact-once proposal remap before official NMS, full implemented-map
  diagnostics, and selected-axis ActionFormer/TriDet configs. The existing
  fully gated `AnchorFreeHead` was not split into subclasses because the
  selected-axis route never enables its physical branch; subclass extraction
  is deferred until a physical integration experiment is separately
  authorized.
- Implemented a Slurm/CUDA acquisition-v2 runtime producer. It strict-loads
  real dense checkpoints into standard and selected heads, checks full/short
  exact-K execution, GT roundtrip, NMS ordering, AMP nulls and complete
  state/RNG/debug restoration.
- Removed external JSON finalization. The runtime producer now writes the final
  finite, exclusive, content-bound receipt directly. Verification re-hashes
  config, checkpoint, data, split, code-gate, calibration, scientific protocol
  and NI-margin-source artifacts and re-binds exact Git/runtime identity.
- Added pre-candidate scientific-protocol anchoring to a clean commit/tree,
  fresh candidate output root, training/calibration NI-margin source, endpoint,
  multiplicity, guardrails and stopping rules. Phase 1, Phase 3 and all Phase 4
  entrypoints require the same verified admission-v2 prerequisite.
- Local compilation, Bash syntax and `git diff --check` passed. The pure
  admission/geometry/stage/launcher suites plus mandatory C3 regressions passed
  `95 tests`; one warning is unrelated. Independent read-only audits found no
  P0 in the selected-axis or admission chain.
- The Torch model suite remains `remote_torch_pending`: the Windows host fails
  loading PyTorch `c10.dll` with WinError 1114. This is not model evidence.
- Current state is `implemented/local_tested/remote_pending`; no experiment has
  run, no performance value was interpreted, Phase 4 and official-final remain
  sealed, and no paper-admissible empirical conclusion exists.
