# Research Log

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
  The expanded focused non-Torch suite now passes 64 tests. A fresh audit on the
  corrected clean commit remains mandatory before deployment.
