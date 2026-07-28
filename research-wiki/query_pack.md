# DUCA-RIME Current Query Pack

Last updated: `2026-07-28`

## Current decision

The user approved direct execution of the four-stage DUCA-RIME adjudication.
The route is an **offline TAD pre-backbone acquisition plugin**. It is not
Online TAD, and it is not yet the paper's final method.

Current evidence level:

| Item | State |
|---|---|
| Scientific route | `user_approved` |
| Four-stage implementation | `implemented` |
| Focused local checks | `tested` |
| Remote authoritative code gate | `passed` |
| Dense reference training | `training_loop_finished_but_jobs_failed_at_checkpoint_compaction` |
| Phase 1 closure | `failed_closed_on_no_padding_ledger` |
| Phase 2/3/4 | `blocked_by_failed_dependency` |
| Latest four-stage transaction | `failed_closed_without_terminal_receipts` |
| H-RIME scientific route | `user_approved / designed` |
| Stage-0 repair implementation | `implemented / local_non_torch_tested / remote_pending` |
| H-RIME deterministic core | `implemented / local_non_torch_tested` |
| H-RIME Stage-1 oracle/evaluation surface | `implemented / local_non_torch_tested / remote_torch_pending` |
| H-RIME shared-scan/model integration | `not_yet_implemented` |
| H-RIME same-total-cost oracle | `not_yet_run` |
| H-RIME Stage-0 recovery transaction | `submitted / scheduler_pending` |
| Paper evidence contract | `user_frozen` |
| DUCA-RIME empirical superiority | `not_yet_empirically_supported` |
| Paper-ready method | `not_yet_paper_ready` |

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

The repaired Stage-0 recovery transaction is now the active scheduler target:

- exact deployment commit:
  `902168a12bc92babd62b6cb1877ce7137f56cea0`;
- root:
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_recovery_902168a1_20260728_183709`;
- submission-manifest SHA-256:
  `fd6fef65ac01e7830c6b5e337684b19a3bad65c1432f819cfecb32e83dfefb85`;
- jobs: code gate `1199978`, Phase 1 `1199979`, dense ActionFormer salvage
  `1199980`, dense TriDet salvage `1199981`, Phase 2 `1199982`, and Phase-3
  controller `1199983`;
- first monitor snapshot: code gate `PENDING (Priority)`, all downstream jobs
  dependency-pending, no terminal receipts;
- dense recovery claim scope:
  `engineering_dense_reference_recovery_not_method_evidence`;
- Phase 4 disabled; official-final sealed.

This status is `ENGINEERING_STATUS`, not an empirical result.

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
   saved-prediction and official-evaluator receipts, but that path has not yet
   executed remotely. It does **not** yet have a connected learned video-budget
   head, an executing shared-video scan, the grouped training path, or
   calibration evidence.

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
`remote_torch_pending`; the learned/shared-scan detector path is not yet
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
