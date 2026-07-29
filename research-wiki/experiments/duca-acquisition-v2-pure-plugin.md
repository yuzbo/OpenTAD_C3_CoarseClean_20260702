# DUCA acquisition-v2 pure-plugin closure

## Status

- Date: `2026-07-29`
- Source decision: `U-PRO-PURE-PLUGIN-1`, audited by `U-PRO-ADMISSION-V21-1`
- User authorization: `approved_for_direct_implementation`
- Architecture: `selected_axis_pure_pre_backbone_plugin`
- Implementation: `implemented`
- Local verification: `static_and_pure_python_tested`
- Runtime verification: `authoritative_code_gate_passed_1204067`
- Admission v2 formal path: `disabled / historical_read_only_or_engineering_fixture`
- Admission v2.1: `stage0_fail_closed_slice_implemented / full_design_blocked`
- Numeric calibration: `not_started / old_implementation_disabled`
- Data feasibility: `failed_for_proposed_per_video_full_plus_short_contract`
- Experiment: `not_yet_run`
- Empirical support: `none`
- Paper claim: `forbidden`
- Phase 4: `sealed`
- Official-final: `sealed`
- Phase-4 training scope: `100-of-200 development role / not official-comparable / entry hard-disabled`

The first authoritative preflight was Slurm job `1204048` on exact commit
`70cf49de82a9d0ed889ed94af9604edd61070e55`. It executed the full Linux/Torch
suite: 247 tests passed and one stale Phase-1 test still expected the
superseded physical-head protocol. No gate receipt was produced and no numeric
calibration, scientific admission, Phase 1, or official-final work started.
Failure signature:
`phase1_uniform_test_expected_superseded_physical_protocol`. The remediation
updates only that regression expectation and asserts selected-axis/standard-head
state; the focused local test passes.

Remediation commit `119db2f83756281729506632a18bfed607794d13`
passed Slurm code-gate job `1204067`; gate-receipt SHA-256 is
`d664a619007f1cafbd4e52f2fd6a053fb0e3b5336dcb2be2b16302912286e5c8`.
The next layer did not start. Audit showed that the current calibrator uses
synthetic head fixtures, does not consume registered training/calibration
videos, lacks video/process-grouped null distributions and independent process
launches, and does not freeze the specified normalized statistic plus
denominator floor. No admissible non-fixture targets, acquisition data manifest,
or NI-margin source is currently registered. These are P0 prerequisites for
Admission v2, not optional polish.

The full follow-up audit `U-PRO-ADMISSION-V21-1` confirmed all six gaps and
added cloned-head tautology, absent full-model train/backward coverage, absent
immutable raw distributions, weak self-declared schemas, unverifiable
development non-access and absent failure receipts. Its preservation of the
selected-axis architecture and real-video/full-model/disjoint-holdout v2.1
direction is accepted.

It is not accepted verbatim. A read-only check of split manifest
`41349cd39a6a550b6e1613de968577b1605c93902edd52a88309121b9e90c057`
and immutable annotation metadata found 100 eligible source videos: 70 produce
only natural full windows, 30 only natural short windows and zero produce both
under the current production enumerator. Short strata contain 7/13/10 videos,
so the proposed per-video full-plus-short rule and minimum-eight first stratum
are impossible. The proposed crossed catastrophic bound is also undefined,
`2 * reporting quantum` is not a scientific NI justification, and hard
network/mount/object-lock claims exceed what repository-level Slurm code can
enforce.

## Accepted scientific decision

The Pro report's core decision is accepted:

1. Organizational Route C is retained, but Route A is the only paper mainline:
   exact-K acquisition occurs before an unchanged selected-axis detector.
2. Dense physical GT is mapped to selected coordinates for training; standard
   ActionFormer/TriDet target assignment, loss and decode remain unchanged.
3. Selected-axis proposals are mapped exactly once to physical time before the
   registered NMS/evaluator.
4. The physical-time head is a separately named detector-integrated
   experimental/diagnostic route and cannot support the pure-plugin or
   universal detector-agnostic claim.
5. The general physical-vs-selected scalar-loss-equivalence admission premise
   is abolished. It survives only as a no-admission diagnostic where its
   applicability is explicit.
6. Replacement admission has three layers: structural/coordinate correctness,
   training-calibrated production-AMP numeric nulls, and a preregistered
   held-out same-total-cost scientific gate.
7. `DUCA-P1-V2-PURE-PLUGIN-CLOSURE` remains the next model-facing experiment,
   but it is not authorized until Admission v2.1 passes. Learned H-RIME,
   Phase 4, official-final, and performance claims remain unauthorized.

## Independent implementation adjudication

The report is not accepted verbatim in three implementation details:

1. The current shared `AnchorFreeHead` fully restores standard behavior when
   `physical_grid_actionformer` is absent. Selected-axis configs remove that
   option and runtime admission rejects any enabled physical path. Immediate
   subclass extraction would enlarge risk without helping Phase-1 v2, so it is
   deferred until a separately claimed physical-integration experiment is
   authorized.
2. Implemented-map diagnostics mirror `TrueTimeMap`, including optional left
   and right half-open knots; they are not limited to appending only `K -> T`.
3. Legacy-v1 reproduction remains immutable diagnostic history and is not a
   dependency for candidate work. Numeric thresholds and the scientific
   protocol must nevertheless be frozen before any candidate development
   result is read.

## Implemented surface

- Selector modes now distinguish `selected_axis_plugin` from
  `physical_head_integration`.
- Selected-axis training emits the complete inverse map and remaps GT
  true-time to selected-time.
- ActionFormer, TriDet and generic selected-axis SingleStage paths enforce a
  dynamic temporal bucket and aligned `[B,K]` mask.
- Both detector post-processing paths remap selected proposals to physical time
  exactly once before official NMS.
- New selected-axis ActionFormer/TriDet configs restore the standard head and
  explicitly declare the pure-plugin claim boundary.
- Full implemented-map geometry, finite exclusive atomic evidence, full
  module/RNG/debug replay restoration and legacy-v1 diagnostic isolation are
  implemented.
- The old Slurm/CUDA runtime path can still load ActionFormer/TriDet heads for a
  random-feature engineering fixture, but its rows are explicitly
  `engineering_fixture` with no admission effect.
- Formal v2 calibration/admission invocation now fails closed, and an old v2
  receipt is accepted only under an explicit historical-read-only parser mode.
  Production entrypoints therefore cannot use v2 to authorize Phase 1/3/4.
- A pure metadata v2.1 feasibility auditor mirrors the production sliding
  enumerator, content-binds its result, and records the impossible proposed
  window contract without consuming decoded frames or candidate output.
- A separate official-comparability audit found that the current Phase-4
  trainer uses only the 100-video `detector_selector_train` role. Its full
  evaluation path is correctly bound to OpenTAD's registered 211-video set,
  but its training result is development-only. The Phase-4 cell entry is now
  hard-disabled pending a full-200-video, global-batch-2 refit and complete
  out-of-fold targets.

## Verification receipt

- Python compilation: passed.
- Bash syntax: passed.
- `git diff --check`: passed.
- Current Admission-v2.1/launcher/official-schedule focused suite: `66 passed`,
  one unrelated deprecation warning.
- Broad non-Torch contract suite including mandatory C3 regressions:
  `219 passed, 3 skipped`, one unrelated deprecation warning.
- Independent selected-axis audit: no P0; exact-K/no-padding, GT remap,
  map-before-NMS exactly once, standard-head restoration and physical-route
  isolation verified.
- The earlier evidence-chain audit found no random-hash/manual-boolean bypass
  within the then-implemented v2 path; the later v2.1 audit supersedes any
  broader interpretation and identifies the current formal blockers above.
- Independent current-diff audit found no P0/P1 authorization bypass; its one
  P2 relative-annotation-path issue was corrected and covered by regression.
- The full Torch suite cannot collect on the Windows host because the installed
  PyTorch fails to load `c10.dll` with WinError 1114. The exact clean-commit
  Linux/PyTorch Slurm code gate is still required before runtime completion;
  this environment error is not a model result.

## Superseded DAG and current stop

The earlier v2 calibration → admission → Phase-1 DAG is superseded. Current
execution is:

1. Commit and code-gate the old-v2 fail-closed/fixture-only correction and
   metadata feasibility auditor.
2. Freeze corrected role-level window coverage, exact crossed statistics,
   scientifically justified NI margin and enforceable isolation contract.
3. Implement the complete real-video v2.1 evidence chain only after step 2.
4. Run no calibration and no Phase 1 before a verified v2.1 receipt exists.

No Phase 2/3/4 job is authorized by this node. No performance number from these
steps may enter the paper.
