# DUCA acquisition-v2 pure-plugin closure

## Status

- Date: `2026-07-29`
- Source decision: `U-PRO-PURE-PLUGIN-1`
- User authorization: `approved_for_direct_implementation`
- Architecture: `selected_axis_pure_pre_backbone_plugin`
- Implementation: `implemented`
- Local verification: `static_and_pure_python_tested`
- Runtime verification: `authoritative_code_gate_passed_1204067`
- Numeric calibration: `blocked / implemented_draft_not_protocol_valid`
- Experiment: `not_yet_run`
- Empirical support: `none`
- Paper claim: `forbidden`
- Phase 4: `sealed`
- Official-final: `sealed`

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
7. The only authorized next experiment is
   `DUCA-P1-V2-PURE-PLUGIN-CLOSURE`; learned H-RIME, Phase 4,
   official-final, and performance claims remain unauthorized.

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
- The Slurm/CUDA runtime gate strictly loads real ActionFormer/TriDet
  checkpoints into selected and standard heads, derives fixture levels from
  actual prior strides, covers full and short exact-K windows, validates GT
  roundtrip/NMS order, and runs production-AMP nulls.
- Runtime admission directly writes the final immutable content-bound receipt.
  A generic external JSON finalizer was removed.
- Admission verification re-hashes every bound artifact and recomputes Git,
  runtime, calibration, scientific-protocol, margin-source and numeric-null
  identities. Phase 1, Phase 3 and every Phase 4 entry require this receipt.

## Verification receipt

- Python compilation: passed.
- Bash syntax: passed.
- `git diff --check`: passed.
- Pure admission/geometry/stage/launcher tests plus mandatory C3 regressions:
  `95 passed`, one unrelated deprecation warning.
- Independent selected-axis audit: no P0; exact-K/no-padding, GT remap,
  map-before-NMS exactly once, standard-head restoration and physical-route
  isolation verified.
- Independent evidence-chain audit: no random-hash/manual-boolean admission
  bypass after artifact re-hashing and direct runtime finalization.
- Torch model tests on the Windows host: not executed because the installed
  PyTorch fails to load `c10.dll` with WinError 1114. This is an environment
  limitation, not a passing or failing model result.

## Frozen next DAG

1. Exact clean commit and remote checkout.
2. Authoritative Linux/PyTorch Slurm code gate.
3. Training/calibration-only ActionFormer/TriDet production-AMP null
   calibration.
4. Pre-candidate scientific protocol freeze from a registered NI-margin source.
5. Admission-v2 runtime producer and immutable terminal receipt.
6. Phase-1 v2 full registered execution/geometry/cost closure.
7. Stop closed on any missing receipt or scientific/identity drift.

No Phase 2/3/4 job is authorized by this node. No performance number from these
steps may enter the paper.
