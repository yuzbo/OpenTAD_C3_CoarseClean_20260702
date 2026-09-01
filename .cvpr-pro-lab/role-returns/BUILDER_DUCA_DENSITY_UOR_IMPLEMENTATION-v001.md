# BUILDER_DUCA_DENSITY_UOR_IMPLEMENTATION-v001

```yaml
receipt_version: BUILDER_DUCA_DENSITY_UOR_IMPLEMENTATION-v001
message_id: msg-20260813T234345Z-df5644375e10
role: Builder
task_id: 019ff799-01e7-7270-a047-d101a94bb778
status: BUILDER_IMPLEMENTATION_COMPLETE_FOR_CRITIC_PRE_RUN_REVIEW
next_owner: Critic
scientific_state: BLOCKED_PRE_RESULT
gate_execution_state: NOT_EXECUTED
decision: NOT_APPLIED
```

## Immutable binding

- Worktree: `C:/Users/skywalker/.codex/worktrees/8712/OpenTAD_C3_CoarseClean_20260702-builder-a6bdc084`
- Frozen parent: `a6bdc084cc145c80b6b2c68d0a38f0deea3e8518`
- Builder implementation commit: `7f07e4545fafda5ca9b86ead14a089b3515a06d0`
- Commit subject: `Implement DUCA density reachability pre-run package`
- Post-commit porcelain count: `0`
- New-file count: `6`
- Untracked prototype runtime/import/copy dependency: `false`

## Changed tracked paths

Modified:

1. `opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py`
2. `opentad/models/detectors/actionformer.py`
3. `opentad/models/detectors/base.py`
4. `research-wiki/experiments/duca-total60-prebackbone-plugin-cvpr.md`
5. `research-wiki/log.md`

New:

1. `configs/adatad/thumos/duca_density_detector_fit_n16r4.py`
2. `configs/adatad/thumos/duca_density_reader_fit_n16r4.py`
3. `configs/adatad/thumos/duca_density_cal_uor_n16r4.py`
4. `tools/bata/duca_density_reachability.py`
5. `scripts/run_duca_density_reachability_n16r4.sh`
6. `tests/test_duca_density_reachability_contract.py`

## Implemented contract

- Added a density-only reader whose learned `duca_density_logits[B,T]` are produced solely from valid-prefix `browser_memory`.
- Added the frozen density-to-trapezoidal-mass, inverse-CDF, fixed-point integer projector path with exact constant-logit canonical-uniform short circuit.
- U/O/R share one production decoder/projector and differ only in density-logit source: U constant, O exact width-16 privileged GT-boundary diagnostic, R learned browser-memory source.
- GT reaches the selector only for O. U/R receive no GT handoff; all arms reject raw-prediction load/save cache paths.
- Added deterministic whole-video FIT/CAL manifest construction and validation, phase-order and identity seals, fresh-output PRE_RUN guard, canonical blocked receipt, U/O/R metric embargo, unchanged pooled OpenTAD evaluator adapter, 10,000-draw paired whole-video PCG64 bootstrap, percentage-point conversion, ranks 500/9500, and the literal frozen ADVANCE/HOLD/KILL rule.
- Added authored-only future detector-FIT, reader-FIT, ordered U/O/R CAL, receipt sealing, and final evaluator commands. The launcher was not invoked.
- Preserved `NOT_EXECUTED`, `PRE_RUN_BLOCKED`, `BLOCKED_PRE_RESULT`, and `decision=NOT_APPLIED` defaults.

## Current-patch identity and optimality receipt

The focused tracked test module passed on the committed source state:

```text
command: conda run -n open_mmlab python -m pytest tests/test_duca_density_reachability_contract.py -q
result: 16 passed, 1 warning in 2.09s
warning: environment NumPy 1.x/2.x API initialization warning; no tested contract failed and the synthetic tests do not use NumPy conversion
```

Bound assertions include:

- `test_constant_logits_are_bit_identical_to_canonical_uniform`: exact integer sequence identity, dtype, and endpoints for `T=768, K=384`.
- `test_current_projector_witness_is_exact_lexicographic_optimum`: exhaustive feasible-set comparison for the current projector at `T=17, K_eff=16`, including the frozen `E2`, `Einf`, `E1`, uniform-displacement, and lexicographic tie order.
- Exact O density reconstruction, reader output-surface restriction, U/R forbidden-signal rejection, O-only GT handoff, raw-cache denial, deterministic disjoint FIT/CAL construction, predecessor sealing, metric embargo, source-only arm delta, pooled bootstrap wiring, percentage-point unit, and literal decision predicates.

Earlier environment/import attempts did not collect tests and are not claimed as verification:

1. Default Windows Python: Torch `c10.dll` load failed with `WinError 1114`.
2. Bundled workspace Python: `No module named pytest`.
3. `open_mmlab` before test-local module isolation: unrelated eager OpenTAD import failed at `mmaction.models.localizers.drn`.
4. `mmaction`: unrelated eager OpenTAD import failed because `mmcv` was absent.

The final tracked test uses a test-local registry shim to load the exact selector and detector-base files without importing unrelated backbones; no production import path was changed for the shim.

Static closure:

```text
git diff --check: PASS
post-commit git status --porcelain=v1: empty
commit parent: a6bdc084cc145c80b6b2c68d0a38f0deea3e8518
```

## Plan deviations

1. `ActionFormer` and `BaseDetector` required narrow existing-file edits because the common detector path consumed and discarded inference `gt_segments` before O could receive its exact current-CAL diagnostic input. The handoff is source-gated to O; U/R receive none.
2. The original CAL config had to set raw-prediction save to false and the detector base now treats the DUCA density selector as a live-reader path, closing the frozen no-cache requirement.
3. The repository memory contract required bounded updates to the existing DUCA experiment node and append-only log. They record only `implemented`/synthetic-contract-tested state and preserve `BLOCKED_PRE_RESULT`.
4. Static inspection found that OpenTAD evaluator values are fractions while the frozen protocol thresholds are percentage points. The implementation converts all four point/bootstrap differences by `100.0` before applying the frozen thresholds.

No scientific scope, arm, threshold, seed, statistic, evaluator semantics, split rule, or runtime authority changed.

## Prohibited surfaces

```yaml
data_or_official_validation_accessed: false
data_root_listed: false
fixture_materialized: false
gpu_cuda_slurm_used: false
remote_invoked: false
training_run: false
real_video_inference_run: false
evaluation_or_metric_run: false
gate_run: false
browser_sources_pro_used: false
claim_promoted: false
```

## Terminal handoff

Builder work is complete only as an immutable tracked PRE_RUN implementation package. Critic owns independent review of commit `7f07e4545fafda5ca9b86ead14a089b3515a06d0`. No gate, FIT, CAL, inference, evaluator, bootstrap, metric, decision, result promotion, or later phase is authorized by this receipt.

BUILDER_TERMINAL: `IMPLEMENTATION_COMPLETE_BLOCKED_PRE_RESULT_NEXT_OWNER_CRITIC`
