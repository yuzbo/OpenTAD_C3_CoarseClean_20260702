# ActionFormer SparseHead DCSR Official-Comparable Preregistration

Date frozen: 2026-07-30

Status: `tested`; G1 gate failed and this preregistered route is terminated

Implementation status: `implemented` and `tested`

Experiment status: G0 passed; validation-only G1 Job `1206273_[0-2]`
completed and failed the frozen non-inferiority gate

Paper status: `paper_ready=false`

## Decision

The hard K384 method is terminated. At protocol freeze, the only retained
sparse-head direction was Dense Cheap Scaffold + Sparse Expensive Refinement
(DCSR). DCSR was required to decouple
compute sparsity from proposal and supervision support:

1. a cheap dense scaffold runs at every valid native FPN query and predicts
   base objectness/class, coarse boundary offsets and support;
2. a selector allocates expensive refinement only to a subset of queries;
3. selected queries receive residual heavy refinement;
4. unselected queries keep their dense scaffold outputs rather than being
   zeroed or removed;
5. selected residuals are scattered back to the full native query grid before
   the unchanged official decoder and Soft-NMS;
6. dense scaffold supervision remains available at all valid queries; sparse
   refinement loss cannot redefine the official positive normalizer for the
   scaffold.

This is a new method, not a rescue reinterpretation of Job `1205599`.

## Frozen G0/G1 implementation identity

The implemented branch/commit/tree is
`codex/actionformer-dcsr-g0-g1-20260730` /
`bf0df83d7400c89fc61f38d169d68085420a2263` /
`2f9346fcfd2bfb7fc5a76a86ef65545030a67469`.

G0 and G1 use different scaffold contracts:

- G0 `official_identity` retains the complete official dense head and disables
  the residual path. It is only an exact routing/geometry/decoder gate and
  cannot support a cheap-head or efficiency claim.
- G1 `cheap_dense_scaffold` uses the frozen one-layer dense scaffold at every
  valid query and a three-layer signed residual refinement at uniform K384.
  Full-grid masks, targets, supervision and normalizer remain unchanged.

The validation-only manifest SHA-256 is
`ba683bc5ddbb1fe219fab0545e9d808808d9b25fc9b32e7c5c0b6339b68b9bbb`;
the frozen development seeds are
`2026073001/2026073002/2026073003`.

Real-CUDA G0 Job `1206168` completed `0:0`; its receipt SHA-256 is
`b87fc59ec6529e83e99f7bf5fbfb7f3bff5ec637060c62057da07a669a8c1ff4`.
All state-key, point, full-mask, pre-decode tensor and final official
Soft-NMS/timestamp outputs are exact. Formal validation-only G1 array
`1206273_[0-2]` completed `0:0` for all three frozen development seeds. Its
paired aggregate failed every frozen continuation bound; the exact result and
post-hoc analysis are recorded in
`actionformer-sparsehead-dcsr-g1-negative-analysis-20260730.md`.

## Frozen hypotheses

- H-DCSR-1: keeping a dense proposal/support floor removes the catastrophic
  K384 proposal-reachability ceiling.
- H-DCSR-2: sparse heavy residual refinement can reduce measured
  feature-to-detection detector cost without exceeding the frozen accuracy
  non-inferiority bounds.
- H-DCSR-3: per-FPN minimum coverage and a length-adaptive residual budget are
  necessary to avoid the 4--16 second/coarse-level failures observed in S0.

## Information boundary

- No THUMOS test GT, test AP, teacher test predictions or cached test
  counterfactuals may influence selector decisions, budget, thresholds,
  checkpoints or stopping.
- Architecture, loss weights, per-level floors, selector and budget are tuned
  only on a deterministic internal holdout carved from the official
  `validation` training split. After freezing, the final models retrain on the
  full official `validation` split and evaluate the `test` split once per
  preregistered seed.
- The released ActionFormer `66.833392` is contextual. Every causal delta uses
  a same-seed, same-commit, same-environment dense control.

## Minimal internal ladder

These rows are method-development gates and are never paper performance rows:

| gate | change | purpose | stop condition |
|---|---|---|---|
| G0 | dense scaffold with residual path disabled | exact native-grid geometry and decoder equivalence | any raw-tensor, timestamp, evaluator or output mismatch |
| G1 | dense scaffold + uniform heavy refinement | test whether dense support alone removes S0 collapse | internal holdout Avg delta below -0.5 pp or @0.6/@0.7 below -1.0 pp |
| G2 | learned residual selector at matched heavy budget | test selector value over uniform | no accuracy/cost benefit over G1 |
| G3 | remove per-FPN floors | isolate level-coverage necessity | retain floors unless removal is non-inferior in all preregistered duration bins |
| G4 | frozen budget curve on internal holdout only | choose one final compute point | no test-set selection; one budget is frozen before official runs |

Internal gates use at least three development seeds. They may choose the final
architecture and budget but cannot appear as official test performance.

## Final official-comparable study

After G0--G4 freeze:

- paired seeds:
  `1234567891`, `2234567891`, `3234567891`, `4234567891`, `5234567891`;
- dense and DCSR arms start from scratch under the same candidate commit,
  official ActionFormer THUMOS I3D data, 5-warmup + 30-optimizer-epoch
  schedule, terminal epoch-35 EMA, runtime, official seven-argument Soft-NMS
  and pinned independent evaluator;
- primary metrics: Avg-mAP and mAP@0.3/0.4/0.5/0.6/0.7;
- secondary diagnostics: class, duration, boundary error, proposal recall and
  calibration, all frozen before viewing test results;
- report paired per-seed deltas, mean, standard deviation and a paired
  bootstrap 95% confidence interval.

Main-table accuracy continuation requires all of:

1. mean DCSR-minus-dense Avg-mAP `>= -0.50 pp`;
2. mean deltas at mAP@0.6 and mAP@0.7 each `>= -1.00 pp`;
3. paired 95% CI lower bounds `>= -1.00 pp` for Avg and `>= -1.50 pp` for
   mAP@0.6/0.7;
4. no identity, data, schedule, checkpoint, evaluator, non-finite-loss or
   result-receipt failure.

## Synchronized cost contract

Cost is measured on the same allocated GPU type, software environment, batch
size and evaluated videos for dense and DCSR. Report separately:

1. feature-file load and preprocessing;
2. host-to-device transfer;
3. dense scaffold;
4. selector;
5. sparse heavy refinement;
6. scatter/reconstruction;
7. unchanged decoder/head work;
8. official Soft-NMS/post-processing;
9. complete feature-to-final-detection wall time;
10. peak GPU memory, MACs/FLOPs and energy when available.

Use CUDA synchronization, warmup, repeated runs, median/p5/p95 and paired
bootstrap confidence intervals. The paper claim boundary is explicitly
“precomputed-I3D feature-to-detection efficiency”; it must not be expanded to
video-backbone or end-to-end raw-video efficiency unless those stages are
separately measured.

An efficiency claim requires:

- median complete detector-pipeline speedup `>= 1.05x`;
- 95% CI lower bound of speedup `> 1.00x`;
- the accuracy conditions above;
- no hidden exclusion of scaffold, selector, scatter, decoder or NMS cost.

## Kill and fallback rules

- If G1 still collapses, DCSR does not solve the dominant failure and the
  SparseHead route is terminated.
- If accuracy passes but synchronized cost does not, DCSR is an accuracy-safe
  architectural ablation, not an efficiency method.
- If cost passes but accuracy does not, no positive claim is allowed.
- No threshold, budget, NMS, seed or checkpoint may be changed after official
  test results are visible.
- Any further method redesign requires a new preregistration and cannot
  overwrite the S0 negative or this protocol.

## Frozen resolution

G1 is a legal validation-only method-kill result. Across the three paired
development seeds, dense and DCSR mean Avg-mAP are `0.5680730871` and
`0.4925110665`; DCSR-minus-dense is `-7.556202 pp`. Mean deltas at
mAP@0.6/mAP@0.7 are `-11.043134/-11.019821 pp`. All three seeds and all five
tIoU thresholds are negative.

These values violate the G1 bounds by a wide margin. Under the frozen kill
rule, G2--G4 and the five-seed official study are not authorized. The status
transition is:

- implementation: `implemented` -> `tested`;
- exact G1 hypothesis: rejected with `empirically_supported` negative evidence;
- route: terminated at G1;
- paper status: `paper_ready=false`.

The G1 numbers are not official THUMOS test results and cannot be compared
numerically with historical `63.xx`, the released ActionFormer `66.833392`, or
the same-run official S0 dense `66.583013`. They are paired internal-holdout
method-selection evidence only. Any successor architecture is a new route and
requires a new preregistration before training.
