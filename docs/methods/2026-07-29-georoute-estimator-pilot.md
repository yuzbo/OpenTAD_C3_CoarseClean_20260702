# GeoRoute estimator/representation exploratory pilot v1

Date: 2026-07-29

Study ID: `georoute_estimator_representation_pilot_v1`

Status: `implemented_pending_remote_test_and_p0`

Authorization boundary: one development-only exploratory seed. This pilot has
no automatic winner, selector promotion, P2/P3, official test, confirmatory
margin, efficiency claim, Geometry Zoom claim, or paper claim.

## Scientific question

The D/K/M preexperiment passed and showed that ordered Plackett-Luce can assign
direct credit to unselected hard-route logits, that the three representation
channels can be isolated, and that route telemetry is prediction-neutral. The
next question is whether estimator choice, token support, or geometry
representation explains detector utility.

The pilot changes one scientifically interpretable factor at a time. It does
not implement dynamic context/geometry/residual roles.

## Frozen common protocol

- source-native tubelet support with no resize or local crop;
- exactly one heavy VideoMAE forward;
- exact unique `K=64`, zero duplicates;
- seed `3407`;
- 20 epochs;
- identical development fit/gate manifest, annotation, class map, video root,
  pretrained initialization, optimizer, scheduler, augmentation, detector,
  post-processing, and final-checkpoint policy;
- `absolute_position_enabled=true` for the pretrained VideoMAE in every arm;
- `uniform_selected` pooling;
- policy temperature `0.7`;
- score-function weight `1.0`, EMA baseline momentum `0.95`;
- geometry smoothness and area-prior weights `0`;
- all three new representation switches explicit;
- final EMA checkpoint only;
- full development telemetry and model/postprocess, loader-wait, window-wall,
  and peak-memory diagnostics;
- no raw-prediction cache, GT/teacher/oracle route input, manual ROI, or
  official test.

## Six arms

| Arm | support | estimator | abs / ROI-rel / geometry projection | learned geometry side channel |
| --- | --- | --- | --- | --- |
| `residual_st_rep_off` | residual-only free exact-K | ST | off / off / off | no |
| `residual_pl_rep_off` | residual-only free exact-K | PL | off / off / off | no |
| `fixed_rep_off` | deterministic uniform exact-K | none | off / off / off | no |
| `fixed_rep_on` | deterministic uniform exact-K | none | on / on / on | yes; representation only, never selection |
| `roi_pl_rep_off` | learned ROI exact-K | PL | off / off / off | learned for support only |
| `roi_pl_rep_on` | learned ROI exact-K | PL | on / on / on | learned for support and representation |

The fixed-on arm keeps the same deterministic uniform membership as fixed-off.
Its geometry head can affect the detector only through the enabled
representation path. The ROI off/on pair shares the same stochastic support
mechanism and differs only in detector-visible representation.

## Preregistered contrasts

1. Estimator:
   `residual_pl_rep_off − residual_st_rep_off`.
2. Representation with fixed support:
   `fixed_rep_on − fixed_rep_off`.
3. Representation with learned ROI support:
   `roi_pl_rep_on − roi_pl_rep_off`.
4. Support under PL and representation-off:
   `roi_pl_rep_off − residual_pl_rep_off`.

Each contrast reports Avg-mAP, mAP@0.3--0.7, the high-IoU composite
`mean(mAP@0.6,mAP@0.7)`, full diagnostic cost fields, and route telemetry.
No post-result `+0.50/+0.30 pp` threshold is used.

## DAG and fail-closed rules

The independent DAG is:

1. six parallel CUDA one-step P0 leaves;
2. an afterok P0 finalizer that validates exact arm bindings, detector
   gradients, unique rendezvous namespaces, final-only storage bounds, and the
   sealed D/K/M parent;
3. six parallel 20-epoch training/evaluation leaves after the P0 suite;
4. an afterany exploratory finalizer.

All P0 leaves are submitted held and released only after immutable deployment
and finalizer receipts exist. Every training leaf uses a unique Slurm job,
torchrun rendezvous ID, cell namespace, bound config, and final checkpoint.
The pilot is additionally bound to the exact D/K/M runtime commit, source
experiment commit, and finalization SHA. Each training result carries its full
immutable binding; the finalizer rereads the raw profile and telemetry,
recomputes window, population, and summary hashes, requires the canonical P0
suite path, and verifies that train and test share the expected Slurm leaf.

Any decode error, OOM, non-finite loss/cost, non-finite required gradient,
missing update, rendezvous failure, missing artifact, input/hash mismatch,
duplicate checkpoint, or partial namespace invalidates that arm. It is not
resumed and no partial checkpoint is interpreted as performance.

The finalizer emits only:

- `PILOT_COMPLETE_NO_PROMOTION`, with descriptive contrasts; or
- `PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`.

## Conditional research decision after the pilot

- If PL improves residual support while fixed representation has little
  effect, prioritize estimator calibration and then freeze disjoint
  confirmatory seeds.
- If fixed representation helps, the previous geometry gain was at least
  partly representation-mediated; future support claims must keep that path
  matched.
- If ROI support beats residual support with representation off, continuous
  geometry is a viable routing hypothesis.
- If ROI representation adds a further benefit, a later model may combine
  geometry support and geometry-conditioned features as two separately
  ablated components.
- If neither estimator nor ROI support helps, do not build full CER; pivot to
  boundary-conditioned evidence or another source-native selector.

Only after a promising contrast is observed may pilot variance inform a new,
disjoint multi-seed confirmatory protocol. Geometry Zoom remains conditional
because this pilot never resamples a denser continuous source-coordinate crop.
