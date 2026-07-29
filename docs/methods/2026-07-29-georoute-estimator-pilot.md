# GeoRoute estimator/representation exploratory pilot v1

Date: 2026-07-29

Study ID: `georoute_estimator_representation_pilot_v1`

Status: `experiment_running_with_residual_pl_amp_hard_fail_pending_closeout`

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
2. an afterany P0 finalizer that validates exact arm bindings, detector
   gradients, unique rendezvous namespaces, final-only storage bounds, and the
   sealed D/K/M parent;
3. six parallel fail-closed wrappers afterany the P0 finalizer; each wrapper
   must validate the sealed PASS P0 suite before creating a cell, and only then
   may run the 20-epoch training/evaluation leaf;
4. an afterany exploratory finalizer over all terminal leaves.

All P0 leaves are submitted held and released only after immutable deployment
and finalizer receipts exist. Every training leaf uses a unique Slurm job,
torchrun rendezvous ID, cell namespace, bound config, and final checkpoint.
The pilot is additionally bound to the exact D/K/M runtime commit, source
experiment commit, and finalization SHA. Each training result carries its full
immutable binding; the finalizer rereads the raw profile and telemetry,
recomputes window, population, and summary hashes, requires the canonical P0
suite path, and verifies that train and test share the expected Slurm leaf.

This all-terminal dependency shape is a mechanical closeout guarantee, not a
relaxation of P0. If any P0 report is missing or invalid, the P0 finalizer seals
a mechanical failure receipt and exits nonzero; all six wrappers then fail
before cell creation because no PASS suite exists; and the finalizer emits
`PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`. It avoids a permanent
`DependencyNeverSatisfied` chain while never authorizing training after failed
P0. P0-finalizer and final-closeout prevalidation or sealing failures also
write hashed fail-safe receipts before re-raising, so a terminal mechanical
record survives a nonzero control Job.

The first runtime `02b6efe7` is retained as immutable deployment-failure
evidence. P0 Jobs `1203380`--`1203385` failed before model reports due to
script-mode import and under-instrumented readiness failures; no training ran.
Finalizer `1203393` separately exposed JSON-key-order validation. The repair
uses module-mode P0, source-root bootstrap, a Slurm-job-scoped `127/8` endpoint
with a kernel-assigned port and unique ID, a 120-second readiness bound with a
hashed failure sidecar, whole-process-group cleanup on failure, same-leaf node
revalidation, and key-order-independent exact arm binding. Same-node
Jobs `1203460/1203461` demonstrated that N16R4 cannot provide
`srun --resv-ports=2`, so reserved ports are not part of this protocol.

The fresh runtime is
`cbe0a08218a2f4550960f7c832f88c8cf77757c1`, synced through the academic
proxy to the exact clean snapshot
`/data/run01/sczc063/yuzibo/projects/opentad_georoute_cbe0a082_20260729_pilotrepair`.
All GeoRoute and required C3 remote Linux tests passed `118/118`. Independent
same-node rendezvous Jobs `1203689/1203690` then ran concurrently on `g0005`
and completed `0:0`; their job-scoped hosts were `127.19.167.237` and
`127.19.167.238`, and all four observed TCPStore ports were distinct.

The fresh pilot namespace is
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_estimator_representation_pilot_cbe0a082_20260729_1849`.
P0 Jobs `1203707`--`1203712` and afterany finalizer `1203713` all completed
`0:0`. The sealed P0 suite has internal SHA-256
`00b7c0e3251f3d384df91cf900267694918d1245b4a5803150e8e2e1465210d2`
and file SHA-256
`6e33c0e863f3c2cf2802271dd742fa92d09f4e22874616facffab5705d535652`;
its decision is `PASS_MECHANICAL_ONLY`. Six training/evaluation leaves
`1203714`--`1203719` were authorized in parallel, with afterany non-promoting
closeout `1203720`.

Job `1203715` (`residual_pl_rep_off`) hard-failed on the first real batch.
AMP skipped the same batch eight times while reducing the scale from `32768`
to `256`, then raised
`FloatingPointError: S1 AMP could not produce a successful optimizer update
after 8 retries`. It produced no checkpoint or metric. The other five leaves
completed only for terminal provenance, and finalizer `1203720` sealed
`PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE` with an empty contrast set.

The root cause is a P0/KAT coverage gap with a concrete numerical mechanism.
The production PL likelihood entered the temporal reduction in FP16. A finite
per-tubelet log-probability summed over 384 tubelets can exceed FP16's `65504`
range before GradScaler can act. The historical estimator KAT used float64 and
`T=1`; the synthetic one-step P0 did not assert the production-horizon
reduction. The next source revision promotes half/bfloat PL likelihood
calculation and the unchanged sum-then-batch-mean policy reduction to FP32. It
also adds an AMP-shaped `T=384`, native-capacity `N=220`, `K=64` backward KAT
whose objective magnitude deliberately exceeds FP16 range, and makes that KAT
mandatory in P0 schema v4 for every score-function arm. The gate binds those
values to the actual decoded `180x320`, floor-native `11x20` source grid rather
than allowing an unrelated `160x160` synthetic grid to stand in for production
capacity. This is a numerical correctness repair, not temporal normalization
or a changed estimator weight.
The failed namespace is never resumed. The numerical repair is exact commit
`30f9ca6fff1572e2eabc6c1b6636c4cc23595a62`, proxy-synced to clean snapshot
`/data/run01/sczc063/yuzibo/projects/opentad_georoute_30f9ca6f_20260729_plamp`.
The complete remote Linux suite passed `120/120`. Standalone CUDA KAT Job
`1203873` completed `0:0` on `g0051`; its production-horizon objective was
`128637.0234375`, all scaled gradients were finite, and its internal/file
SHA-256 values are
`7d0ccc346b95180d02a5ddcf4253ac0278e83f39a6f7e434357c86067e3c8e84`
and
`75ef280473f5032fd734fb86f1f58207702c1999d34c5c7132d40ff5017ae4a4`.
This passes numerical correctness only. A new full six-arm namespace may start
only after the old run seals INCOMPLETE and the new source passes its expanded
per-arm P0.

That replacement was deployed as Jobs `1204015`--`1204028` after the old
closeout and all-at-once capacity gates passed. All six schema-v4 P0 leaves and
P0 finalizer `1204021` completed, but residual-PL stage `1204023` again failed
on real batch 0 after the scale-256 attempt. The v4 P0 had a second coverage
gap: its model forward/backward ran without autocast or GradScaler, while its
AMP horizon KAT differentiated only an isolated logits tensor. It did not test
the scaled gradient through actual scout and model parameters.

The next numerical revision therefore keeps the same PL likelihood,
sum-then-batch-mean objective, estimator weight, six arms, seed, K and epochs,
but executes the complete low-cost scout/route graph in FP32 outside autocast.
P0 schema v5 replaces the disconnected assurance with a full-model AMP
forward, scaled backward, unscale, required-parameter finiteness check and
zero-learning-rate optimizer step at the registered floor scale `256`. The
isolated `T384/N220/K64` KAT remains a subordinate arithmetic check. This is
precision hardening, not loss normalization, clipping, or a scientific
intervention.

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
