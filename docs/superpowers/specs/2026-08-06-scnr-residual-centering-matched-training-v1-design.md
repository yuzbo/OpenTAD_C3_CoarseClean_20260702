# SCNR residual-centering matched training v1 design

## Decision

Run one fresh, development-only, seed-3407 matched training pair on the approved
one-native-cell G1 anchor:

| Variant | `georoute_branch_calibration_mode` | Meaning |
|---|---|---|
| `none_control` | `none` | Unchanged dynamic SCNR control |
| `residual_window_center` | `residual_window_center` | Subtract the valid full-window residual-modifier mean |

The two cells are not the old M2 G1/G2 floor arms. Both use G1 geometry and are
trained from the same pretrained initialization. The sole intended variable is
the residual-window centering transform already admitted by the frozen-checkpoint
mechanism probe.

No additional Pro-model discussion is required before this single-variable
development experiment. The mechanism, matchedness boundary, failure semantics,
and next decision are fully specified here.

## Scientific question

The frozen-checkpoint probe established that an additive residual-branch offset
causes categorical role collapse and that mean centering restores context/ROI
reachability. It did not establish whether a freshly optimized centered model is
more useful to the detector. This experiment asks only:

> Under the same G1 dynamic ROI+TokenSelect Hybrid recipe and exact compute
> budget, does residual-window centering improve development high-IoU detection
> after matched fresh training?

The experiment does not compare ROI floors, select G1 over G2, prove role
complementarity, or establish an efficiency benefit.

## Immutable shared recipe

Both cells inherit the validated M2 G1 `native_1cell_main` recipe:

- THUMOS14 Fit training and Gate development populations, with their existing
  manifest, annotation, class-map, video-root, and no-leak bindings;
- VideoMAE two-frame tubelets, `T=384`, native source grid `11x20`;
- Scheme A,
  `u=q_base+max(0,delta_roi,delta_residual)`, with no independent `q_ctx`;
- continuous G1 ROI floor `(1/20,1/11)` and otherwise identical geometry;
- exact full-window `B=24576`, fully induced dynamic `K_t` including `K_t=0`,
  true ragged execution, no padding, and masked-zero carrier;
- support-only representation, unchanged detector/head/evaluator/NMS, no GT,
  teacher, oracle, prediction cache, role quota, target fraction, or post-hoc
  reassignment in routing;
- seed `3407`, world size `1`, local/global batch `1`, 60 epochs, 160 successful
  updates per epoch, and exactly 9,600 successful updates;
- identical data order, route-private successful-update-keyed randomness,
  default deterministic warn-only training, AMP and EMA, final-EMA-only
  checkpoint retention, `solver.fp16_compress=false`, and at most eight
  same-batch AMP retries;
- fresh per-cell P0 and fresh training. No old M2 checkpoint is used as a
  control, initialization, or resumed state.

Every shared input and protocol field is included in a common matched-protocol
hash. Each cell additionally binds its variant name and calibration mode.

## Execution and evaluation

Each cell performs, in order:

1. source/config/data/storage/Slurm identity validation;
2. a fresh no-performance dynamic-SCNR P0;
3. fresh 60-epoch training and publication of exactly one epoch-59 EMA
   checkpoint plus atomic sidecar;
4. `accuracy_a` followed by `accuracy_b` on the complete Gate population,
   serially on the same Slurm-visible GPU;
5. exact duplicate validation before any cross-cell contrast.

Training retains the production deterministic warn-only backend. Both accuracy
replays force strict math SDPA and disable TF32 to eliminate the previously
localized memory-efficient-SDPA replay drift. Metric evaluation and out-of-band
route telemetry are enabled; performance profiling is disabled. The two replays
must have byte-identical raw predictions, identical semantic prediction payloads,
identical route-payload hashes, identical population hashes, and identical
metrics. This strict evaluation backend is common to both freshly trained cells;
absolute values are not compared with historical M2 predictions.

## Fail-closed integrity gate

No performance contrast is emitted unless both cells satisfy all of the
following:

1. exact clean runtime, immutable input, config, P0, checkpoint, sidecar, Slurm,
   population, and artifact self-hash validation;
2. exactly 60 epochs and 9,600 successful updates, with one final checkpoint and
   no resume or temporary checkpoint;
3. exact `B=24576` per window, valid-only unique selection, true ragged/no-padding
   execution, masked-zero semantics, and matching no-leak receipts;
4. exact `accuracy_a == accuracy_b` raw prediction and route-payload replay;
5. matched cell signatures and common-protocol hashes, with calibration mode as
   the only experimental difference;
6. in `residual_window_center`, aggregate selected context and ROI counts are
   each nonzero and the selected residual fraction is strictly below one;
7. in `none_control`, calibration is a numerical identity; in the centered cell,
   the maximum absolute valid post-centering residual mean is at most `1e-4`.

Any missing, failed, nonterminal, mismatched, or tampered cell produces an
`INCOMPLETE_NO_PERFORMANCE_INFERENCE` receipt with empty contrasts. There is no
partial-cell salvage.

## Preregistered development screen

After the complete integrity gate passes, define centered-minus-control deltas
from the duplicate-validated Gate metrics. The development screen passes only if
all three signs hold:

1. `delta(mAP@0.6) > 0`;
2. `delta(mAP@0.7) > 0`;
3. `delta(Avg-mAP) >= 0`.

Strict high-IoU signs avoid inventing an unpowered post-hoc numerical margin.
A tie at either high-IoU threshold, a crossing, or lower Avg-mAP is
`HOLD_CENTERING_NOT_AUTHORIZED_FOR_COST`. Passing is only
`PASS_SINGLE_SEED_CENTERING_ACCURACY_SCREEN`.

## Claim and promotion boundary

A pass authorizes only the design and execution of a new same-GPU full-stack
paired-cost protocol using both counterbalanced orders `ABBA` and `BAAB`. Exact B
does not imply equal attention-pair or system cost because centering can change
the distribution of `K_t` across native clips.

Only a later accuracy-plus-cost Pareto pass may authorize fresh disjoint-seed
training for seeds `3408/3409`. This seed-3407 screen alone never authorizes:

- a multi-seed or official-test run;
- M3 or ROI-floor selection;
- an efficiency, complementarity, Hybrid-efficacy, or paper claim;
- learned-null, scout-projection, `K_t>=1`, variable window-level B, MoD, or any
  second model intervention.

## Expected resources

Historical M2 cells required roughly 4 h 52 min to 5 h 15 min each. The two cells
may train independently in parallel and consume approximately 10--12 GPU-hours
including strict duplicate accuracy replay. The finalizer uses one site-required
GPU only as disclosed scheduling overhead and performs no model inference.
