# SCNR residual-centering paired full-stack cost v1 design

## Decision

Run the already authorized cost study directly. No further Pro-model discussion
is required because the scientific variable, source checkpoints, measurement
scope, order control, statistical unit, decision margin, and promotion boundary
are all frozen before any cost result is observed.

This study consumes the completed seed-3407 `none_control` and
`residual_window_center` epoch-59 EMA checkpoints. It performs no training,
resume, tuning, model change, or new accuracy evaluation.

## Source evidence and question

The matched-training finalizer at model runtime
`16137484c5ccad422e017e67a81c1a07d1ed2fbb` passed its preregistered accuracy
screen. Centering minus control is `+2.05 pp` Avg-mAP, `+2.14 pp` mAP@0.6, and
`+1.16 pp` mAP@0.7 on the duplicate-validated Gate development population.
The finalizer explicitly authorizes a same-GPU `ABBA+BAAB` full-stack cost
study and keeps seeds 3408/3409, official test, and paper claims closed.

The remaining question is deliberately narrow:

> Does residual-window centering preserve full-stack latency and energy within
> a predeclared 5% non-inferiority margin when both trained checkpoints are
> replayed on the same physical GPU under order-balanced measurement?

The centering intervention is an identifiability/accuracy repair, not a cost
reduction mechanism. Therefore matched-cost non-inferiority, rather than a
mandatory 5% speedup, is the promotion criterion. A stricter Pareto result is
recorded separately if observed.

## Immutable source binding

The cost run must recursively validate and bind:

- matched-training root
  `/data/run01/sczc063/yuzibo/scnr_residual_centering_matched_training_16137484_s3407_20260806_061352`;
- stage Jobs `1223819` and `1223820`, after-any finalizer `1223821`, and
  finalization SHA-256
  `2a9351a3c21c850f28aab4bd162f7b69f3ca40921a97304431a8a760d6ebbe8a`;
- common matched-protocol SHA-256
  `34defbdbc30e7fff10bbb05d7e6665dd29b8128f8f03cd389250bca9e3e7493c`;
- control checkpoint SHA-256
  `5350a03b1584ab8e0023b6c212fc2a3b8526c45de169b5660d827762a9dd6ff4`;
- centered checkpoint SHA-256
  `e45a37708c68e1ea02bec02bc14dfbe199ea8303f562088469d98a8fe45c7028`;
- the exact 40-video Gate population and its telemetry population SHA-256
  `35aaa9192b4dfd4bd03599450fbbceed6c7d60e98d8df7f582bd39df26f40aa8`.

The later cost execution commit is allowed to add cost-only contracts,
profilers, launchers, tests, and documentation. A Git-diff gate must prove that
`opentad/`, `configs/`, train/test entrypoints, the trained residual-centering
contract/runner, and inherited measurement primitives are byte-unchanged from
the model runtime commit. Runtime and cost-execution commits remain distinct in
every receipt.

## One-job execution protocol

Exactly one held Slurm Job is submitted, receipt-bound, and then released. It
uses one Slurm-visible GPU as logical `cuda:0`, world size one, batch size one,
five allocated CPUs, four detector/input CPUs, and one isolated NVML sidecar
CPU. `CUDA_VISIBLE_DEVICES` is never assigned or overwritten by project code.

One `NvmlSidecarPowerSampler` starts before the first pass and stops after the
eighth pass. It samples the exact GPU UUID every 20 ms into node-local scratch;
the final trace, attempt report, coverage, cadence, and integrated energy are
hash-validated. Diagnostic route telemetry is disabled inside timed forward.

Each pass performs 50 warmup samples, then the complete Gate population in
serial loader order. No warmup or measured observation may be discarded
post-hoc. The exact eight-pass order is:

| Pass | Arm | Block position |
|---:|---|---|
| 0 | `none_control` | ABBA-A1 |
| 1 | `residual_window_center` | ABBA-B1 |
| 2 | `residual_window_center` | ABBA-B2 |
| 3 | `none_control` | ABBA-A2 |
| 4 | `residual_window_center` | BAAB-B1 |
| 5 | `none_control` | BAAB-A1 |
| 6 | `none_control` | BAAB-A2 |
| 7 | `residual_window_center` | BAAB-B2 |

The four declared control/center pass pairs are `(0,1)`, `(3,2)`, `(5,4)`,
and `(6,7)`. This balances which arm encounters each block's leading and
trailing thermal/cold state while retaining every pass.

## Full-stack measurement scope

Every physical window records:

- serial decode, preprocessing, and collate with `num_workers=0`;
- host-to-device transfer;
- scout, route, patch embedding, native ragged VideoMAE backbone, sparse
  adapter, projection, neck, and detector head CUDA events;
- detector forward, window postprocessing, and complete decode-to-window wall
  time;
- video-level gather/NMS, amortized over the pass population;
- end-to-end serial latency, peak allocated/reserved GPU memory, timed-window
  GPU energy, exact `K_t`, selected role counts, and ragged attention pairs.

The profiler reports raw per-window records, per-pass summaries, pooled arm
p50/p95/mean/min/max, energy per sample, gross timed energy, hardware/software
fingerprints, and all source/config/checkpoint/population hashes. FLOPs are not
a substitute and evaluator metrics are not recomputed during timing.

## Preregistered statistical analysis

Primary cost ratios are centered over control:

1. end-to-end serial p50 latency;
2. mean GPU energy per measured sample.

For each metric, calculate the four paired-pass ratios and use their geometric
mean as the point estimate. Compute a deterministic 10,000-replicate 95%
bootstrap interval by resampling complete video clusters and the four declared
counterbalanced pass pairs. The bootstrap seed is `20260806`. Windows are not
treated as independent videos, and no outlier rejection is permitted.

This interval is conditional on one GPU Job. It captures Gate video and
within-job paired-order variation, not independent-job or seed variance.

## Preregistered decision gate

The accuracy portion is inherited unchanged from the sealed matched-training
finalizer: centered mAP@0.6 and mAP@0.7 must each be strictly higher and
Avg-mAP must be non-lower.

The cost portion passes only if the 95% upper confidence bound of both primary
center/control ratios is at most `1.05`. The 5% margin is the predeclared
smallest material cost disadvantage tolerated for an intervention whose
measured development accuracy improved by more than one percentage point at
both high-IoU thresholds. It is fixed before cost observation and cannot be
changed in response to the result.

- Accuracy pass + both cost upper bounds `<=1.05`:
  `PASS_ACCURACY_AND_PAIRED_COST_NONINFERIOR_SEEDS_AUTHORIZED`.
- A complete profile that violates either bound:
  `HOLD_COMPLETE_PAIRED_COST_TRADEOFF_NO_SEEDS`.
- Any missing/tampered source, pass, timing stage, energy coverage, population,
  GPU identity, or receipt:
  `FAIL_INCOMPLETE_PAIRED_COST_NO_INFERENCE`.

`strict_pareto_observed=true` is descriptive only when both point ratios are
`<=1.0` and at least one 95% upper bound is `<=1.0`. It is not required to
authorize seed confirmation and does not itself permit a paper claim.

## Claim and promotion boundary

A pass authorizes only a separately frozen fresh matched confirmation for seeds
3408 and 3409. It does not itself establish multi-seed generalization, Hybrid
complementarity, superiority to dense/fixed/random/free-token baselines,
paper-grade efficiency, official-test performance, or a final method.

Paper-facing evidence still requires at minimum:

1. three-seed paired accuracy with uncertainty;
2. independent repeated cost Jobs, not only a conditional within-job bootstrap;
3. exact official AdaTAD reproduction and matched native-source dense/fixed/
   random/free-token controls;
4. short-action and boundary analyses;
5. a second detector or dataset;
6. a sealed official-test run only after every development decision is frozen.

If the cost gate fails, centering remains a valid single-seed development
accuracy finding but no additional seed or efficiency claim opens.
