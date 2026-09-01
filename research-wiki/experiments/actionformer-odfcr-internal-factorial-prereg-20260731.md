---
type: experiment_preregistration
node_id: design:actionformer-odfcr-internal-factorial-20260731
title: "ActionFormer ODF-CR official-dense-floor internal factorial"
status: tested
updated: 2026-08-01
paper_main_table_eligible: false
official_test_authorized: false
---

# ActionFormer ODF-CR internal factorial preregistration

## Decision context

The terminated DCSR G1 lost `-7.556202 pp` Avg-mAP against its same-run dense
control. Frozen no-training replay attributed `-7.418076 pp` to the one-layer
scaffold/decomposition, only `-1.239537 pp` to K384 residual support, and
`+1.101411 pp` value to all-query residual execution over scaffold-only. This
motivates a separately named successor, ODF-CR: preserve an official-quality
dense proposal floor and test whether an independently gated residual has value.
It does not revive DCSR G1 or hard query deletion.

## Frozen internal 2x2

The training matrix is scaffold depth `1/3` by residual `off/all_valid`:

- `d1_off`: one-layer dense diagnostic floor, no residual;
- `d1_all`: the same one-layer floor plus an independent all-query residual;
- `d3_off`: exact official three-layer ActionFormer dense head, no residual;
- `d3_all`: the exact official floor plus an independent all-query residual.

The new isolated mode is `official_dense_floor_factorial`. Existing
`official_identity` and `cheap_dense_scaffold` semantics and checkpoint loading
must remain unchanged. Residual final projections start at exact zero; training
retains official full-grid targets, masks, normalizer, decoder and NMS. K384 is
not a fifth trained arm; it is a frozen-checkpoint counterfactual replay.

## Data, seeds and comparability

Holdout-v2 is built only from the 200 official THUMOS `validation` videos. The
builder must read and bind the previous 160/40 manifest, select its new
40-video holdout exclusively from the old train-160, require new/old holdouts
to be disjoint, and use the other 160 videos for training. Both sides contain
all 20 classes; test records are rejected. Frozen training seeds are
`2026073101/2026073102/2026073103`. They are paired training replicates on one
fixed holdout, not independent validation splits.

These are internal architecture-selection results. Absolute values must not be
compared with historical `63.xx`, released `66.833392`, or official S0
`66.583013`; no paper main-table or official-test claim is authorized.

## Gates

Before training, a real-CUDA G0 must prove zero-tolerance equality between
official dense and `d3_off` for state keys/tensors, native points/masks,
pre-decode logits/offsets and final labels/scores/timestamps. It also proves
zero residual output at initialization and paired floor initialization for the
depth-one arms.

Residual utility continues only when the paired `d3_all-d3_off` contrast has:

- mean Avg-mAP delta `>= +0.25 pp`;
- positive Avg delta in at least two of three seeds;
- mean mAP@0.6 and @0.7 deltas both `>= 0.00 pp`;
- complete identity/evaluator/checkpoint/receipt integrity.

Each seed is subtracted before computing the arithmetic mean and sample
standard deviation; equality at a threshold passes. If utility passes, frozen
`d3_all` checkpoints replay `stratified_uniform` K384 with hash seed
`2026073100`, exactly `min(384, valid_query_count)` residual queries per video
and receipt-bound allocation IDs/hashes. The support gate requires Avg penalty
`>= -0.50 pp` and @0.6/@0.7 penalties each `>= -1.00 pp`.

## Interpretation

- `d1_off << d3_off`: the shallow proposal floor explains the prior collapse;
- `d3_all ≈ d3_off`: residual compute lacks useful incremental value and stops;
- residual utility pass plus K384 failure: support density is the bottleneck;
- both gates pass: write a separate official/cost preregistration before any
  official test evaluation or paper claim.

The design was frozen at candidate commit `77244d5` on
`codex/actionformer-densefloor-factorial-20260731`.

## 2026-07-31 implementation and formal execution

The isolated implementation is frozen and pushed at exact commit/tree
`01cdb78d2b7668098b6b13a1e49433d48fbc1a8d` /
`e70d2956a197b1204e721239178e76152efe282b`. It includes the four configs,
holdout-v2 builder/consumers, exact G0, independent evaluator, per-seed
finalizer, paired G2 aggregate, conditional K384/G3 tools, Slurm launchers and
focused tests. Linux preflight passed `71` tests.

The immutable holdout-v2 SHA-256 is
`b8cac555f3d31e02468dbca3b3b0ada2d30b05bf046c10eb16304abb92499d1a`:
160 training and 40 decision videos from official validation, with the new
holdout selected only from the previous train-160 and disjoint from the
previous holdout-40. No test record or test prediction is used.

Formal array `1209259_[0-2]` runs the three frozen seeds. All three real-CUDA
G0 receipts pass every exact check. Their SHA-256 values are:

- seed `2026073101`: `212835e56ddbd7538ec14173f52e6fa1323adef82b11ff446429b3b314cbbfc7`;
- seed `2026073102`: `7b6aa7a00583d59000c9eb3e4cdaac1145b9cd92643e66bc72c8a3fa85978aa6`;
- seed `2026073103`: `25ac7539f8b9e86bf66b5eace8b0905f5420a763364afb0857f2ba152b5a68f8`.

Each proves official `d3_off` state/points/masks/pre-decode/final-output
identity, identical `d3_all` floor, exact-zero initial residual projections and
paired depth-one scaffold initialization on `cuda:0`/FP32. All three tasks have
entered `d1_off` training. The unique aggregate successor is Job `1209267`
with dependency `afterok:1209259`.

Three engineering events are preserved and are not model results:

- `remote_profile_nonzero_under_errexit_v1`: deployment v1 stopped before
  runtime/run-root/Slurm creation;
- `yaml_1_1_off_coercion_residual_support_v1`: deployment v2 produced Linux
  `65 passed, 5 failed` because unquoted YAML `off` parsed as Boolean; zero
  Slurm jobs;
- `aggregate_submit_gpu_count_missing_v1`: v3 main array was already submitted
  when the cluster rejected the first G2 submission for lacking explicit GPU
  count. Recovery submitted only G2 with `--gpus=1`; the main array was not
  duplicated.

Deployment receipt SHA-256 is
`ee5ae82d7f7bfbf6aa3e67615136e99c207bb643d0b23908ed4e5b596ea5ac5d`.
At formal launch, status was `experiment_running`: no arm metric, G2/G3 result,
paper row, official-test result or efficiency claim existed yet.

## 2026-08-01 terminal matrix and G2 verdict

Formal array tasks `1209259/1209260/1209261` completed `0:0` in
`00:35:25/00:35:15/00:32:17`. Dependent G2 Job `1209267` completed `0:0` in
four seconds. The runtime remains clean at the frozen branch/commit/tree. The
three matrix completion receipts and their SHA-256 values are:

- seed `2026073101`: `c4c7bcf3c63314e3c436bc1d9bb903b05350f7dfafb1e1c0e0cb8aabfc0f409f`;
- seed `2026073102`: `397ce608487a27afd3e9a94c773717a720bedef07e80b94a93016e53af393549`;
- seed `2026073103`: `46bca957890f44e2cd03d7b132c5087bf58713b210cbca38d4c686352e0ac698`.

The validated G2 aggregate SHA-256 is
`9172eddcbf5f9a4943b303e20b57f4492f0a44b18c39f892d5829b1f0a79ddec`.
Mean internal holdout metrics are:

| arm | Avg | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 |
|---|---:|---:|---:|---:|---:|---:|
| `d1_off` | 61.2262 | 80.5426 | 75.2085 | 67.1938 | 51.6636 | 31.5228 |
| `d1_all` | 65.0952 | 82.5106 | 78.6108 | 67.3667 | 56.3965 | 40.5913 |
| `d3_off` | 68.7863 | 84.7522 | 79.2429 | 70.7168 | 63.0897 | 46.1298 |
| `d3_all` | 68.6056 | 85.5988 | 80.0094 | 70.3512 | 60.3429 | 46.7259 |

All values are percentages on the fixed internal holdout. The paired
`d3_all-d3_off` Avg deltas are `-0.6970/+1.5070/-1.3520 pp`, with mean
`-0.1806 pp`, sample SD `1.4978 pp`, and only one positive seed. Mean threshold
deltas are `+0.8466/+0.7666/-0.3656/-2.7468/+0.5960 pp`. Therefore the frozen
G2 checks fail at mean Avg, positive-seed count and @0.6; only @0.7 passes.
`residual_utility_gate_pass=false` is a legal model negative. No K384 replay or
G3 job/receipt exists, exactly as required by the protocol.

The other factorial contrasts are diagnostic rather than continuation gates:

- `d1_all-d1_off`: `+3.8689 pp` Avg, per seed
  `+1.0743/+1.2892/+9.2433 pp`;
- `d3_off-d1_off`: `+7.5600 pp` Avg, including `+11.4261/+14.6071 pp` at
  @0.6/@0.7;
- depth-by-residual interaction: `-4.0496 pp` Avg.

The fixed-holdout, three-training-seed uncertainty is wide. For the G2 Avg
contrast, the descriptive paired-seed t interval is approximately
`[-3.90,+3.54] pp`; it is not a population generalization interval and does not
establish equivalence, non-inferiority or statistical significance.

## Read-only terminal attribution

The terminal analysis read all twelve attested `eval_results.pkl` files
(`8,000` post-NMS detections per arm/seed), the frozen 40-video holdout and its
`454` validation GT instances. It performed no training, checkpoint mutation,
test evaluation or model selection.

### Dense floor depth

The depth-three residual-off floor improves mAP increasingly at strict IoU:
`+4.21/+4.03/+3.52/+11.43/+14.61 pp` over `d1_off`. Class-aware recall@200
changes by `+0.44/+0.59/+0.88/+5.87/+13.00 pp`; mean best same-label tIoU
increases `+0.0486`, while normalized start/end errors fall
`0.0366/0.0305`. At @0.7, recall improves by `+13.27 pp` for `<2s`,
`+14.04 pp` for `2--4s`, `+12.40 pp` for `4--8s` and `+7.64 pp` for
`8--16s`. The `16--32s` and `>=32s` bins show larger point estimates but contain
only five and three GT instances. This is strong internal evidence that the
one-layer floor caused much of the earlier high-IoU collapse; it is not an
official benchmark or an independent-holdout claim.

### Residual on the depth-three floor

`d3_all` slightly improves class-aware recall@200 and best-match boundary
statistics, including only `+0.07 pp` recall at @0.7 and `+0.0023` mean best
tIoU, yet loses `2.7468 pp` mAP at @0.6 and `0.1806 pp` Avg. The mismatch points
away from gross proposal absence and toward ranking/class composition or
overfitting/interference. It is highly heterogeneous:

- largest mean class AP losses are SoccerPenalty `-10.55 pp`, CricketShot
  `-5.79 pp`, CleanAndJerk `-5.56 pp` and GolfSwing `-4.20 pp`;
- largest gains are ThrowDiscus `+8.95 pp`, TennisSwing `+5.57 pp`, Shotput
  `+2.95 pp` and LongJump `+2.89 pp`;
- @0.7 duration-recall deltas are `-2.72/-1.17/+4.39/+2.08/-26.67/-11.11 pp`
  for `<2/2--4/4--8/8--16/16--32/>=32s`, with the last two bins too small for a
  stable conclusion;
- per-video class-aware recall@200/@0.7 has 11 positive, 14 negative and 15
  unchanged videos; its video bootstrap interval for the mean is
  `[-2.51,+4.75] pp`.

Score mean rises only `0.00136` and q99 falls `0.00644`. Descriptive TP rates
in score bins and retained same-label overlap do not show a single monotonic
failure: post-NMS overlap fractions change only `+0.12/+0.42 pp` at overlap
0.5/0.7. These are not probability-calibration estimates, and pre-NMS
suppressed proposals were not saved, so calibration or NMS causality remains
unidentified.

TensorBoard scalars show the same learning-rate schedule for every arm.
`d3_all` has lower final-loss average over the last 10% of logged steps than
`d3_off` (`0.2397` versus `0.3338`), with both classification and regression
loss lower, despite slightly worse holdout Avg. Thus persistent nonfinite
training, a dead branch or simple failure to reduce the objective is not a
sufficient explanation. No residual amplitude, gate, activation, gradient norm
or gradient-cosine telemetry exists, so gradient conflict itself is not proven.

### Competing explanations and falsifiers

1. **Capacity saturation / residual overfit or redundant perturbation.** The
   negative interaction, lower `d3_all` training loss without validation gain,
   and strong class/seed heterogeneity support this account. Slight recall and
   boundary improvements plus the positive seed-02 result are counterevidence.
   It predicts that a frozen-floor residual-only probe remains near zero and
   that benefit reappears only when floor capacity is intentionally reduced.
2. **Score/ranking or class-specific post-processing interference.** The @0.6
   AP loss despite non-worse recall/best-match boundaries supports ranking
   interference. Stable retained-overlap rates and non-worse high-score TP
   rates argue against NMS as the sole cause. A no-training, preregistered replay
   that saves pre-NMS logits/proposals and sweeps only a frozen calibration
   transform would falsify or support this account.
3. **Seed/holdout composition.** Wide seed uncertainty and opposite class/video
   directions support sample sensitivity; the fairly stable negative @0.6
   delta (cross-seed SD `0.6008 pp`) is counterevidence to pure random noise. A
   fresh disjoint validation holdout with new paired seeds is the minimal
   generalization test.
4. **Gradient conflict or residual-scale miscalibration.** The architecture
   permits it, but no gradient/gate telemetry exists and the lower training
   loss cuts against gross optimization failure. A separately preregistered
   short frozen-scaffold or low-residual-LR diagnostic with gradient cosine and
   residual-norm ledgers is needed; the completed protocol does not authorize a
   rescue run.

## Route and claim decision

Status moves from `experiment_running` to `tested`; the exact failure of the
predeclared residual-utility claim is `empirically_supported`. The unmodified
`all_valid` residual on the official depth-three floor is rejected for this
route, and K384/G3 remain forbidden. The depth-three official dense floor is
retained as the architecture prerequisite for any future route. The result
does not test sparse conditional execution and therefore does not reject
conditional sparse routing in general.

The strongest allowed wording is: on this fixed internal validation holdout,
three paired training seeds show that the depth-three official dense floor
substantially outperforms the one-layer diagnostic floor, while adding the
all-valid residual to the depth-three floor fails the preregistered G2 utility
gate. No paper main-table, official test, efficiency, statistical-significance,
equivalence or universal sparse-routing claim is authorized.

## Operational closure

After the terminal artifacts, attribution and claim trace were validated, the
heartbeat monitor `sparsehead-official-matched-monitor` was retired. The app
delete RPC repeatedly timed out while invoked from its own heartbeat, so its
configuration was recoverably moved out of the active automation directory.
This operational fallback did not inspect, submit or cancel any Slurm job.
