# ActionFormer Official Dense Floor + Conditional Residual Factorial Design

Date: 2026-07-31

Branch: `codex/actionformer-densefloor-factorial-20260731`

Status: `designed`

Paper status: `paper_ready=false`

## 1. Decision

The terminated DCSR G1 route will not be tuned or resumed. This specification
defines a new validation-only route named Official Dense Floor + Conditional
Residual (ODF-CR).

The decisive experiment is a four-arm training factorial:

| arm | dense floor | residual during training/inference |
|---|---|---|
| `d1_off` | one-layer diagnostic scaffold | disabled |
| `d1_all` | one-layer diagnostic scaffold | all valid native queries |
| `d3_off` | exact official three-layer ActionFormer head | disabled |
| `d3_all` | exact official three-layer ActionFormer head | all valid native queries |

The two `*_all` terminal checkpoints additionally receive no-training
counterfactual replay with uniform K384 residual execution. K384 replay is not
a trained arm and cannot be reported as one.

## 2. Why this differs from terminated DCSR G1

G1 combined a one-layer scaffold, a three-layer residual head and K384 residual
execution in one trained model. Its scaffold-only counterfactual was already
`-7.4181 pp` below dense, while all-query residual recovered only `+1.1014 pp`
and K384 support added a `-1.2395 pp` penalty. Scaffold capacity, residual
utility and support sparsity were therefore not independently controlled.

ODF-CR makes the official head the proposal floor when depth is three and
separates the residual into its own module. It asks, in order:

1. Does `d3_off` reduce exactly to official dense ActionFormer?
2. How much is lost by using a one-layer floor (`d1_off - d3_off`)?
3. Does a residual add useful information at each floor depth?
4. Holding a trained checkpoint fixed, what is the pure K384 execution-support
   penalty?

## 3. Model contract

### 3.1 New isolated mode

Add a new mode, `official_dense_floor_factorial`. Do not change the semantics
or checkpoint loading of `official_identity` or `cheap_dense_scaffold`.
The current code does not implement this mode; no existing mode may be
relabeled or reinterpreted to stand in for it.

The new mode has these fields:

- `scaffold_num_layers`: `1` or the official `head_num_layers` (`3`);
- `residual_enabled`: Boolean;
- `residual_execution_support`: `all_valid` for trained residual arms and
  `off` when disabled;
- `residual_num_layers`: exactly the official `head_num_layers`;
- `residual_scale`: `1.0`;
- `training_loss_support`: fixed
  `official_all_valid_fpn_queries`.

Unknown values fail closed.

### 3.2 Official-depth floor

When `scaffold_num_layers == head_num_layers`, the scaffold is the existing
official `cls_head` and `reg_head`; no duplicate scaffold is instantiated.

For `d3_off`:

- no residual module is instantiated;
- state keys, initialization order, points, masks, pre-decode outputs and final
  predictions must be exact to the official dense model;
- any mismatch is an engineering failure and blocks training.

For `d3_all`, the same official heads remain the dense floor and a separately
named residual branch is instantiated after the floor. The floor initialization
must remain exact to `d3_off`.

### 3.3 One-layer diagnostic floor

When `scaffold_num_layers == 1`, instantiate the same one-layer scaffold
contract used by G1. Residual-off and residual-on arms must initialize the
scaffold identically for a shared seed; residual construction occurs after the
scaffold.

The existing official heads may remain registered but are not executed as the
floor in depth-one arms. Receipts must report executed modules separately from
registered parameters.

### 3.4 Independent residual branch

The new residual modules are:

- `odfcr_residual_cls_head`;
- `odfcr_residual_reg_head`.

Their final projections are zero-initialized so the initial model output equals
the dense floor. Hidden and final residual parameters must be included in the
optimizer when enabled.

All-query support uses the complete valid FPN mask directly; it must not emulate
all-query execution by guessing a large budget. K384 is introduced only by the
frozen counterfactual replay tool after training.

### 3.5 Loss and decoder

- Original full FPN masks, targets, label assignment and positive normalizer
  remain unchanged in every arm.
- The residual branch changes predictions, not supervision support.
- The official decoder, pre-NMS threshold, seven-argument Soft-NMS and maximum
  output count remain unchanged.
- No test GT, teacher signal, cached test prediction or evaluation-derived
  selector may enter the forward path.

## 4. Data and seed contract

Repeated use of the previous 40-video holdout would make the new decision
adaptive to already observed validation results. Freeze a new holdout-v2 before
any ODF-CR metric:

- source population: the same 200 official THUMOS `validation` videos;
- holdout-v2: 40 videos selected only from the previous 160-video development
  training set;
- holdout-v2 is disjoint from the previous G1 40-video holdout;
- training-v2: the remaining 160 videos, which may include the old holdout;
- all 20 classes must be present on both sides;
- manifest records exact IDs, counts, source annotation SHA-256 and the previous
  manifest SHA-256;
- test records are rejected.

The holdout-v2 builder must require the previous manifest as an input and emit
schema `actionformer_odfcr_internal_holdout_v2`. It must verify that the
previous manifest contains exactly 160 training and 40 holdout validation
videos, form its candidate pool exclusively from those 160 old training IDs,
select exactly 40 class-covered videos, and define training-v2 as the other 160
official-validation videos. The evaluator and launcher must validate these
set-membership semantics, including disjointness from the old holdout, rather
than accepting only a matching manifest hash.

New development seeds:

- `2026073101`;
- `2026073102`;
- `2026073103`.

They are disjoint from every currently registered G1 or prior official seed. A
future official preregistration must reject all known development seeds. Every
arm for one seed shares data order, optimizer, schedule, environment and
evaluator. These three seeds are paired training replicates on one fixed
holdout, not independent validation splits and not a population-level
generalization claim.

## 5. Exact controls

All four arms share:

- official THUMOS I3D features and annotation source;
- FPN projection, point generator, backbone-free feature input and decoder;
- 5 warmup + 30 optimizer epochs;
- batch size, optimizer, learning rate, weight decay and gradient clipping;
- terminal epoch-35 EMA checkpoint;
- official holdout evaluator and tIoU `0.3:0.7`;
- pre-NMS/NMS settings;
- random seed and deterministic flags;
- full-grid loss support and normalizer.

Only scaffold depth and residual presence change during training. K384 changes
only residual execution masks during frozen replay.

## 6. Gates

### G0: exact implementation identity

Before any factorial training:

1. official dense and `d3_off` state keys and tensors are exact;
2. native points and masks are exact;
3. pre-decode class logits and offsets are exact;
4. final labels, scores and physical timestamps are exact;
5. `d3_all` floor tensors at initialization are exact to `d3_off`;
6. residual outputs are exact zero at initialization;
7. `d1_off` and `d1_all` scaffold initial tensors are exact for a shared seed.

The CUDA equivalence probe runs both models in `model.eval()` under
`torch.no_grad()` with the same ordered input tensors, masks, state and
deterministic flags. Equality is bitwise with zero tolerance. Its receipt binds
device, dtype, source identity and input-tensor hashes.

Any failure is engineering-only and blocks the run.

### G1: scaffold depth effect

Primary contrast:

`d1_off - d3_off`.

This is diagnostic and has no continuation threshold. It quantifies how much
of terminated G1 is explained by the shallow floor.

### G2: residual utility

Contrasts:

- `d1_all - d1_off`;
- `d3_all - d3_off`;
- depth × residual interaction.

Conditional residual computation continues only if `d3_all - d3_off` satisfies
all of:

- mean Avg-mAP delta `>= +0.25 pp`;
- at least two of three seed-level Avg deltas are positive;
- mean mAP@0.6 and mAP@0.7 deltas each `>= 0.00 pp`;
- no identity, non-finite, checkpoint, evaluator or receipt failure.

Every reported delta is in percentage points. Compute each seed's paired
subtraction first, then report the arithmetic mean and sample standard
deviation across the three seeds; equality at a threshold passes. A paired
bootstrap interval may be reported descriptively but is not a gate.

Otherwise the residual is not valuable enough to justify conditional compute.

### G3: frozen K384 support replay

For `d3_all`, compare frozen all-query versus K384 residual execution on the
same raw inputs and checkpoint.

The replay selector is frozen as `stratified_uniform`, budget `384`, selector
hash seed `2026073100`, with exactly `min(384, valid_query_count)` residual
queries per video. Allocation IDs/hashes are receipt-bound, and deterministic
tie-breaking must be identical across arms and seeds.

Continuation requires:

- mean Avg-mAP penalty `>= -0.50 pp`;
- mean mAP@0.6 and mAP@0.7 penalties each `>= -1.00 pp`;
- same dense-floor tensors and immutable checkpoint;
- no invalid/filter/rounding or output-schema mismatch.

If residual utility passes but K384 fails, the result supports a support-density
bottleneck and stops fixed K384. It does not authorize threshold tuning.

## 7. Metrics and diagnostics

Primary:

- Avg-mAP;
- mAP@0.3/0.4/0.5/0.6/0.7;
- paired per-seed deltas, mean and sample standard deviation.

Secondary:

- pre-NMS proposal recall at fixed top-K;
- post-NMS class-aware and class-agnostic recall;
- per-class AP at 0.6/0.7;
- duration-stratified recall;
- start/end boundary error normalized by GT duration;
- score distribution and descriptive score-conditioned TP rate;
- residual-to-floor logit and offset norms;
- first-step and per-epoch parameter-group gradient norms;
- residual selected-query coverage by FPN level.

Cost diagnostics:

- floor, residual, selector/replay-mask and decoder MAC estimates;
- synchronized floor/residual/head latency;
- complete feature-to-final-detection holdout latency and peak memory.

Internal cost is diagnostic only. No efficiency claim is allowed before a
separate official protocol.

## 8. Execution and evidence

Use a Slurm array with one task per seed. Within each allocation, execute the
four arms serially in a fixed, receipt-recorded order. Each arm receives a
fresh output directory and cannot resume or overwrite another arm.

Per-seed completion requires exactly four arm attestations, exact config hashes,
manifest identity, seed, source commit/tree, environment and terminal EMA
checkpoint. Cross-seed aggregation requires exactly the three frozen seeds and
four frozen arm IDs.

Negative model gates write a valid completion with `gate_pass=false` and exit
zero. Traceback, OOM, non-finite loss, identity mismatch or missing artifacts
are engineering failures and exit nonzero.

## 9. Claim boundary

Allowed:

- internal evidence about scaffold depth, residual utility and frozen residual
  support on holdout-v2;
- termination of this new route if its frozen gates fail.

Forbidden:

- official THUMOS test performance;
- comparison of absolute internal values with historical `63.xx` or official
  `66.xx`;
- paper main-table rows;
- end-to-end video efficiency or state-of-the-art claims;
- treating K384 replay as a separately trained model;
- universal claims about sparse heads.

If all gates pass, the next step is a new official preregistration. Passing does
not itself authorize official test evaluation.

## 10. Failure interpretation

- `d3_off` identity failure: implementation defect; fix before training.
- `d1_off << d3_off`: shallow floor capacity/parameterization explains the old
  collapse.
- `d3_all ≈ d3_off`: residual has no useful value; terminate conditional
  residual compute.
- `d3_all > d3_off`, K384 replay fails: residual is useful, but fixed K384
  support is structurally insufficient.
- both utility and K384 gates pass: freeze the architecture and write a separate
  official/cost preregistration.
