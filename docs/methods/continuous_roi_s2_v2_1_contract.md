# Continuous-RoI S2 Crop-Sufficiency Contract v2.1

## 1. Protocol Status

- Route: offline temporal action detection with spatial crop selection.
- Stage: `S2-P`, pre-policy representation sufficiency.
- Research status: `designed`.
- Implementation status: `in_progress`; authorized by the machine-readable
  protocol and static validator.
- Training status: blocked until focused implementation tests and a formal
  full-model one-step CUDA Gate pass.
- Parent research commit:
  `387a1930ec19f8c3ed716c383c4eeabcbaab50eb`.
- Superseded protocol:
  `continuous_roi_s2_preregistration_v2`.
- Official test: sealed for every S2 outcome.

The canonical protocol SHA-256 is
`ef806b7cd37c704d14a54211b1d4e2f9fb88b75599da918272cc6acad157b3af`.
The static audit passed eight contract families and all 128 outcome-state
assignments; its audit SHA-256 is
`5af59b755dd4528fe3e4fd989bb20da71ee40e43ecb5add34083b8ae96057f9d`.

This contract is a narrow corrigendum to the Pro-authored v2. It retains the
accepted source-coordinate geometry, temporal tube, sampler parity, shared
VideoMAE, AdaTAD-derived detector path, evidence sealing, statistics, and cost
ledger. It replaces the v2 training arms, reference population, privilege
comparisons, cost accounting, outcome state machine, and cluster contract.

## 2. Scientific Questions

S2 answers only three questions:

1. Can a `global96 + local128` crop representation preserve TAD performance
   relative to a full-frame `dense160` representation at equal input pixels
   and first-layer spatial tokens?
2. Under equal candidate count, center trajectories, trained support, and
   post-seal GT privilege, does variable source width/height provide useful
   headroom over fixed-size continuous-center crops?
3. Is the measured representation path still potentially cost-viable after
   adding one conservative future-selector reserve?

S2 does not train, evaluate, or claim a deployable crop policy. In particular,
it does not contain a content-conditioned ROI head, reinforcement-learning
policy, detector-confidence crop optimizer, or gate-time selector.

## 3. Claims And Anti-Claims

### 3.1 Permitted S2 claims

- `representation_sufficient`: a finite, privileged, search-conditioned crop
  reference is non-inferior to `dense160` on the frozen development gate.
- `variable_size_headroom`: variable width/height improves over a paired
  fixed-size continuous-center reference under equal privilege.
- `prospective_cost_viable`: measured representation cost plus one frozen
  selector reserve satisfies the registered cost bounds.

### 3.2 Forbidden claims

- S2 does not prove a learned selector works.
- A finite candidate population is not an oracle, continuous optimum, or
  global upper bound.
- A negative reference result does not disprove Continuous-RoI.
- A fixed `128x128` local output does not mean the source crop is fixed-size.
- Variable source area does not reduce local-backbone FLOPs.
- No S2 result authorizes official-test access or a paper-ready claim.

## 4. Compared Systems

Three model families are trained for each seed `3407, 3408, 3409`:

| Model | Input | Training geometry | Role |
|---|---|---|---|
| `D160` | full-frame letterbox `160x160` | none | matched dense comparator |
| `G96` | full-frame letterbox `96x96` | none | global-context diagnostic |
| `U128` | `global96 + local128` | common-support geometry mixture | only crop representation model |

There are exactly nine training runs. `U128` has no ROI policy head.

All decision-critical crop cells are evaluated from the corresponding seed's
single `U128` final-EMA checkpoint:

| Cell | Geometry | Selection | Privilege | Primary role |
|---|---|---|---|---|
| `ANCHOR` | center `128x128` source box | fixed | none | fixed-center diagnostic |
| `FS-RAND` | fixed-size, variable center | stable hashed candidate | none | no-selection diagnostic |
| `VS-RAND` | variable center/size/aspect | stable hashed candidate | none | no-selection diagnostic |
| `FS-PREF` | fixed-size, variable center | post-seal per-window join | temporal gate GT | primary fixed-size reference |
| `VS-PREF` | variable center/size/aspect | post-seal per-window join | temporal gate GT | primary variable-size reference |
| `D0-FIX` | historical 21 integer boxes | one fit-selected ID | fit GT only | appendix diagnostic |
| `D0-PREF` | historical 21 integer boxes | post-seal per-window join | temporal gate GT | appendix diagnostic |

The decisive variable-size contrast is `VS-PREF - FS-PREF`. The two arms share
the same center trajectories, candidate count, raw inference budget, proposal
population, privileged join, trained checkpoint, detector, and evaluator.
They differ only in whether area/aspect are fixed at the anchor or vary.

## 5. Continuous Geometry

The frozen source geometry is `H=180`, `W=320`, `q=W/H=16/9`.

For every one of 12 temporal knots, logits are
`s=(s_x,s_y,s_a,s_r)`. Decode:

```text
a = 0.18 + 0.18 * sigmoid(s_a)
r = exp(log(0.75) + log(3) * sigmoid(s_r))
w = sqrt(a * r / q)
h = sqrt(a * q / r)
cx = 0.5*w + (1-w)*sigmoid(s_x)
cy = 0.5*h + (1-h)*sigmoid(s_y)
box = (cx, cy, w, h)
```

The anchor is the S1 center `128x128` source box:

```text
cx0 = cy0 = 0.5
w0 = 0.4
h0 = 0.7111111111111111
a0 = 0.28444444444444444
r0 = 1.0
s0 = (0, 0, 0.3237870770938973, -1.0363260485493035)
```

Twelve knot logits are linearly interpolated in logit space to 48 clips. One
decoded box applies to all 16 frames in a clip. The decoder is the only box
repair mechanism. Post-hoc clamp, rigid translation, integer rounding, and
padding are forbidden.

## 6. Common-Support Training Distribution

`U128` learns a representation under an exogenous, result-independent geometry
mixture. It never predicts geometry.

For successful optimizer update `u` and batch slot `b`:

```text
sample_ordinal = 2*u + b
family = ["anchor", "fixed_size", "variable_size"][sample_ordinal % 3]
```

Across 4,800 successful updates and batch size two, each family occurs exactly
3,200 times.

The stochastic key is:

```text
(training_seed, successful_update, batch_slot, video_id, window_start)
```

The key generates a stateless Philox `12x4` uniform tensor. Each channel is
mapped into bounded knot logits and filtered twice in time with the
replicate-padded kernel `[0.25, 0.50, 0.25]`.

```text
sx = 3.0 * (2*u_x - 1)
sy = 3.0 * (2*u_y - 1)
sa = s_a0 + 1.5 * (2*u_a - 1)
sr = s_r0 + 1.5 * (2*u_r - 1)
```

Family interventions are applied before temporal filtering:

- `anchor`: all four channels equal the anchor logits;
- `fixed_size`: `sx,sy` are stochastic and `sa,sr` equal the anchor;
- `variable_size`: all four channels are stochastic.

The generator state is not a trainable parameter and receives no detector
gradient. AMP retries restore the same batch, stochastic key, model buffers,
RNG, scheduler, EMA, and successful-update count.

## 7. Representation Model

`U128` reuses one `VisionTransformerAdapter` parameter instance and executes it
twice:

```text
global96 -> shared VideoMAE
local128  -> same shared VideoMAE
```

Both evaluations are charged. The pretrained non-adapter core and unused
`fc_norm` stay frozen. All adapters receive gradients from both branches.

The trainable fusion is:

```text
g = channel_layer_norm(Fg)
l = channel_layer_norm(Fl)
u = concat(g, l, l-g)
alpha = 0.25 + 0.50 * sigmoid(conv1d_1152_to_1(u))
delta = conv1d_384_to_384(gelu(conv1d_1152_to_384(u)))
F = channel_layer_norm((1-alpha)*Fg + alpha*Fl + 0.10*tanh(delta))
```

The two fusion convolutions use kernel size one. Fusion has exactly `594,049`
trainable parameters. Global and local temporal occupancy heads each use
`Conv1d(384,20,1)`, for `15,400` additional parameters. Total new `U128`
parameters are exactly `609,449`. There is no ROI-semantic head and no policy
head.

The fused 384-point feature is deterministically interpolated to 768 points
with linear `align_corners=False` semantics, then passed to the inherited
ActionFormer projection, FPNIdentity, ActionFormerHead, losses, proposal
decode, sliding-window aggregation, soft-NMS, and official evaluator.

## 8. Training

- fit population: frozen 160-video split;
- batch size: 2;
- successful updates per epoch: 80;
- epochs: 60;
- exact successful updates: 4,800;
- checkpoint: final EMA only;
- seeds: `3407, 3408, 3409`;
- AMP retries: at most 8 for the same batch;
- scheduler and EMA advance only after a successful optimizer step;
- no intermediate gate metric is used for checkpoint selection.

`U128` loss:

```text
L = L_detector + lambda_g(u)*L_global_aux + lambda_l(u)*L_local_aux
```

Auxiliary targets are temporal class occupancy from fit GT. They are not
spatial boxes or pseudo boxes.

```text
rho(u) = clip((u-800)/1600, 0, 1)
lambda_g = 0.25 - 0.15*rho
lambda_l = 0.50 - 0.30*rho
```

Detector loss must backpropagate through the original head, projection,
fusion, both branch features, and shared adapters. It is not required to
backpropagate into exogenous S2 geometry. The differentiable sampler must
nonetheless pass an isolated finite-difference gradient gate so S3 can reuse
it without a second sampler.

## 9. Paired Candidate Population

The reference population contains one anchor plus 16 Owen-scrambled Sobol
trajectories generated with seed `20260720`.

- Draw shape: `[16,12,4]`.
- Transform and temporal filter: identical to Section 6.
- Candidate `0`: exact anchor.
- Candidates `1..16`: stable IDs ordered by Sobol draw order.
- `FS` and `VS` candidate `k` share exactly the same `sx,sy`.
- `FS` replaces `sa,sr` with the anchor before filtering.
- `VS` retains all four transformed channels.
- Nested populations are prefixes of size `1,5,9,17`.

The population is finite, result-independent, and frozen before training. It
is not optimized by detector confidence and is not called an oracle.

`FS-RAND` and `VS-RAND` select a non-anchor candidate with:

```text
1 + stable_hash(seed, video_id, window_start, family) % 16
```

No GT, teacher, detector prediction, or result cache enters that decision.

## 10. Raw Seal And Privileged Join

All candidate inference runs without gate GT, teacher, oracle, reference ID,
or selected candidate in the process environment, argv, open descriptors, or
Python object graph. Raw predictions, box tubes, candidate IDs, checkpoint
hashes, and manifests are sealed before a separate CPU join can read temporal
gate GT.

For each candidate and sliding window:

1. Use inherited ActionFormer proposals after sigmoid.
2. Keep all finite proposals with score above `0.001`.
3. Keep the top 2,000 class-expanded proposals by score.
4. Preserve duplicates and finite zero-length proposals.
5. Clip each overlapping temporal GT segment to the window and retain its
   class if at least one source frame remains.
6. Greedily match proposals to clipped GT per class, ordered by descending
   score and stable proposal ordinal; each proposal and GT can match once.

The candidate utility tuple is:

```text
(
  matched_gt_count_at_tIoU_0_7,
  matched_gt_count_at_tIoU_0_5,
  sum_matched_tIoU_at_0_5,
  -false_positive_score_mass_top_100,
  -stable_candidate_id
)
```

`false_positive_score_mass_top_100` sums scores of the top 100 unmatched
proposals whose maximum same-class GT tIoU is below `0.1`. Lexicographic
maximum selects the window candidate. The same implementation and tuple are
used by `FS-PREF`, `VS-PREF`, and `D0-PREF`.

After selection, predictions are mapped to original time, merged across
windows, and processed by the inherited soft-NMS and official evaluator.

## 11. Search-Adequacy Gate

`Q` is not detector-confidence convergence. It requires:

1. exact generator hash and paired `FS/VS` center trajectories;
2. all 48 Sobol dimensions have quartile counts `[4,4,4,4]`;
3. all decoded boxes are finite and in bounds;
4. candidate IDs and nested prefixes are complete;
5. median non-anchor pairwise tube IoU is below `0.90` for both families;
6. the post-seal `17` versus `9` candidate privileged result changes
   Avg-mAP by at most `0.25` percentage points and mAP@0.7 by at most `0.50`
   percentage points for both `FS` and `VS`; and
7. at least `75%` of windows selected at size 17 use a candidate already in
   the size-9 prefix.

Failure of items 6 or 7 means the finite reference is inconclusive and may
authorize a new, pre-result candidate-support protocol. It is not evidence
against Continuous-RoI.

## 12. Statistics And Decision Thresholds

Detection uses a paired two-level bootstrap over seeds and gate videos with
20,000 replicates and PCG64 seed `20260720`. All cells share the same draws.
Fixed class support and the inherited official evaluator are mandatory.
Simultaneous one-sided bounds use a max-T family.

`S_VS`, variable-size representation sufficiency, requires all:

| Contrast | Adjusted condition |
|---|---:|
| `VS-PREF - D160` Avg-mAP | LCB `>= -1.00 pp` |
| `VS-PREF - D160` mAP@0.7 | LCB `>= -1.50 pp` |
| Short-Q1 Recall@100@0.7 | LCB `>= -0.03` |
| start boundary error ratio | UCB `<= 1.15` |
| end boundary error ratio | UCB `<= 1.15` |
| per-seed guardrail | no seed Avg delta `< -3.00 pp` |

`S_FS` uses the same conditions for `FS-PREF - D160`.

`H`, variable-size headroom, requires:

| Contrast | Adjusted condition |
|---|---:|
| `VS-PREF - FS-PREF` Avg-mAP | LCB `>= +0.75 pp` |
| `VS-PREF - FS-PREF` mAP@0.7 | LCB `>= +0.25 pp` |
| seed sign guardrail | Avg point estimate positive for at least 2 of 3 seeds |

The result-blind power audit may use only independent historical prediction
evidence or conservative synthetic variance envelopes. It may not use any
S2 prediction, checkpoint, metric, or fit residual. If 80% power at familywise
error at most `0.055` cannot be established, formal raw inference does not
start and the outcome is `NO_DECISION_INVALID_EVIDENCE`.

## 13. Cost Contract

S2 reports three cost ledgers separately:

1. training cost;
2. finite reference sweep and privileged-join cost;
3. prospective deployable cost.

The measured `U128` prospective path includes:

```text
decode
CPU global preprocessing
global H2D
global VideoMAE
external box decode/interpolation
box D2H
runtime crop/resize
local H2D
local VideoMAE
fusion
projection/neck
ActionFormerHead
post-processing/NMS
end-to-end
```

S2 has no measured ROI policy head. Exactly one future-selector reserve is
added to the prospective path:

```text
p50 latency +3.0 ms/window
p95 latency +5.0 ms/window
gross GPU energy +0.50 J/window
peak GPU memory +64 MiB
FLOPs +0.50 GFLOPs/window
parameters +1,000,000
```

For every seed and gate window, one ABBA block contains four total invocations:
two `D160` and two `U128`. Across 129 windows and three seeds this is 1,548
total invocations, split into 774 per arm. Warmups are separate.

NVML uses a persistent sidecar with a 20 ms target and 100 ms fail-closed
maximum gap. The trace is buffered without sampling-loop filesystem I/O and is
published atomically after stop. Gross GPU energy is integrated without idle
subtraction. CPU energy is not claimed.

Cost viability `K` retains the v2 simultaneous upper-bound thresholds after
adding the one selector reserve:

```text
p50 ratio <= 0.90
p95 ratio <= 0.95
gross GPU energy ratio <= 0.92
peak memory ratio <= 1.05
FLOPs ratio <= 0.90
parameters <= D160 + 2M and ratio <= 1.10
```

## 14. Executable N16R4 Contract

Every GPU stage uses the audited site pattern:

- outer allocation: two GPUs and eight CPUs only when required by site memory
  policy;
- exact inner step: one GPU, five CPUs, and 96,000 MiB;
- model process uses logical `cuda:0`;
- never override Slurm's `CUDA_VISIBLE_DEVICES`;
- bind logical CUDA UUID, cgroup-visible NVML UUID, CPU set, and memory limit;
- set `PYTHONNOUSERSITE=1` and use the bound Conda NumPy `1.23.5`.

Formal storage is estimated before namespace creation. The protocol caps its
own namespace at 12 GiB and requires:

```text
free_bytes >= estimated_namespace_bytes + 8 GiB
free_inodes >= estimated_files + 10,000
```

No absolute 512-GiB requirement is used. Only final-EMA checkpoints are kept.
Raw proposals use a compact typed representation with a predeclared top-k.

Every failed or contaminated namespace remains immutable. It is never deleted,
resumed, repaired in place, or reused. A descendant campaign must recursively
bind the parent inventory, receipts, logs, and first failure.

## 15. Outcome State Machine

Define:

- `I`: complete valid provenance, data, training, raw seal, statistics, and
  cost evidence;
- `G`: geometry, source, sampler, gradient, and evaluator gates pass;
- `Q`: paired finite candidate support is adequate;
- `S_VS`: variable-size reference is sufficient;
- `S_FS`: fixed-size reference is sufficient;
- `H`: variable-size headroom passes;
- `K`: prospective cost viability passes.

Evaluate in order:

| Order | Condition | Outcome | Authorization |
|---:|---|---|---|
| 1 | `not I` | `NO_DECISION_INVALID_EVIDENCE` | repair first evidence defect only |
| 2 | `I and not G` | `INCONCLUSIVE_GEOMETRY_OR_SUPPORT` | repair geometry/sampler |
| 3 | `I and G and not Q` | `INCONCLUSIVE_REFERENCE_SUPPORT` | new result-independent support protocol |
| 4 | `I and G and Q and S_VS and H and K` | `SUFFICIENT_AND_VARIABLE_SIZE_HEADROOM` | freeze S3 continuous-policy protocol |
| 5 | `I and G and Q and S_VS and H and not K` | `SUFFICIENT_BUT_COST_NOT_VIABLE` | optimize compute path, no efficiency claim |
| 6 | `I and G and Q and S_VS and not H` | `SUFFICIENT_CONTINUOUS_NO_VARIABLE_SIZE_HEADROOM` | do not justify deformable size policy |
| 7 | `I and G and Q and not S_VS and S_FS` | `SUFFICIENT_FIXED_SIZE_ONLY` | fixed-size location route only |
| 8 | otherwise | `REFERENCE_REPRESENTATION_INSUFFICIENT` | stop S3; do not permanently kill the idea |

No outcome opens official test automatically.

## 16. Implementation Gates

Implementation may begin only after a static validator proves:

- schema and protocol hash are exact;
- `U128` contains no selector or ROI policy head;
- geometry-family counts are exactly balanced;
- all decisive cells are in common training support;
- `FS-PREF` and `VS-PREF` have equal candidate and privilege contracts;
- search adequacy does not use confidence convergence;
- selector cost is charged exactly once;
- ABBA arithmetic is explicit;
- the resource and failure policy matches Section 14; and
- the outcome state machine is total and mutually exclusive.

The implementation sequence is:

1. static protocol validator and focused tests;
2. geometry decoder, tube interpolation, exogenous generator, and samplers;
3. source/global transform and common-support `U128` wrapper;
4. fusion, auxiliary losses, optimizer coverage, and successful-update hook;
5. D160/G96/U128 configs and full-model one-step gate;
6. paired candidate generator, raw seal, privileged join, and analyzer;
7. full-stack profiler and immutable Slurm launchers;
8. clean remote tests and one no-result CUDA Gate;
9. formal nine-run training matrix only after every ancestor passes.
