---
type: design_spec
title: "DUCA-UVT: utility-value portal for dynamic pre-backbone acquisition"
status: user_approved_full_implementation
branch: codex/duca-uvt-utility-value-20260819
base_commit: 7529fba607f8ddfef74d8309efa466d73a956a60
date: 2026-08-19
scope: full_implementation_and_remote_development_deploy
---

# DUCA-UVT: Utility-Value Portal Design (Revised)

## 1. Objective and scope

Add a single learnable utility-value residual `V(t)` to the current DUCA
pre-backbone dynamic acquisition model, and train that residual through three
separately provable mechanisms:

1. train-only GT geometry utility;
2. value-head self-EMA stability distillation;
3. gated detector feedback through a continuous Query Cross-Attention portal.

The hard frame selection remains deterministic. No dense detector teacher is
introduced, and no teacher/cache/GT enters inference.

This revision is authorized by the user for complete implementation and a remote development-seed deployment. It still does not produce or claim mAP, efficiency, plugin generality, or paper readiness. Any terminal result remains diagnostic until prior gates close.

## 2. Relation to paper claims and prior gates

This spec is subordinate to the frozen DUCA contracts and prior gates.

### 2.1 Paper scope

The original broad goal included TVG / TAD / STD components. This spec covers
only one TAD-side pre-backbone component. The following are explicitly
deferred to separate specs and are not implemented here:

- TVG language/video grounding evidence;
- STD spatial actor/object evidence;
- frozen-backbone train-free main arm;
- second detector and second dataset;
- end-to-end latency/energy/memory paper evidence.

Any paper claim requires a separate complete experiment and claim spec.

### 2.2 Prior gates

Before any development seed of this component, the current gap status must be
recorded:

| Gate | Meaning | Required here |
|---|---|---|
| G1 | matched dense/exact-uniform/random baseline | Not closed; no comparison is made in this spec |
| G2 | ST surrogate vs hard one-swap utility alignment | Not closed; portal feedback stays disabled |
| G4 | trained-checkpoint full-stack cost | Not closed; no efficiency claim |
| G6 | requested/effective/unique K ledger | Partially present; must be audited in every run |

Until G1, G2, and G4 are closed in the project, any DUCA-UVT training result is
diagnostic only.

## 3. Frozen design decisions

- Base is `codex/duca-full-official-rerun@7529fba6`.
- The current remote matrix `1244133` must not be modified.
- The network adds exactly one new learned head `V(t)`.
- No separate `frame_value_head` and `budget_value_head`.
- `V(t)` is a signed residual added to the existing fused frame-selection score.
- Dynamic outer-K evidence is the valid-mean of `sigmoid(V(t))`.
- Query tokens are four trainable, class-agnostic tokens:
  `{start, end, interior, background}`.
- No language/text model is implemented in this version.
- Detector feedback enters only through the Query Cross-Attention portal, and
  only after the complete four-layer finite-difference gate passes.
- Hard frame indices are always detached from detector feedback.
- Inference is teacher-free, EMA-free, GT-free, cache-free, and ledger-free.
- Effectiveness is proven per training configuration before any fused claim.

## 4. Method identity

```text
offline TAD
+ pure pre-backbone dynamic acquisition
+ single learnable utility residual V(t)
+ train-only GT geometry supervision
+ value-head self-EMA stability distillation
+ gated continuous detector-feedback portal
+ deterministic boundary-foveated hard sampling
```

The method is not:

- online TAD;
- a new detector;
- the retired three-stage dense-teacher pipeline;
- a multi-score ensemble;
- an end-to-end differentiable claim through hard frame indices.

## 5. Architecture and data flow

```text
dense low-resolution scout
  -> temporal representation z_t
  -> Query Cross-Attention Portal
       query = {start, end, interior, background}
  -> value_evidence -> V(t)
  -> selection_score(t) =
        fused_action_boundary_score(t)
        + alpha_value * V(t)
  -> Boundary-Foveated Decoder
       top fused-score allocation
       + deployable boundary-center neighborhood quota
       + greedy MMR redundancy suppression
       + deployable boundary-pair protection
  -> hard selected physical frames
  -> variable-length VideoMAE / official detector
```

Training-side auxiliary paths:

```text
GT geometry target  -> smooth_l1(V, V_geo_target)
self-EMA value head -> rank/scale distillation, detached
gated detector portal -> only after complete four-layer gate
```

Inference omits all auxiliary paths and portal feedback.

## 6. New modules and interfaces

### 6.1 DucaUtilityGeometryTargets

File: `opentad/models/selectors/duca_utility_geometry_targets.py`

Responsibilities:

- build `V_geo_target` from training GT segments and valid masks;
- build per-window endpoint-pair weights;
- build audit metadata for target provenance.

Interface:

```python
def build_geometry_value_target(
    *,
    gt_segments,
    gt_labels,
    valid_mask,
    boundary_radius,
    short_action_duration_sec,
    short_action_weight,
) -> GeometryValueTarget
```

`GeometryValueTarget` is a dataclass:

```text
frame_target: [B,T] float, zero-mean over valid positions
pair_weight:  [B,T] float, non-negative, valid-masked
audit: dict
```

### 6.2 DucaValueHeadGroup

File: `opentad/models/selectors/duca_value_head_group.py`

The reader currently does not expose a `value_evidence` key. The implementation
must add that key as the scout temporal representation immediately before the
action/boundary/uncertainty/redundancy heads.

Interface:

```python
class DucaValueHeadGroup(nn.Module):
    def forward(self, value_evidence, valid) -> DucaValueOutput
```

`DucaValueOutput` is a dataclass:

```text
value: [B,T] float
provenance: dict
```

Provenance fields:

```text
uses_dense_detector_teacher=False
uses_self_ema_teacher=True/False
uses_gt_geometry_target=True/False
uses_detector_feedback_at_inference=False
uses_raw_prediction_cache=False
```

### 6.3 DucaValueEMA

File: `opentad/models/selectors/duca_value_ema.py`

Responsibilities:

- maintain an EMA of the value head parameters only;
- produce detached distillation targets;
- never enter detector or selector inference forward.

Interface:

```python
class DucaValueEMA:
    def update(self, value_head) -> None
    def state_dict(self) -> dict
    def load_state_dict(self, state) -> None
    def detach_targets(self, value_evidence, valid) -> Tensor
```

The EMA is updated after optimizer steps and is excluded from gradient.

### 6.4 DucaValueLearningLosses

File: `opentad/models/losses/duca_value_learning_losses.py`

The loss module returns a dataclass `DucaValueLossBundle`:

```text
geometry_value_loss: Tensor
self_ema_value_distill_loss: Tensor
gated_portal_value_loss: Tensor
metadata: dict
```

Metadata records all `uses_*` flags and active loss weights.

## 7. Single V(t) semantics and legacy equivalence

`V(t)` is a signed residual:

```text
selection_score(t) = fused_score(t) + alpha_value * V(t)
```

Legacy equivalence is defined at two levels.

### 7.1 Score-level equivalence

When `alpha_value=0`:

```text
selection_score == fused_action_boundary_score
```

### 7.2 Decoder-level equivalence

`V_off` is exactly legacy DUCA only when all new decoder mechanisms are off:

```text
boundary_quota=0
boundary_center_top_m=0
mmr_lambda=0
boundary_pair_protection=False
```

The `V_off` arm and the focused test must enforce this combination.

## 8. GT geometry value target

For each training window:

1. Build raw targets:

```text
raw_target = +1 on endpoint neighborhoods
raw_target =  0 on action interior outside endpoint neighborhoods
raw_target = -1 on background
```

2. Define short-action factor:

```text
short_action_factor =
  clamp(1 - duration / short_action_duration_sec, 0, 1)
```

where `short_action_duration_sec` is a config value, default 2.0 seconds.

3. Pair weighting:

```text
pair_weight = 1 + short_action_weight * short_action_factor
```

4. Normalize to zero valid mean:

```text
V_geo_target = raw_target - mean_valid(raw_target)
```

Because of this normalization, exact numeric values are not preserved. The
contract is sign order and zero mean:

```text
endpoint > interior > background
mean_valid(V_geo_target) = 0
```

5. Loss:

```text
L_geo = mean_valid(smooth_l1(V, V_geo_target) * pair_weight)
```

## 9. Self-EMA stability distillation

Scope:

- EMA only the value head parameters;
- do not EMA action/boundary/uncertainty/redundancy heads;
- do not use EMA in inference.

`V_ema` is not a standalone task-utility proof. A randomly initialized EMA
teacher carries no task semantics. Therefore `V_ema` must:

- initialize its value head from the best available `V_geo` checkpoint;
- be evaluated as a stability regularizer, not as an independent mechanism.

Targets:

```text
V_ema(t) = detached EMA-value-head output
```

First-implementation loss:

```text
L_ema_rank = pairwise margin rank loss over valid positions,
             target order from detached EMA value
L_ema_scale = smooth_l1 on standardized V, prevents collapse
L_ema = L_ema_rank + lambda_scale * L_ema_scale
```

The exact lambda is set in config and is not selected on official validation
mAP.

## 10. Query Cross-Attention portal

### 10.1 Query tokens

Four trainable, class-agnostic tokens:

```text
start
end
interior
background
```

They are initialized as small learned embeddings. No text encoder is used.

### 10.2 Portal forward

Dimensionality is explicit:

```text
z_t:            [B,T,D]
queries:        [4,D]
query_context:  [B,4,D] = CrossAttn(q=queries, kv=z_t)
global_value:   [B,D]   = Linear(mean(query_context, dim=1))
u_t:            [B,T,D] = z_t + global_value[:,None,:]
V(t):           [B,T]   = value_head(u_t).squeeze(-1)
```

### 10.3 Detector feedback path

Detector feedback reaches `V(t)` only through the existing differentiable
rank-transport surrogate applied to `selection_score`.

The implementation must make the detach boundary explicit:

```text
hard_scores = selection_score.detach()
hard_indices = argsort(hard_scores)        # no gradient
soft_scores  = selection_score             # differentiable copy
transport    = hard_one_hot + soft_distribution - soft_distribution.detach()
```

Gradient path:

```text
detector loss
  -> soft rank-transport distribution
  -> selection_score
  -> V(t) and query tokens
```

No gradient flows through hard indices.

### 10.4 Detector feedback gate

Portal feedback is enabled only when the complete frozen four-layer
finite-difference gate passes. The prefilter below is not sufficient:

```text
train-only rank/hard-swap artifact with positive rank agreement
```

The complete gate requires:

1. single-frame swaps;
2. approximately 1% / 5% / 10% dispersed multi-frame swaps;
3. contiguous block swaps;
4. global density steps at 0.25 / 0.5 / 1.0 strength with full hard re-decode.

It must report video-cluster Spearman/Kendall lower bounds, direction accuracy,
top-decile gain, and matched-random regret. Until this gate passes, portal
feedback weight is forced to zero.

Hard frame indices remain detached in all modes.

### 10.5 Audit

Each forward writes:

```text
portal_feedback_enabled: bool
portal_gate_passed: bool
portal_feedback_weight: float
hard_indices_detached: true
```

## 11. Boundary-foveated decoder and MMR

The hard decoder is deterministic and receives the same fused score used in the
current `dynamic_B` path.

### 11.1 Deployable boundary centers

Boundary centers are top-M local maxima of deployable `boundary_logits`.
Inference never uses GT segments.

### 11.2 Deployable boundary-pair protection

If two deployable boundary centers are within `boundary_pair_max_gap` frames,
and one neighborhood is selected, the decoder must include the other
neighborhood before filling by global score. This protects short-action pairs
without using GT at inference.

### 11.3 Exact-K algorithm

```text
inputs:
  K, boundary_quota, boundary_center_top_m,
  boundary_radius_decode, boundary_pair_max_gap

assert boundary_quota <= K
assert boundary_center_top_m >= 0

global_budget = K - boundary_quota
global_picks  = top global_budget by fused score
boundary_centers = top boundary_center_top_m local maxima of boundary_logits
boundary_picks   = union over centers of [b-r, b+r]

pair_picks = deployable pair protection from boundary_picks
candidates = unique(global_picks + boundary_picks + pair_picks)

if len(candidates) < K:
    fill remaining by fused score rank
if len(candidates) > K:
    candidates = greedy_mmr_select(candidates, exactly K)

final = sorted(candidates)
assert len(final) == K
assert unique(final)
assert all(0 <= p < valid_len)
```

### 11.4 Greedy MMR

```text
candidate_score =
  fused_score(c)
  - lambda_mmr * max_similarity(c, already_selected)
```

Similarity uses detached cheap scout features. Tie-break is by smaller physical
position. The final set has exactly K elements.

### 11.5 Constraints

Final selection must satisfy:

```text
sorted, unique, exact K, within valid range,
physical-time positions, no duplicate padding.
```

Short windows and clip alignment use the existing variable-compute rules.

## 12. Dynamic-K interface and calibration

The value head is a raw logit evidence source, not a pre-sigmoid utility.

```text
budget_evidence = mean_valid(sigmoid(V(t)))
```

The existing `marginal_utility_v0` mapping consumes `budget_evidence` as the
utility score directly. The implementation must not apply a second sigmoid.

Before any development seed, `score_midpoint` must be calibrated on training
windows only so that the realized average K matches `average_budget`. The
calibration uses an exponential moving average and is never selected on
validation/test mAP.

## 13. Training configuration arms

All arms use the same single V(t) head.

| Arm | GT geometry | Self-EMA | Portal detector feedback | Decoder |
|---|---|---|---|---|
| `V_off` | off | off | off | legacy decoder off |
| `V_geo` | on | off | off | new decoder on |
| `V_ema` | off, initialized from V_geo | stability only | off | new decoder on |
| `V_geo_ema` | on | on | off | new decoder on |
| `V_geo_ema_portal` | on | on | gated by full four-layer gate | new decoder on |

Fused claims require each active component to have positive incremental
evidence over the immediately simpler arm.

## 14. Config and launcher changes

New config:

```text
configs/adatad/thumos/duca_uvt_value_portal_n16r4.py
```

It extends the current matrix config and exposes:

```text
value_mode
alpha_value
alpha_value_max
value_scale_normalization
boundary_radius
boundary_radius_decode
short_action_duration_sec
short_action_weight
boundary_quota
boundary_center_top_m
boundary_pair_max_gap
mmr_similarity_type
mmr_lambda
portal_feedback_enabled
portal_gate_required
portal_ramp_start
ema_decay
value_loss_weights
```

Launcher:

```text
scripts/run_duca_uvt_value_portal_n16r4.sbatch
```

The launcher is `PRECHECK_ONLY=1` fail-closed by default and requires explicit
gate artifacts before full training.

## 15. Focused verification

### 15.1 Unit tests

- GT target has zero valid mean and sign order:
  `endpoint > interior > background`.
- Score-level legacy equivalence when `alpha_value=0`.
- Decoder-level legacy equivalence only in `V_off` with all decoder mechanisms
  disabled.
- Value head output is `[B,T]`.
- Value provenance fields are complete.
- EMA state does not appear in inference forward.
- Portal gate absent forces zero feedback weight.
- Portal gate present but incomplete four-layer gate also forces zero weight.
- Boundary-foveated decoder preserves exact K, uniqueness, order, valid range.
- Boundary centers are local maxima of deployable boundary_logits, never GT.
- MMR is deterministic, uses detached scout features, and returns exactly K.
- Hard indices remain detached from V(t) gradients.
- Loss module returns `DucaValueLossBundle`, not bare tensors.

### 15.2 Config and static checks

- Config parses for every value mode.
- `py_compile` for changed files.
- `git diff --check`.
- No teacher/cache/GT payload is accepted in `forward_test`.

### 15.3 CUDA one-step gate

Required before any development seed:

- one real VideoMAE/ActionFormer forward with `V_geo_ema` mode;
- finite detector and value losses;
- nonzero value-head gradients from geometry and self-EMA losses;
- portal feedback gradient present only after the complete gate.

The gate also records diagnostic metrics only:

```text
mAP@0.6 / mAP@0.7
short-action stratified recall
boundary distance
requested/effective/executed K
```

These metrics are diagnostic, not claims.

## 16. Out of scope

- formal multi-seed Slurm matrix;
- dense detector teacher utility;
- language/text encoder and TVG evidence;
- STD spatial actor/object evidence;
- DPP with learned kernel;
- low-resolution keyframe/compressed-domain frontend optimization;
- frozen-backbone train-free main arm;
- second detector or second dataset;
- paper claim and efficiency evidence.

## 17. Risks and kill rules

1. If `V_geo` does not improve selection quality or terminal mAP over `V_off`,
   GT geometry value is not promoted.
2. If `V_ema` does not improve stability over `V_geo`, self-EMA is not promoted.
3. If the complete four-layer portal gate fails, portal feedback stays
   disabled; no threshold is lowered.
4. If the decoder violates exact K, uniqueness, or physical order, the run
   fails closed.
5. If MMR changes selection beyond a preregistered overlap bound without
   measurable boundary-recall gain, MMR is removed.
6. If actual average K drifts away from `average_budget`, dynamic K is
   recalibrated before any comparison.
7. If `alpha_value` or its gradient scale dominates the fused score beyond
   `alpha_value_max`, the run fails or value contribution is clipped.

## 18. References

- Uni-AdaFocus method notes exist in the main repository only:
  `research-wiki/papers/wang2024-uni-adafocus.md`
- AdaSpot method notes exist in the main repository only:
  `research-wiki/papers/xarles2026-adaspot.md`
- Retired detector-aware teacher:
  `research-wiki/ideas/detector-aware-teacher.md`
- Current dynamic-B config:
  `configs/adatad/thumos/duca_full_official_dynamic_matrix_v001.py`
- Total-60 rank/direct gate contract:
  `research-wiki/experiments/duca-total60-prebackbone-plugin-cvpr.md`
