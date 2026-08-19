---
type: design_spec
title: "DUCA-UVT: utility-value portal for dynamic pre-backbone acquisition"
status: designed
branch: codex/duca-uvt-utility-value-20260819
base_commit: 7529fba607f8ddfef74d8309efa466d73a956a60
date: 2026-08-19
scope: code_and_focused_gate_only
---

# DUCA-UVT: Utility-Value Portal Design

## 1. Objective

Add a single learnable utility-value residual `V(t)` to the current DUCA
pre-backbone dynamic acquisition model, and train that residual through three
separately provable mechanisms:

1. train-only GT geometry utility;
2. self-EMA distillation of the value head only;
3. gated detector feedback through a continuous Query Cross-Attention portal.

The hard frame selection remains deterministic. No dense detector teacher is
introduced, and no teacher/cache/GT enters inference.

This spec covers code implementation and focused verification only. It does not
authorize a full Slurm training matrix.

## 2. Frozen design decisions

- Base is `codex/duca-full-official-rerun@7529fba6`.
- The current remote matrix `1244133` must not be modified.
- The network adds exactly one new learned head `V(t)`.
- No separate `frame_value_head` and `budget_value_head`.
- `V(t)` is a signed residual added to the existing fused frame-selection score.
- Dynamic outer-K evidence is the valid-mean of `V(t)`, not a second score.
- Query tokens are four trainable, class-agnostic tokens:
  `{start, end, interior, background}`.
- No language/text model is implemented in this version.
- Detector feedback enters only through the Query Cross-Attention portal, and
  only after a train-only rank/hard-swap gate passes.
- Hard frame indices are always detached from detector feedback.
- Inference is teacher-free, EMA-free, GT-free, cache-free, and ledger-free.
- Effectiveness is proven per training configuration before any fused claim.

## 3. Method identity

```text
offline TAD
+ pure pre-backbone dynamic acquisition
+ single learnable utility residual V(t)
+ train-only GT geometry supervision
+ value-head self-EMA distillation
+ gated continuous detector-feedback portal
+ deterministic boundary-foveated hard sampling
```

The method is not:

- online TAD;
- a new detector;
- the retired three-stage dense-teacher pipeline;
- a multi-score ensemble;
- an end-to-end differentiable claim through hard frame indices.

## 4. Architecture and data flow

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
       + boundary top-M neighborhood quota
       + greedy MMR redundancy suppression
       + short-action endpoint-pair protection
  -> hard selected physical frames
  -> variable-length VideoMAE / official detector
```

Training-side auxiliary paths:

```text
GT geometry target  -> smooth_l1(V, V_geo_target)
self-EMA value head -> rank/scale distillation, detached
gated detector portal -> only after rank/hard-swap gate
```

Inference omits all auxiliary paths and portal feedback.

## 5. New modules and interfaces

### 5.1 DucaUtilityGeometryTargets

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
    short_action_weight,
) -> GeometryValueTarget
```

`GeometryValueTarget` contains:

```text
frame_target: [B,T] float, zero-mean over valid positions
pair_weight:  [B,T] float, non-negative, valid-masked
audit: dict
```

### 5.2 DucaValueHeadGroup

The `value_evidence` input is the scout temporal representation immediately
before the action/boundary/uncertainty/redundancy heads. The reader exposes
it under a single `value_evidence` key.

File: `opentad/models/selectors/duca_value_head_group.py`

Responsibilities:

- consume `value_evidence` from the scout;
- output the single `V(t)`;
- expose only one score to the selector decision path.

Interface:

```python
class DucaValueHeadGroup(nn.Module):
    def forward(self, value_evidence, valid) -> DucaValueOutput
```

`DucaValueOutput` contains:

```text
value: [B,T] float
provenance: dict
```

Provenance fields:

```text
uses_teacher=False
uses_ema_at_inference=False
uses_gt_at_inference=False
uses_raw_prediction_cache=False
uses_detector_feedback_at_inference=False
```

### 5.3 DucaValueEMA

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

### 5.4 DucaValueLearningLosses

File: `opentad/models/losses/duca_value_learning_losses.py`

Losses:

- `geometry_value_loss`
- `self_ema_value_distill_loss`
- `gated_portal_value_loss`

All losses are scalar tensors with `uses_*` audit metadata.

## 6. Single V(t) semantics

`V(t)` is a signed residual:

```text
selection_score(t) = fused_score(t) + alpha_value * V(t)
```

Default behavior:

```text
V(t) = 0  -> exact legacy DUCA
```

Dynamic outer-K evidence:

```text
K_evidence = sigmoid(mean_valid(V(t)))
```

The budget mapping remains the existing `marginal_utility_v0` mapping. The
single value head therefore contributes to both frame ranking and dynamic K,
but does not create a second independent score.

## 7. GT geometry value target

For each training window:

1. Endpoint neighborhoods:

```text
distance to nearest GT start or end <= boundary_radius:
    V_geo_target = +1
```

2. Action interior outside endpoint neighborhoods:

```text
V_geo_target = 0
```

3. Background:

```text
V_geo_target = -1
```

4. Short-action pair weighting:

```text
pair_weight = 1 + short_action_weight * short_action_factor
```

5. Normalization:

```text
mean_valid(V_geo_target) = 0
```

Loss:

```text
L_geo = mean_valid(smooth_l1(V, V_geo_target) * pair_weight)
```

## 8. Self-EMA distillation

Scope:

- EMA only the value head parameters;
- do not EMA action/boundary/uncertainty/redundancy heads;
- do not use EMA in inference.

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

## 9. Query Cross-Attention portal

### 9.1 Query tokens

Four trainable, class-agnostic tokens:

```text
start
end
interior
background
```

They are initialized as small learned embeddings. No text encoder is used.

### 9.2 Portal forward

```text
query_context = cross_attention(queries, z_t)
value_evidence = pooled query_context
V(t) = value_head(value_evidence)
```

### 9.3 Detector feedback path

Detector feedback reaches `V(t)` only through the existing differentiable
rank-transport surrogate applied to `selection_score`. The hard top-k indices
are computed from a detached copy of `selection_score`; the soft distribution
used for backward is computed from the differentiable copy. Therefore the
gradient path is:

```text
detector loss
  -> soft rank-transport distribution
  -> selection_score
  -> V(t) and query tokens
```

No gradient flows through hard indices.

### 9.4 Gating rule

Detector feedback through the portal is allowed only when:

1. `portal_feedback_enabled=true` in config;
2. a train-only rank/hard-swap gate artifact exists;
3. the gate indicates positive rank agreement;
4. the training scheduler has reached the portal ramp start.

When the gate is absent or negative:

- portal feedback loss weight is forced to zero;
- only GT geometry and self-EMA may train V(t).

Hard frame indices remain detached in all modes.

### 9.5 Audit

Each forward writes:

```text
portal_feedback_enabled: bool
portal_gate_passed: bool
portal_feedback_weight: float
hard_indices_detached: true
```

## 10. Boundary-foveated decoder and MMR

The hard decoder is deterministic and receives the same fused score used in the
current `dynamic_B` path.

### 10.1 Boundary-foveated allocation

```text
global_budget = K - boundary_quota
global picks   = top global-budget by fused score
boundary centers = top-M local maxima of the existing deployable
                   boundary_logits, never GT in inference
boundary picks = neighborhood union of [b-r, b+r]
final set      = sorted(unique(global picks + boundary picks))
```

If the final set has fewer than K frames, fill with the remaining top-ranked
valid frames.

### 10.2 Greedy MMR redundancy suppression

For selected candidates beyond K, use deterministic greedy MMR:

```text
score - lambda_mmr * max_similarity(selected, candidate)
```

Similarity is computed from cheap scout features, not heavy detector features,
and is deterministic.

### 10.3 Constraints

Final selection must satisfy:

```text
sorted, unique, exact K, within valid range,
physical-time positions, no duplicate padding.
```

Short windows and clip alignment use the existing variable-compute rules.

## 11. Training configuration arms

Effectiveness is proved separately for each learning mechanism. All arms use
the same single V(t) head.

| Arm | GT geometry | Self-EMA | Portal detector feedback |
|---|---|---|---|
| `V_off` | off | off | off |
| `V_geo` | on | off | off |
| `V_ema` | off | on, with scale anti-collapse | off |
| `V_geo_ema` | on | on | off |
| `V_geo_ema_portal` | on | on | gated |

Fused claims require each active component to have positive incremental
evidence over the immediately simpler arm.

## 12. Config and launcher changes

New config:

```text
configs/adatad/thumos/duca_uvt_value_portal_n16r4.py
```

It extends the current matrix config and exposes:

```text
value_mode
alpha_value
boundary_radius
short_action_weight
boundary_quota
boundary_radius_decode
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

## 13. Focused verification

### 13.1 Unit tests

- GT target: zero-mean, endpoint positive, interior neutral, background
  negative, short-action pair weighting.
- V(t)=0 equivalence to legacy fused score.
- Value head provenance fields.
- EMA state does not appear in inference forward.
- Portal gate absent forces zero feedback weight.
- Portal gate present allows configured ramp weight.
- Boundary-foveated decoder preserves exact K, uniqueness, order, valid range.
- MMR is deterministic and does not access detector features.
- Hard indices remain detached from V(t) gradients.

### 13.2 Config and static checks

- Config parses for every value mode.
- `py_compile` for changed files.
- `git diff --check`.
- No teacher/cache/GT payload is accepted in `forward_test`.

### 13.3 CUDA one-step gate

Required before any development seed:

- one real VideoMAE/ActionFormer forward with `V_geo_ema` mode;
- finite losses for detector and value losses;
- nonzero gradients for value head from geometry and self-EMA losses;
- portal feedback gradient present only in portal mode after gate.

This gate is not a performance result.

## 14. Out of scope

- full Slurm training matrix;
- dense detector teacher utility;
- language/text encoder;
- DPP with learned kernel;
- low-resolution keyframe/compressed-domain frontend optimization;
- frozen-backbone train-free main arm;
- second detector or second dataset.

## 15. Risks and kill rules

1. If `V_geo` does not improve selection quality or terminal mAP over `V_off`,
   GT geometry value is not promoted.
2. If `V_ema` does not improve over `V_geo`, self-EMA is not promoted.
3. If portal gate fails, portal feedback stays disabled; no threshold is lowered.
4. If fused decoder violates exact K, uniqueness, or physical order, the run
   fails closed.
5. If MMR changes selection beyond a preregistered overlap bound without
   measurable boundary-recall gain, MMR is removed.
6. If actual average K drifts away from the declared target budget, dynamic K
   is recalibrated before any comparison.

## 16. References

- Uni-AdaFocus notes: `research-wiki/papers/wang2024-uni-adafocus.md`
- AdaSpot notes: `research-wiki/papers/xarles2026-adaspot.md`
- Retired detector-aware teacher:
  `research-wiki/ideas/detector-aware-teacher.md`
- Current dynamic-B config:
  `configs/adatad/thumos/duca_full_official_dynamic_matrix_v001.py`
- Total-60 rank/direct gate contract:
  `research-wiki/experiments/duca-total60-prebackbone-plugin-cvpr.md`
