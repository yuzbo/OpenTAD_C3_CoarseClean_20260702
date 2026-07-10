# PhysTime-TAD 2.0 Design Specification

## Status

This specification supersedes `2026-07-10-phystime-tal-design.md`. The earlier
document remains an audit record, but its normalized-time encoder and
Voronoi-style support approximation are not the final method.

## Method Claim

PhysTime-TAD is an offline temporal action detector for irregular video
observations. It is neither a frame selector nor an online detector. Its main
claim is narrow and testable:

> A detector defined as a support-integrated operator on physical time is more
> consistent under observation-grid refinement and missing observations than a
> detector whose temporal operators run on selected-token rank.

The canonical coordinate is seconds. Normalized time, log duration, and support
length may be auxiliary inputs, but may never replace seconds in target
assignment, decoding, NMS, or evaluation.

## Input Contract

For a batch of irregular observations, the detector consumes:

```text
features:              [B, C, K]
timestamps_sec:        [B, K]
support_intervals_sec: [B, K, 2]
valid_mask:            [B, K]
duration_sec:          [B]
domain_start_sec:      [B]
domain_end_sec:        [B]
```

Valid timestamps must be finite and strictly increasing. Every valid support
interval must be finite, non-empty, contain its timestamp, and lie within the
video. Padding carries zero support. Ground truth and predictions use absolute
video seconds. Selected-axis ground-truth remapping and inverse remapping are
forbidden.

Feature-token experiments must retain each token's original ownership interval.
Dropping a token creates a real gap; neighboring supports are never enlarged to
fill it. Raw-video experiments are accepted only when every output token is
derived from one auditable contiguous decoded clip. Rank-adjacent sparse frames
may not be collapsed into a fabricated continuous support interval.

## Support Ownership

Overlapping input supports are clipped by timestamp midpoints to obtain
non-overlapping ownership intervals. Clipping can remove overlap but cannot
expand a support. Therefore missing regions remain missing and temporal mass is
not counted twice.

For query cell `R_q` and observation ownership interval `I_i`, the evidence mass
is

```text
m_qi = length(R_q intersect I_i).
```

A query with zero covered mass is invalid and must emit a finite zero feature.

## Support-Integrated Measure Attention

Each physical query directly aggregates the original observations:

```text
w_qi = m_qi * exp(content_logit_qi + relative_time_logit_qi)
y_q  = sum_i w_qi V(x_i) / (sum_i w_qi + eps)
```

The relative-time features are expressed in seconds and normalized by the query
cell width for numerical conditioning. Temporal mass, rather than token count,
controls an observation's contribution. Under a constant kernel, splitting a
support into identical-feature sub-supports is exactly additive.

No selected-rank self-attention or stride-two token pyramid is permitted before
this projection.

## Physical Query Pyramid

Level `l` uses cells of width

```text
cell_width_l = base_query_spacing_sec * 2**l.
```

Cells are aligned to the global zero-second origin and cropped to the current
window domain. Query count depends on physical duration and spacing, never on
the number of observations `K`. Every pyramid level is projected directly from
the original irregular observations.

## Detection Head

The head predicts, at every valid physical query:

- class logits;
- non-negative left/right distances measured in query-cell widths;
- start and end event intensities.

Decoded boundaries are absolute seconds. Endpoint probabilities integrate an
intensity over the cell:

```text
lambda = softplus(endpoint_logit)
p_event = 1 - exp(-lambda * cell_width_sec)
```

The loss is

```text
L = L_classification + L_regression
    + lambda_endpoint * L_integrated_endpoint
    + lambda_discretization * L_common_coverage_consistency.
```

Consistency is computed only on common valid physical query cells and on
pre-NMS class probabilities, endpoint probabilities, and decoded segments. It
never compares selected-token indices.

Actionness, budget, entropy, max-gap, selector utility, teacher caches, and
offline ledgers are outside this method.

## OpenTAD Integration

`PhysTimeTAD` is a registered detector with its own projection and head. It may
consume pre-extracted features directly or a backbone output only when strict
support metadata is supplied. Training and inference call the same projection
and decoder. The post-processing contract marks predictions as seconds so that
OpenTAD does not apply snippet-axis conversion.

The first formal configuration uses THUMOS I3D feature tokens because their
original temporal ownership cells can be audited. This is the feature-geometry
track, not evidence that raw-video re-extraction is already solved.

## Mandatory Gates

### Gate 0A: Operator properties

- strict validation and padding invariance;
- no support expansion across gaps;
- exact constant-kernel support-split invariance;
- invariance to duplicate token density when total support mass is unchanged;
- finite zero output on uncovered queries;
- physical query count independent of `K`;
- finite gradients through observations, attention, and endpoint intensity.

### Gate 0B: Real model path

- build the registered detector from a real OpenTAD config;
- run one train step and one inference step;
- assert all trainable parameters are covered by the optimizer;
- assert outputs are absolute seconds and require no inverse remap;
- run a remote CUDA smoke before full training.

### Gate 0C: Leakage audit

- sampling and geometry transforms do not read GT to choose observations;
- inference rejects selected-axis GT flags, teacher fields, caches, and ledgers;
- feature-token support provenance is recorded in metadata;
- ambiguous raw sparse-frame-to-token support fails closed.

## Evidence Boundary

Passing software gates proves contract correctness, not paper effectiveness.
The paper claim additionally requires matched THUMOS experiments against
selected-axis ActionFormer, timestamp embedding, linear interpolation, mTAN-like
projection, FrameDrop/TRC, TE-TAD, and LiquidTAD under uniform, random, bursty,
and contiguous-gap observations at matched `K` and compute.
