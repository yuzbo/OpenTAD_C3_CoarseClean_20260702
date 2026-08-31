# Continuous-RoI S2 Reference Protocol v2.2

## Status and scope

This document is the result-blind corrigendum for the historical S2 v2.1
reference design.  It does not change a model, checkpoint, training config,
dataset, evaluator, NMS rule, metric, or threshold.  It authorizes neither raw
inference nor official-test access.  Its sole purpose is to decide whether a
future development-only fixed-size versus variable-size reference sweep can be
made interpretable and reproducible.

- Task:
  `ZOOMTOKEN-CONTINUOUS-ROI-S2-V2.2-REFERENCE-PROTOCOL-CORRIGENDUM-AND-KNOWN-ANSWER-CLOSURE-v001`.
- Execution base: `9e774eeeeb0e325ebcfc7abd453cb4d2bf1a4ddb`.
- Parent v2.1 protocol SHA-256:
  `ef806b7cd37c704d14a54211b1d4e2f9fb88b75599da918272cc6acad157b3af`.
- Canonical v2.2 protocol SHA-256:
  `644f0c5648e0f5be004db3a3e7240a8f24a3c1d561f933502bccb8dca200cb46`.
- Candidate-manifest known-answer SHA-256:
  `ad7e4ba19ea960e77d69c97d8cd8d692275758b19c7526c8206cfa4f9547cb17`.
- Annotation-free 129-window population SHA-256:
  `e46e42fb886f898bdb43e21020c8c10b9305a6b313f8b4f7efc8bcb02efa617e`.
- Role contract: `KEEP`.

Passing this protocol means only that a fresh Pro may consider authorizing the
development reference sweep.  It is not crop-sufficiency, accuracy, cost,
generalization, or paper evidence.

## Shared physical centers

The v2.1 statement that FS and VS share `sx,sy` is withdrawn.  Under the frozen
bounded-center decoder,

```text
center = size/2 + (1-size) sigmoid(s_center),
```

equal logits do not imply equal physical centers when sizes differ.  v2.2
therefore constructs one shared physical center and independently inverts the
decoder for each arm:

```text
s_center_arm = logit((shared_center-size_arm/2)/(1-size_arm)).
```

For every tubelet, the common center is sampled inside the intersection of the
FS and VS feasible center intervals.  Equivalently, the center-fraction is
mapped through the larger arm size on each axis.  This changes neither trained
weights nor training geometry; it corrects only the unexecuted reference
population.

The `size == 1` boundary is explicit.  It has the unique valid center `0.5` and
uses center logit `0`.  Any other center at size one is invalid.  There is no
division by zero, epsilon substitution, implicit clamp, translation, padding,
or rounding.

Twelve result-independent controls are filtered twice with replicate-padded
`[0.25, 0.50, 0.25]`.  Control quantities are interpolated to 48 tubelets with
linear `align_corners=True` semantics.  Arm-specific center logits are derived
after the per-tubelet sizes and common physical centers exist; this is what
makes equality hold at all 48 tubelets rather than only at 12 knots.

## Frozen candidate generator

The implementation is exactly `torch.quasirandom.SobolEngine` from Torch
`2.0.1`:

```text
dimension = 48
scramble = true
seed = 20260720
skip = 0
draw = one call for 16 rows with dtype torch.float64
reshape = [16,12,4]
```

Candidate `candidate-000` is the exact anchor.  Candidates `001..016` follow
the single Sobol draw order.  There are no repeated draws, capacity sweeps,
candidate-count sweeps, confidence optimization, or result-dependent ordering.
The four channels use the v2.1 bounded-logit transforms before the frozen
filter.  FS fixes area/aspect at the anchor; VS retains the transformed
area/aspect.  Both arms use the common-center construction above.

The complete 17-by-48 manifest stores float64 values as 17-significant-digit
decimal strings and is serialized as compact, sorted-key UTF-8 JSON.  Its
canonical SHA-256 is the candidate known answer listed above.  The validator
must reproduce the bytes with a second clean generation; version drift is a
protocol failure, not permission to update the hash.

## Frozen development population

The source manifest is the immutable S1 v4 manifest with file SHA-256
`8e5a8901cb24b735750d5766405996dcac022b37f5a79fdbbdaa1f5479bf141d`
and semantic SHA-256
`10b14faac57d4631dfae93c9a7d14eb81b8dc308f0e80232469e5b7c974589ca`.
The development database SHA-256 is
`0985d3711ab31f404ff0be5a1ba75420796a6807d486410337078b38090bf749`.
The fit/gate split counts are `160/40`, with hashes frozen in the JSON
protocol.  Official test remains sealed.

The v2.1 gate construction is mechanically replayed once on CPU to freeze the
raw population: feature stride 4, sample stride 1, window size 768, overlap
0.5, last-window realignment, and strict completeness `>0.75`.  This produces
129 ordered windows.  Its emitted raw manifest contains only video/window/media
coordinates and the gate split identity; it contains no annotation, labels,
segments, targets, teacher data, or preferred ID.  The development database is
used only by the privileged protocol-freeze process and is not an input to a
future raw inference process.

## Raw/privileged separation

The raw process may receive only the candidate manifest, sanitized raw
population manifest, rendered config, checkpoint, and media.  Its result-blind
enumerated `candidate_id` is permitted and required.  Annotation, ground-truth
segments, target caches, teachers, and `preferred_candidate_id` are forbidden
from its argv, environment, input roles, object graph, and output schema.

A raw payload is sealed by canonical SHA-256 before any privileged process can
open development GT.  The separate CPU join must verify that exact seal before
it can create preferred IDs.  A changed raw payload fails the join.  FS-PREF,
VS-PREF, and D0-PREF share the same join implementation, candidate privilege,
and search timing.

The 0.7 and 0.5 greedy matching states are independently reinitialized.
Proposals are ordered by descending score then stable proposal ordinal.  The
utility tuple, false-positive state, half-open frame intervals, and final
negative stable-ID tie-break are fully enumerated in the JSON protocol.

## Frozen diagnostics and statistics

- D0 consists of the 21 half-open `128x128` boxes from the Cartesian order
  `y0=[0,26,52]`, then `x0=[0,32,64,96,128,160,192]`, with
  `q=7*k+j`.  It is a diagnostic, never a continuous-space upper bound.
- Short-Q1 is fit duration `<=1.5 s`; Recall@100 is per-video, after full-video
  NMS, same-class one-to-one matching at tIoU `>=0.7`.  It requires at least
  30 GT and 8 classes.
- Boundary start/end errors use score-ordered same-class one-to-one matching at
  tIoU `>=0.5`, normalize by `max(duration,1e-6)`, cap at one, assign unmatched
  GT error one, and aggregate per video before the video-cluster statistic.
- The paired two-level bootstrap uses 20,000 PCG64-seed-20260720 replicates:
  sample three seeds with replacement, then 40 gate videos per selected seed.
  Windows, proposals, and instances are not resampling units.
- All hypotheses are re-expressed as higher-is-better before max-T.  Detection
  and cost are separate families.  Zero bootstrap standard error or missing
  evidence yields `NO_DECISION_INVALID_EVIDENCE`.
- Pairwise tube IoU is the arithmetic mean of the 48 same-tubelet spatial IoUs.
  Search diversity is the median over all 120 unordered pairs among the 16
  non-anchor candidates, separately for FS and VS.  An invalid box invalidates
  the evidence.

No threshold in this section was selected from S2 predictions.

## Exact-nine read-only binding

The formal closure binds the existing D160/G96/U128 by seeds
`3407/3408/3409`: Jobs `1177668..1177676`, their rendered configs, completion
receipts, final-EMA checkpoint bytes and metadata sidecars, and the immutable
campaign/deployment receipts.  Every cell is fixed at 60 epochs, 4,800
successful updates, and `state_dict_ema`.  The canonical training-only matrix
receipt has file SHA-256
`14e0fac382e2bd5b570a7a9240258d34b83b6e16129e924bc343705fea0446db`
and internal SHA-256
`9eedfa1e3d30af7be2902325e589b4898cff326fcd200f509210e36b8c37dda5`.

Receipt hashes alone do not substitute for missing checkpoint bytes.  A
missing or changed checkpoint, sidecar, config, completion receipt, source
manifest, development database, or population known answer is a terminal
identity blocker.

## Closure and terminal states

The only formal action is one clean Linux CPU-only invocation of
`tools/validate_continuous_roi_s2_v2_2_protocol.py`.  It performs static
semantic validation, regenerates both known answers, checks raw/privileged
separation, and verifies every external identity without loading a checkpoint
into a model.  It does not run training, GPU inference, prediction, metrics,
cost measurement, or official test.

If all identities and known answers pass, the terminal is:

```text
CONTINUOUS_ROI_S2_V2_2_PROTOCOL_READY_FOR_FRESH_PRO
```

Any missing identity, byte mismatch, generator instability, unequal center,
privilege leak, or rule ambiguity yields:

```text
STOP_CONTINUOUS_ROI_S2_REFERENCE_ROUTE_BEFORE_INFERENCE
```

Either terminal requires one fresh exact-Project Pro review before any raw
reference inference or successor task.
