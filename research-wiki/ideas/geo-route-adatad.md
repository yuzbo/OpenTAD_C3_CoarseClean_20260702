---
type: idea
node_id: idea:geo-route-adatad
title: "NativeTokenSelect-first routing for offline TAD"
stage: experiment_running
status: p0r_submitted_priority_pending
tags: ["offline-tad", "native-token", "token-selection", "geometry", "adatad"]
added: 2026-07-22
---

# NativeTokenSelect-first Routing for Offline TAD

## One-line thesis

First establish detector-supervised, ROI-free exact-K selection of valid
source-native tubelets with coordinate-lineage packed temporal adaptation.
Only if that base survives matched controls may continuous geometry be retained
as a structured add-on. The unchanged AdaTAD-derived detector loss is the
primary learning signal.

## Design status and boundary

The correctness replacement is implemented; clean-commit remote focused tests
pass and P0R is submitted but still priority-pending. Historical P0
`PASS_MECHANICAL_ONLY` belongs to
the superseded implementation, and historical P1 failed on storage before any
result. Neither authorizes the replacement. The held Continuous-RoI S2
training-only receipt and its resampled local-tensor path remain immutable
historical evidence. No replacement development metric, official test, cost
result, empirical support, or paper claim exists.

## Why geometry is conditional

Geometry parameterizes a structured spatial support distribution with
continuous center, scale, aspect ratio, and temporal trajectory. It may provide
contiguous context and temporal coherence, but those benefits are hypotheses.
The primary base is a truly geometry-free selector because one rectangle may
miss disjoint actor, object, and scene evidence. Geometry is retained only if a
corrected hybrid strictly adds under the same validity, pooling, adapter,
budget, and total-cost contract.

## Proposed minimal model

1. A lightweight dense global scout observes the full offline video at reduced
   spatial resolution and predicts ROI knots every fixed temporal stride.
2. The native support follows the pretrained floor Conv3d semantics. For
   180x320 input it uses 176x320, an 11x20 grid, and a boolean validity mask;
   it creates no replicated or interpolated patch support.
3. The ROI-free base fixes geometry to the full frame and freezes the geometry
   head. Its residual scorer selects exact-K valid tubelets. Geometry-enabled
   variants use interpolated `(cx, cy, w, h)` knots to define a structured
   field over the same absolute native grid.
4. In hybrid, the geometry branch receives a fixed share `K_geo` of the exact input token
   budget. A residual scorer operating on the scout receives `K_res` tokens
   outside or complementary to that support. A small fixed global-context
   allowance `K_ctx` prevents all scene evidence from vanishing. The union is
   packed to `K_geo + K_res + K_ctx = K_input`.
5. Attention, MLP, and the original temporal Adapter operate on packed selected
   tokens. The Adapter preserves absolute spatial lineage across adjacent
   tubelets; absent lineage neighbors are zero and full-K matches dense.
6. Every P1 arm uses uniform-selected aggregation, producing one feature per
   temporal tubelet and preserving the detector-facing `[B,384,768]` contract.

A-MoD is not part of the minimal model or P1R. It is a conditionally gated P2
extension only after a learned route survives the primary controls.

ToMe-style merging is not part of the initial main model. It is an important
matched baseline and may be restricted to low-value global-context tokens in a
later ablation; merging ROI-local tokens risks blurring the fine spatial detail
needed for high-tIoU temporal localization.

## Learning and cost rules

- The primary objective is the audited AdaTAD-derived Focal classification and
  DIoU regression loss; this configuration has no independent quality loss.
  There is no spatial GT, teacher, oracle, test signal, or manual ROI in the
  policy path.
- Native-token selection uses differentiable soft support only for the relaxed
  dense path. Hard exact-K membership has no ordinary pathwise ROI derivative;
  P0 must compare a score-function hard-policy estimator with a clearly marked
  biased straight-through surrogate. Nonzero gradients alone do not establish
  a useful routing mechanism.
- Geometry constraints only prevent degenerate boxes and implausible temporal
  jitter. They may not replace detector supervision or encode a hand-designed
  action prior.
- The full cost ledger includes scout, geometry/residual routing, native patch
  embedding, packed backbone and Adapter, detector, NMS, decode,
  preprocessing, H2D, latency, memory, and energy. Patch-token or FLOP counts
  alone cannot establish an efficiency claim.

## Required matched comparison matrix

All variants must share source frames, pretrained VideoMAE initialization,
AdaTAD detector/head/loss, training updates, seeds, token budget, and real
end-to-end cost protocol:

1. P0R: dense native numerical/reference route plus score-function and corrected
   hybrid
   routes; it checks one real CUDA step, real detector losses, exact-K,
   validity, full-K parity, packed Adapter accounting, gradients, memory,
   component trace, storage profile, and one-heavy-forward accounting only.
2. P1R: dense native, fixed lattice, fixed lattice plus learned geometry
   side-channel, random, free native TokenSelect, geometry/ROI-only, and
   geometry plus residual TokenSelect. Pooling and Adapter execution are
   identical across arms.
3. P2: promote only the P1 winner to three seeds and budgets. ToMe and
   A-MoD are separately gated extensions, never assumptions embedded in the
   P1 primary claim.
4. P3: freeze a surviving configuration before a second detector/dataset and
   a one-time sealed official test.

Report Avg-mAP and mAP at tIoU 0.3--0.7, especially 0.6/0.7, short-action and
boundary diagnostics, exact token/depth utilization, per-tubelet coverage,
ROI temporal stability, and measured end-to-end latency/memory/energy.

## Paper evidence package

The theory package establishes conditional operation-count, score-function,
and structured-approximation statements only; it makes no theorem about mAP
or wall-clock speed. The paper figure tooling renders source-bound
architecture, Pareto, high-IoU, budget, ablation, diagnostic and stability
figures plus raw-seed and LaTeX tables from validated records outside the
repository. The new fixed-lattice-plus-geometry control is mandatory before
attributing a gain to changing native-token support rather than merely
injecting a learned geometry embedding.

## Decision rule

First require `free` to beat fixed lattice, random, and the
fixed-lattice-plus-geometry side-channel on the frozen high-tIoU accuracy rule
and cost less than dense. Failure stops learned routing. Only after this base
passes may hybrid geometry be considered; it must beat free, random, and the
geometry side-channel without greater total cost. A pass promotes Route A;
otherwise the simpler Route B advances. Even a Route A pass is native-token
geometry routing, not generic dynamic cropping or a sequential pixel zoom.

## A-MoD correction

A-MoD is a valid pretrained-model adaptation baseline, not merely a
from-scratch language-model technique. Its attention-derived routing reports
no additional trainable routing parameters and adaptation from pretrained
transformers. This motivates a VideoMAE compatibility experiment, but does not
prove compatibility, speed, or localization benefit in AdaTAD/TAD.

For this candidate, the required schedule is an initial dense prefix followed
by alternating Dense-MoD pairs. A-MoD must score a MoD block from the full
attention state of the preceding dense block; a consecutive all-MoD tail would
not preserve that paper-level premise and is not the intended comparison.

## FlashVID transfer boundary (2026-07-23)

FlashVID is now recorded as a relevant video-token-compression reference, but
not as an empirical precedent for GeoRoute. Its reported 90% visual-token
reduction with 99.1% retained score is a LLaVA-OneVision VLLM result: 57.9
versus 58.4 average score under an aligned 10% retention-budget protocol. It
is neither 99.1% absolute accuracy nor TAD mAP.

Its useful hypothesis is narrower: a selector should jointly preserve task
relevance, feature diversity, and motion-tolerant cross-frame correspondence.
FlashVID itself runs a full vision encoder to obtain features and attention,
then compresses for the LLM under `torch.no_grad()`. It therefore cannot be
ported as a pre-backbone AdaTAD efficiency method and cannot support our
detector-gradient or native-token claims.

P1 is deliberately unchanged. If and only if the P1 hybrid survives against
free TokenSelect, P2 may test a scout-only, FlashVID-inspired
relevance-diversity-correspondence residual baseline with exact-K lineage and
one-heavy-forward accounting. It must be labelled an adaptation, not a
FlashVID reproduction, and is removed if it loses on high-tIoU or total cost.

## External v1 review absorption (2026-07-23)

The archived external review is `HOLD`, not an implementation acceptance. Its
code findings are accepted: the current U128 route uses fixed-output
`grid_sample`, evaluates VideoMAE twice, contains no learned policy, and the
current ActionFormer configuration exposes Focal plus DIoU rather than a
quality loss. Its insistence on a one-heavy-forward native-tubelet P0 and the
Dense-MoD interval requirement is accepted.

The following implementation choices remain hypotheses rather than frozen
model facts: a review-proposed 48-knot / 16-source-frame cadence, `K=64`, a fixed 96-pixel scout,
a CPU-pinned source gather, a dense-scatter sparse adapter, a 4,800-update
schedule, and numerical latency or mAP thresholds. A score-function estimator
is mathematically honest for a stochastic hard policy, but must be measured
against a labelled straight-through surrogate for variance, detector utility,
and total cost before it is made the main algorithm. Semantic violations kill
the claim; an early gather or adapter bottleneck is a HOLD/pivot condition, not
automatic evidence that the research hypothesis is false.

## Connections

Relations are maintained only in `research-wiki/graph/edges.jsonl`.
