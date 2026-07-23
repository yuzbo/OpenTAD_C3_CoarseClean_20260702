---
type: idea
node_id: idea:geo-route-adatad
title: "Geometry-Residual-Depth Routing for offline TAD"
stage: implemented
status: implemented_local_pending_cuda_p0
tags: ["offline-tad", "native-token", "roi", "token-selection", "a-mod", "adatad"]
added: 2026-07-22
---

# Geometry-Residual-Depth Routing for Offline TAD

## One-line thesis

Use a continuous ROI trajectory as a low-dimensional, temporally coherent
prior over native spatiotemporal tokens; retain a small free-token residual
budget for disjoint evidence; then use attention-derived Mixture-of-Depths
routing only in later backbone blocks. The unchanged AdaTAD-derived detector
loss is the primary learning signal.

## Design status and boundary

This is a locally implemented candidate awaiting its independent CUDA P0
gate, not a result or authorization to reinterpret the sealed Continuous-RoI
S2 training-only receipt. The held S2 reference protocol and its old
resampled local-tensor path remain immutable historical evidence. No
development metric, official test, cost result, or paper claim exists for this
candidate.

## Why ROI is retained

ROI is not treated as an alternative to token selection. It parameterizes a
structured spatial support distribution with continuous center, scale, aspect
ratio, and temporal trajectory. This provides contiguous local context,
temporal geometric continuity, and potentially block-friendly native-token
packing. A free token selector is retained because one rectangle can miss
disjoint actor, object, and scene evidence.

## Proposed minimal model

1. A lightweight dense global scout observes the full offline video at reduced
   spatial resolution and predicts ROI knots every fixed temporal stride.
2. Interpolated `(cx, cy, w, h)` knots define an elliptical or rectangular
   probability field over the original absolute `2 x 16 x 16` VideoMAE
   tubelet grid. The source pixels are not resampled: selected native tubelets
   preserve their absolute physical coordinates across their two source frames.
3. The geometry branch receives a fixed share `K_geo` of the exact input token
   budget. A residual scorer operating on the scout receives `K_res` tokens
   outside or complementary to that support. A small fixed global-context
   allowance `K_ctx` prevents all scene evidence from vanishing. The union is
   packed to `K_geo + K_res + K_ctx = K_input`.
4. A mask-aware spatial aggregation produces one feature per temporal tubelet,
   preserving the detector-facing `[B, 384, 768]` contract. Absolute position
   and ROI-relative geometry are ablated rather than assumed beneficial.
5. The first representation blocks stay dense. The later tail alternates
   `Dense -> MoD -> Dense -> MoD` rather than turning every remaining block
   sparse: each MoD block consumes the immediately preceding dense block's
   complete attention-derived importance. It preserves a per-tubelet minimum
   and allocates remaining capacity by attention score, so every detector time
   position remains represented. It is a depth-compute mechanism, not an
   input-crop substitute.

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
  embedding, sparse backbone, A-MoD blocks, adapter, detector, NMS, decode,
  preprocessing, H2D, latency, memory, and energy. Patch-token or FLOP counts
  alone cannot establish an efficiency claim.

## Required matched comparison matrix

All variants must share source frames, pretrained VideoMAE initialization,
AdaTAD detector/head/loss, training updates, seeds, token budget, and real
end-to-end cost protocol:

1. P0: dense native numerical/reference route plus ROI/free score-function
   routes; it checks one real CUDA step, real detector losses, exact-K,
   gradients, memory and one-heavy-forward accounting only.
2. P1: dense native, fixed lattice, fixed lattice plus learned geometry
   side-channel, random, free native TokenSelect, geometry/ROI-only, and
   geometry plus residual TokenSelect.
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

If free TokenSelect dominates geometry-only and geometry-plus-residual routing
at both high tIoU and matched end-to-end cost, do not retain ROI as the paper
main claim; pivot to geometry-aware token selection or state the negative
result. If structured routing improves the high-tIoU/cost Pareto, retain the
claim narrowly as geometry-structured token-depth allocation for offline TAD,
not as generic dynamic cropping.

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
