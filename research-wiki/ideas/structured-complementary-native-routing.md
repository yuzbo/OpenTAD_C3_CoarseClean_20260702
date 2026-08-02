---
type: idea
node_id: idea:structured-complementary-native-routing
title: "Structured Complementary Native Routing for offline TAD"
stage: tested
status: pilot_terminal_finalizer_input_failure_no_empirical_support
tags: ["offline-tad", "native-token", "roi", "token-selection", "structured-routing", "adatad"]
added: 2026-08-02
updated: 2026-08-02
---

# Structured Complementary Native Routing for Offline TAD

## Purpose

Test whether the strongest old descriptive signal came from a fixed
decomposition of source-native support into deterministic context,
geometry-conditioned ROI, and complementary residual evidence, rather than
from arbitrary Free TokenSelect or detector-visible geometry representation.

This route directly develops ROI + TokenSelect Hybrid. It does not return to a
Free-first selector, and it does not assume the old Hybrid result already proves
the mechanism.

The frozen `8/28/28`, exact-K64 realization is not the intended final model. It
is a causal probe that holds compute and role counts fixed while testing whether
ROI and residual evidence have identifiable utility. The user-restated final
objective retains the continuous temporal ROI trajectory and makes the total
heavy-token budget `K_t` as well as the evidence-role allocation temporally
adaptive. The user explicitly rejected fixed total K with only a dynamic role
split and accepted the native two-frame VideoMAE tubelet as the decision unit,
giving 384 dynamic decisions over the 768-frame window.

Budget freedom is deliberately staged. The first dynamic implementation keeps
an exact configurable window total `sum_t K_t=B` so utility comes from learned
redistribution rather than extra compute. Content-dependent window totals are a
separate successor only after redistribution passes matched controls.

Context is not a permanent deterministic anchor in the intended model. Its
count is jointly allocated with ROI and residual evidence at every tubelet; no
fixed context floor or post-hoc uniform repair is allowed.

Zero heavy-token tubelets are allowed in Stage 1. Their carrier semantics must
be explicit and excluded from the executed-heavy-K ledger. A separately trained
`K_t>=1` ablation will test whether full heavy temporal coverage is necessary.
The selected main carrier is masked zero: scatter an all-zero heavy feature for
that tubelet and propagate an explicit heavy-valid mask so downstream temporal
aggregation cannot mistake absence for an observed heavy token. It carries no
scout content and consumes no heavy execution. `learned-null` and
`scout-projection` are independent, separately trained carrier ablations rather
than inference-time substitutions on the masked-zero checkpoint.

The user approved a global exact-window-budget allocator rather than a separate
count-then-select hierarchy. Each physical native token has three competing
context/ROI/residual role utilities but can enter the hard union at most once;
one global projection selects exactly B physical tokens over the 384x220
window. Per-tubelet `K_t` and all role counts therefore emerge from the union,
including valid zeros. The detailed ragged VideoMAE executor remains under
design review because the implemented packed path assumes constant per-tubelet
K; padding that path would not be genuine exact-B heavy computation.

The approved Stage-1 cost contract keeps this projection global over the whole
window. Exact B denotes exactly B unique selected and heavy-executed physical
tokens; it does not assert equal attention FLOPs across routes. If `b_c` is the
executed-token count in native 16-frame VideoMAE clip `c`, the executor must
receipt `P=sum_c b_c^2` independently from B, as well as actual ragged bucket,
patch-embedding, attention, MLP and coordinate-lineage Adapter calls and
end-to-end p50/p95. A fixed per-clip quota is not the final main design because
it would suppress the intended full-window temporal adaptivity. Advancement is
conditional on a measured cost/Pareto gate, and padding/dummy tokens cannot be
reported as exact-B execution.

## Uni-AdaFocus-informed role and training design (`designed`)

The 2026-08-02 paper-and-official-code audit corrects two potential
misreadings. Uni-AdaFocus does not train its selector with Plackett-Luce or
reinforcement learning. It conditions spatial and temporal policies on a cheap
global encoder, detaches those global features at the policy boundary, trains
continuous geometry through a deep-feature-interpolation auxiliary
classification loss, and trains temporal weights through a differentiable
Monte Carlo decomposition of expected frame loss. Its hard focus indices and
local crop actions are detached in the heavy path.

The user approved removal of the independent `q_ctx(t,n)` head. The cheap scout
representation is already the full-window context used to condition allocation.
The approved lower-redundancy parameterization is

`u_hard(t,n) = q_base(t,n) + max(0, delta_roi(t,n), delta_res(t,n))`,

followed by the already approved global physical-token top-B. Backward uses
`u_soft=q_base+tau*logsumexp((0,delta_roi,delta_res)/tau)` as the ST relaxation.
The operational role is the hard argmax over `(0,delta_roi,delta_res)`, so the
context count remains dynamic without a separately symmetric context scorer;
ROI and residual modifiers remain directly ablatable. Role IDs stay
support-only. A future role-specific representation would require a new design
and would reopen the need for a distinct context head.

The approved main estimator family is correspondingly Uni-inspired rather than
a literal port: keep the exact hard selected support in the detector forward;
use a stop-gradient coarse-feature surrogate to provide low-variance gradients
for continuous ROI and soft allocation; keep detector supervision TAD-specific;
and retain PL only as a separately trained matched estimator ablation. The
immutable fixed-pilot recovery remains required evidence but cannot select the
dynamic main estimator. This design does not import resized crops, fixed focus
counts, classification-only frame loss, the full-frame size penalty, or
conditional exit. The role/estimator family is `designed`, not implemented or
tested.

The user approved the training and gradient boundary. Let `Z` be the cheap
full-window scout representation. A train-only auxiliary TAD head jointly trains
the scout from fit/train supervision; this is not a separately pretrained or
cached coarse classifier. The policy consumes `stopgrad(Z)`. The true detector
loss remains active throughout training on the unique global hard top-B and true
ragged heavy path. Its ST bridge updates the route heads at hard-selected tokens,
while the detector and heavy backbone remain learnable independently of that
bridge; it does not provide a dense counterfactual for unselected tokens.

The dense counterfactual is a backward-only global soft-budget projection over
the same physical candidates, constrained to `0<p(t,n)<1` and
`sum_(t,n) p(t,n)=B`. It softly aggregates detached `Z` into a training-only TAD
proxy. The proxy loss updates the policy and auxiliary head but not the scout,
heavy backbone, or hard execution. It is absent at inference and cannot be
reported as selected/executed B. Its weight follows successful optimizer steps:
it is active early, then annealed to zero before a final hard-path-only phase.
No area, coverage, expected-cost, fixed-context, or fixed-`K_t` loss is part of
the main Stage-1 objective. Exact schedule constants, degeneration safeguards,
error handling, ablations, and tests remain under section-by-section review.

### Formula-level Uni-AdaFocus comparison (`designed`)

The paper states a deformable patch as centre coordinates plus height and width,
but official commit `88464883` implements an equivalent top-left form. With four
sigmoid actions `(l_y,l_x,s_h,s_w)`, its 224-pixel path computes
`H_p=96+(224-96)s_h`, `W_p=96+(224-96)s_w`,
`y_0=l_y(224-H_p)`, and `x_0=l_x(224-W_p)`, before resizing the
selected source rectangle to a fixed local-CNN input. Hence
`c_x=x_0+W_p/2` has exactly the bounded-interval structure
`c_x=w/2+(1-w)l_x` after normalization. The earlier proposed equation is not a
new geometric identity; its substantive change is replacing Uni's fixed
`96/224` minimum by a native-grid floor `1/W_grid,1/H_grid`.

Uni's spatial proxy interpolates a crop from detached global feature maps and
optimizes auxiliary classification. The paper identifies size collapse under
that proxy and adds Eq. 15,
`alpha[(H-H_p)^2+(W-W_p)^2]`; the code equivalently penalizes normalized size
actions away from one. Its hard local crop action remains detached. SCNR-TAD
instead selects unique native physical tokens under exact B, does not resize an
ROI into a second fixed local input, and needs high-IoU TAD rather than video
classification supervision. Therefore the fixed 96-pixel floor and
full-frame-seeking penalty should not be copied automatically. The user approved
the task-native replacement: derive independent axis floors at runtime as
`w_min=1/W_grid` and `h_min=1/H_grid`, retain no size/area/coverage penalty in
the main setting, and compare it only with a separately trained matched 2x2-cell
floor sensitivity arm. The shared geometry primitive now implements an explicit
`native_cells` mode while its default `static_normalized` mode preserves old
config behavior. Exact source `4be718449033e95dc6d15029ec4ef889397c9066`
passed the focused geometry/routing suite `36/36` and the complete GeoRoute plus
required C3 regression set `194 passed, 1 skipped` in a clean N16R4 Linux/Torch
snapshot. This is component-level `tested`, not a complete dynamic Stage-1
route, empirical floor optimum, performance result, or paper evidence.

## Principle

For every one of 384 native tubelets over an 11x20 patch grid, select exactly
64 valid patches as 8 deterministic context + 28 ROI + 28 residual. ROI and
residual are sampled without replacement by an explicit conditional
Plackett-Luce factorization. The detector-derived risk is exactly Focal
classification plus DIoU regression; a scalar action-independent EMA baseline
reduces variance. Route sampling uses a private RNG keyed by study seed,
successful update, rank, and role.

All detector-visible geometry/coordinate side channels are disabled. The
pretrained VideoMAE absolute position remains on. Geometry changes membership
only, and a temporal trajectory-shift control tests content alignment.

## Distinctive design

- fixed role quotas make the causal component comparisons identifiable;
- context supplies a deterministic coverage floor without a coverage loss;
- ROI supplies contiguous geometry-conditioned evidence;
- residual supplies non-ROI complementary evidence;
- complete conditional exclusion gives a valid joint hard-policy likelihood;
- full temporal support, exact-K, native patch semantics, and one heavy forward
  are preserved;
- diagnostic telemetry is per role, while cost replay excludes telemetry from
  its timed forward.

## Expected target

The exploratory target is not a paper claim. It is to determine whether
Hybrid-PL has a strict high-IoU increment over context+residual and context+ROI,
beats Fixed/Random, falls when geometry is temporally misaligned, and reduces
model+postprocess p50 relative to Dense. A complete pass may authorize only a
separately frozen dynamic-allocation design and disjoint-seed study; it cannot
promote fixed `8/28/28` as the final architecture.

## Evidence boundary

The old Hybrid number is descriptive motivation only. Exact runtime `0f64218d`
passed remote Linux/CUDA tests and the no-performance mechanical P0, including
exact role counts, private role RNG, finite ROI/residual branch gradients, and a
world-two FP32-DDP KAT. All nine single-seed stage Jobs completed `0:0`, but
finalizer `1213703` failed closed on an order-sensitive deployment-receipt check:
canonical JSON sorted the `jobs.stages` mapping while the validator compared
mapping insertion order with the frozen arm order. The canonical receipt hash,
stage-key set, Job bindings, and every other deployment predicate passed. The
sealed result has empty contrasts, so the route is not empirically supported,
paper-ready, Online TAD, Geometry Zoom, or a complete system-efficiency result.
A separately versioned immutable-input recovery finalizer is required before
reading performance.

## Pro review

Full absorption and qualified agreement:
`docs/methods/reviews/2026-08-02-hybrid-causal-pro-review-absorption.md`.

## Connections

Relations are maintained only in `research-wiki/graph/edges.jsonl`.
