---
type: idea
node_id: idea:structured-complementary-native-routing
title: "Structured Complementary Native Routing for offline TAD"
stage: experiment_running
status: pilot_running_after_p0_pass_no_empirical_support
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
world-two FP32-DDP KAT. All nine single-seed cells are submitted and the study is
therefore `experiment_running`; no performance result exists yet. The route is
not empirically supported, paper-ready, Online TAD, Geometry Zoom, or a complete
system-efficiency result.

## Pro review

Full absorption and qualified agreement:
`docs/methods/reviews/2026-08-02-hybrid-causal-pro-review-absorption.md`.

## Connections

Relations are maintained only in `research-wiki/graph/edges.jsonl`.
