---
type: idea
node_id: idea:structured-complementary-native-routing
title: "Structured Complementary Native Routing for offline TAD"
stage: implemented
status: p0_pending_no_empirical_support
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
separately frozen, disjoint-seed confirmatory study.

## Evidence boundary

The old Hybrid number is descriptive motivation only. The present route is
`implemented`; local pure contract checks pass, but remote Torch/CUDA P0 and
all nine performance cells are pending. It is not empirically supported,
paper-ready, Online TAD, Geometry Zoom, or a complete system-efficiency result.

## Pro review

Full absorption and qualified agreement:
`docs/methods/reviews/2026-08-02-hybrid-causal-pro-review-absorption.md`.

## Connections

Relations are maintained only in `research-wiki/graph/edges.jsonl`.
