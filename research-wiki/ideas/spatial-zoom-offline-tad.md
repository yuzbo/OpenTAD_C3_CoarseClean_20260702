---
type: idea
node_id: idea:spatial-zoom-offline-tad
title: "Dense-time spatial zoom for offline TAD"
stage: designed
status: native_crop_vertical_slice_authorized
tags: ["offline-tad", "spatial-redundancy", "roi", "adatad"]
added: 2026-07-15
---

# Dense-Time Spatial Zoom For Offline TAD

## Hypothesis

Preserve every temporal observation and reduce only spatial redundancy. A
low-resolution global path may identify continuous action-relevant spatial
regions; expensive high-resolution computation can then be allocated to those
regions while an official-derived AdaTAD detector retains full temporal
coverage.

## Decision Ladder

1. R0 control: preserve the completed dense160/224/256 training and its
   incomplete test/profile evidence as a full-frame resolution-sensitivity
   control. It is not crop evidence and is not a mandatory gate.
2. Native-Crop S1: use source-frame coordinates and preserve native local
   pixel density while keeping the full temporal axis. First test an honest
   oracle/teacher-reference ROI-tube candidate library under matched
   pixel/FLOP/full-stack budgets and no official-test leakage.
3. Learned Spatial Zoom: only after Native-Crop S1 GO, learn a low-resolution
   scout, continuous ROI tubes, local high-resolution processing, and
   global/local feature fusion.

## Non-Claims

No native-crop or learned crop model is currently implemented. The existing
dense-resolution matrix does not establish crop sufficiency, Zoom accuracy,
efficiency, detector generality, or publishability.

## Pro Review Absorption

The 2026-07-20 Pro review returned `PROCEED_NATIVE_CROP_S1`. This authorizes
only a development-only, no-training vertical slice; it is not empirical GO.
The accepted next structure is a source-coordinate global/local crop path that
keeps the full temporal axis and the existing `[B,384,768]` detector contract.

The review's eight-candidate teacher library is treated only as a
library-conditional upper bound. Its failure cannot kill continuous learned
crop without a coverage certificate. Final masked pooling is also insufficient
to undo padded-token mixing inside ViT attention, so fixed crops should be
translated in-bounds whenever the source frame is large enough. Crop sizes,
knot frequency, motion constraints, teacher protocol, and numerical GO/KILL
thresholds remain provisional.

The sole authorized implementation task is a source-geometry census followed
by a no-teacher `global96 + fixed center/local128 + shared VideoMAE-S +
384-point fusion + [B,384,768]` vertical slice with strict no-resize,
backward, parity, no-leak, and cost-schema tests. R0 remains frozen.

## Connections

Relations are maintained only in `research-wiki/graph/edges.jsonl`.
