---
type: idea
node_id: idea:spatial-zoom-offline-tad
title: "Dense-time spatial zoom for offline TAD"
stage: tested
status: continuous_deformable_roi_goal_s2_redesign_required
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

The final method is not a fixed-resolution or fixed-window selector. A
lightweight global branch must regress a continuous source-coordinate box
`(cx, cy, w, h)` whose center, width, height, scale, and aspect ratio may vary
with the video and over temporal groups. The local branch may resample that
variable source ROI to a fixed tensor shape for batching and predictable
compute; that fixed tensor shape must never be described as a fixed source ROI.

## Decision Ladder

1. R0 control: preserve the completed dense160/224/256 training and its
   incomplete test/profile evidence as a full-frame resolution-sensitivity
   control. It is not crop evidence and is not a mandatory gate.
2. Native-Crop S1: close the source-frame crop, shared global/local backbone,
   detector-gradient, provenance, and no-leak data path. Formal CUDA Gate
   `1174671` passed; this is infrastructure evidence only.
3. D0 discrete diagnostic: fixed center and the 21 fixed `128x128` candidates
   may remain implementation controls. They are not the final method and
   cannot kill the continuous route.
4. Continuous-RoI S2: test crop sufficiency with variable
   `(cx,cy,w,h)` boxes and explicit scale/aspect/area constraints before a
   deployable policy is claimed.
5. Learned Spatial Zoom: after continuous S2 headroom, train a low-resolution
   scout and differentiable continuous ROI policy jointly with the TAD
   detector, followed by source-coordinate runtime crop and global/local
   fusion.

## Non-Claims

A development-only fixed-center Native-Crop vertical slice and its formal CUDA
Gate are complete. They do not implement a continuous ROI regressor. No
continuous crop-sufficiency, learned crop policy, Zoom accuracy, efficiency,
detector generality, or publishability claim exists.

## Pro Review Absorption

The 2026-07-20 Pro review returned `PROCEED_NATIVE_CROP_S1`. This authorizes
only a development-only, no-training vertical slice; it is not empirical GO.
The accepted next structure is a source-coordinate global/local crop path that
keeps the full temporal axis and the existing `[B,384,768]` detector contract.

Any fixed candidate library is only a discrete diagnostic, not a
library-conditional upper bound and not the final research object. Its failure
cannot kill continuous learned crop. The user's final target follows the
Uni-AdaFocus deformable-patch principle: regress location and variable width
and height, use a differentiable sampling path for end-to-end learning, and
prevent degenerate zero-area boxes with explicit constraints. Crop-output
tensor size, temporal grouping, size/aspect bounds, smoothing, supervision,
and numerical GO/KILL thresholds remain to be preregistered.

The completed fixed-center S1 slice is retained as infrastructure. The next
authorized design task is a continuous-RoI S2 protocol; the fixed 21-candidate
v1/v1.1 protocol is superseded as a decisive gate. No learned selector is
implemented until continuous sufficiency and its matched cost semantics are
frozen.

## Connections

Relations are maintained only in `research-wiki/graph/edges.jsonl`.
