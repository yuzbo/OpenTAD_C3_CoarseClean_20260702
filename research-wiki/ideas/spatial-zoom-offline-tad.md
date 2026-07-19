---
type: idea
node_id: idea:spatial-zoom-offline-tad
title: "Dense-time spatial zoom for offline TAD"
stage: designed
status: native_crop_contract_pending_pro_review
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

## Connections

Relations are maintained only in `research-wiki/graph/edges.jsonl`.
