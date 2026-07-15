---
type: idea
node_id: idea:spatial-zoom-offline-tad
title: "Dense-time spatial zoom for offline TAD"
stage: designed
status: s1_falsification_gate_running
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

1. S1: matched dense160/224/256 determines whether useful spatial-resolution
   headroom exists at high tIoU and for short actions.
2. S2: only after S1 GO, oracle/teacher ROI crops test whether the dense
   high-resolution gain can survive sparse spatial allocation at recoverable
   measured cost.
3. S3: only after S2 GO, learn a low-resolution scout, continuous ROI tubes,
   local high-resolution processing, and global/local feature fusion.

## Non-Claims

No learned crop model is currently implemented. S1 does not establish Zoom
accuracy, efficiency, detector generality, or publishability.

## Connections

Relations are maintained only in `research-wiki/graph/edges.jsonl`.
