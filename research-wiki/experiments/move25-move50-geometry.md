---
type: experiment
node_id: exp:move25-move50-geometry
title: "Move25/Move50 dilation geometry diagnostics"
idea: idea:lattice-center-radius
status: historical
verdict: partial
confidence: medium
metrics: "Geometry and historical detector outputs are diagnostic only."
provenance: "docs/methods/2026-07-08-46cacc1-visualization-hold-review-absorption.md"
added: 2026-07-11T00:00:00+08:00
---

# Move25 / Move50 Geometry

## Question

局部移动与膨胀能否提高边界附近覆盖，同时保持固定预算？

## Verdict

选点存在聚集性，但聚集中心有偏差。现有统计只能说明相关，不能证明 geometry 导致 proposal/high-IoU 改善。

## Required interpretation

必须同时检查 endpoint nearest distance、bracket coverage、action-local holes、probe latency、repair count 与 detector best proposal tIoU。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
