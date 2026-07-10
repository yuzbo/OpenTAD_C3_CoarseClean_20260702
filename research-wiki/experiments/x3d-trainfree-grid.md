---
type: experiment
node_id: exp:x3d-trainfree-grid
title: "Frozen X3D train-free actionness grid/export"
idea: idea:trainfree-x3d
status: terminated
verdict: no
confidence: high
metrics: "Runtime failure for main-method suitability; no final downstream claim."
provenance: "docs/methods/2026-07-09-duca-jct-progressive-deployment.md"
added: 2026-07-11T00:00:00+08:00
---

# X3D Train-Free Grid

## Observation

多个 grid/export job 长时间运行并产生 partial JSONL，但无法及时形成正式 summary/downstream result。该现象本身证明 dense frozen X3D 不是低成本 pre-backbone main scout。

## Decision

终止所有密集 X3D 主实验。JSONL route 只允许 appendix replay，且必须诚实记为 offline precompute。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
