---
type: experiment
node_id: exp:phystime-feature-track
title: "PhysTime-TAD I3D feature-token pilot track"
idea: idea:phystime-tad-2
status: cancelled
verdict: no
confidence: high
metrics: "Authoritative cancellation state in docs/evaluation/results.md."
provenance: "docs/evaluation/phystime-tad-pilot-manifest.json"
added: 2026-07-11T00:00:00+08:00
---

# PhysTime Feature Track

## Original plan

下载 ActionFormer I3D two-stream features，比较 support measure、point-only、no consistency、selected-axis、timestamp baseline 与 K192/384/768。

## Cancellation reason

用户要求直接在 AdaTAD raw-video 端到端路径验证稀疏检测头与真实计算节省。预提取 feature 轨道不能回答该问题，因此 data/real gate/pilots 均取消。

## Reuse

保留 geometry/projection/head 单元测试和 feature-input diagnostics。禁止把这些 jobs 称为 PhysTime-AdaTAD 论文证据。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
