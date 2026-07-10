---
type: experiment
node_id: exp:phystime-adatad-k384
title: "PhysTime-AdaTAD matched raw-video K384 head comparison"
idea: idea:phystime-adatad-1
status: planned
verdict: partial
confidence: low
metrics: "No result yet."
provenance: "docs/superpowers/specs/2026-07-11-phystime-adatad-1-design.md"
added: 2026-07-11T00:00:00+08:00
---

# PhysTime-AdaTAD K384

## Primary experiment

同一 THUMOS raw-video sample、同一 deterministic GT-independent K384 indices、同一 official VideoMAE-S adapter：

1. selected-axis ActionFormerHead；
2. physical-grid ActionFormer；
3. PhysTime measure projection + PhysTimeHead。

## Gates before queue

- raw geometry/GT seconds tests；
- all-config parity + same-index checksum；
- adapter/projection/head optimizer coverage；
- one real THUMOS CUDA decode-forward-backward-inference；
- no feature archive/env；
- gate-dependent exactly three formal jobs。

## Current verdict

尚无结果。frontmatter 的 `partial` 仅表示实验定义已冻结，不表示 claim 有部分支持；完成后必须由 result-to-claim 更新。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
