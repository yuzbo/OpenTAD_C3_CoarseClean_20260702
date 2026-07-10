---
type: experiment
node_id: exp:duca-cost-structural-audit
title: "DUCA full-stack cost and official-structure audit"
idea: idea:duca
status: engineering_complete_science_pending
verdict: partial
confidence: high
metrics: "Profiler and source audit exist; no decisive trained-checkpoint cost/one-swap result recorded."
provenance: "a5e1774 on codex/gas-vt-stage23-detector-aware-20260706"
added: 2026-07-11T00:00:00+08:00
---

# DUCA Cost and Structural Audit

## Implemented

统一 full-stack profiler、成本守恒、bridge 成本纠错、official OpenTAD blob/config 对照、selected-axis coordinate disclosure 和配套 tests/validators。

## Still missing

正式 trained checkpoint 下 dense/uniform/periodic/DUCA 同硬件成本矩阵、one-swap finite-difference 与 surrogate alignment、same-selected-frames geometry 裁决。

## Branch warning

`a5e1774` 不在当前 PhysTime HEAD ancestry 中；需要用固定 commit 读取。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
