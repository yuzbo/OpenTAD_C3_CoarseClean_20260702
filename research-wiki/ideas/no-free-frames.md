---
type: idea
node_id: idea:no-free-frames
title: "No-Free-Frames full-stack audit"
stage: active
outcome: pending
thesis: "高效视频方法必须把 decode、preprocess、scout、H2D、backbone、head、postprocess 和 amortized offline cost 全部计入。"
risks: "单独作为论文可能被视为 benchmark/engineering；必须伴随方法发现。"
based_on: []
target_gaps: ["gap:G7"]
added: 2026-07-11T00:00:00+08:00
updated: 2026-07-11T00:00:00+08:00
---

# No-Free-Frames

## 必要性

X3D export、offline ledger、padded dynamic K 已经实际证明“少算 backbone FLOPs”可以把成本转移到别处。该协议是所有效率主张的伴随线。

## PhysTime 用法

K384 与 dense768 比较时记录 raw decode frame count、CPU preprocess、H2D、VideoMAE adapter、projection/head、NMS、p50/p95 latency、peak memory 与 throughput。

## 当前裁决

作为评测协议 active，不作为当前独立方法。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
