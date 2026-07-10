---
type: idea
node_id: idea:paction-selector
title: "PAction learned selector"
stage: archived
outcome: mixed
thesis: "直接学习 p_action、delta 与边界代理下的固定预算选点。"
risks: "仍是 actionness/GT surrogate；高 mAP 可能来自半密度输入、几何 repair 或数据先验。"
based_on: []
target_gaps: []
added: 2026-07-11T00:00:00+08:00
updated: 2026-07-11T00:00:00+08:00
---

# PAction Learned Selector

## 观察

PAction learned fixed384 在历史 Stage1 中通常比 GAS-VT 简单而强，边界支持和 hole 统计也更好。它说明直接的 p_action 衍生表示可以是强 sparse-input baseline。

## 不可过度解释

- 它不是 raw `p_action` top-k 的同义词，但也不是 detector-derived utility。
- 它不证明优势来自边界聚集；需要 action-instance geometry 与 proposal tIoU 连接。
- 它不代表最终插件或独立 detector。

## 当前裁决

保留为强历史 baseline；任何未来 learned sampler 都必须至少解释为何优于它。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
