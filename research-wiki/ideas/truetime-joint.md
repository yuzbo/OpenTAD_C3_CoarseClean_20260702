---
type: idea
node_id: idea:truetime-joint
title: "TrueTime 与 detector-gradient joint selector"
stage: superseded_component
outcome: mixed
tags: ["truetime", "joint-training", "gradient"]
added: 2026-07-11
---

# TrueTime 与 detector-gradient joint selector

## One-line thesis

hard forward 选择 original-time positions，soft/ST backward 接收下游 ActionFormer loss，并映回真实时间。

## 为什么提出

解决多阶段不优雅和 selector 与 TAD 目标脱节的问题。

## 已有证据

TrueTime metadata、后映射、gradient proof 和 joint wrapper 逐步实现；后来并入 DUCA。

## 当前选择或否定理由

思想保留并由 DUCA 继承，但 toy/smoke gradient 不能独立作为最终方法证据。

## 风险与失败模式

selected-axis 把不等物理间隔当等间隔；nonzero gradient 不代表 hard utility 对齐。

## 下一次允许采取的动作

完成 one-swap finite-difference 与 same-selected-frames geometry audit。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
