---
type: idea
node_id: idea:detector-aware-teacher
title: "Train-only detector-aware teacher utility"
stage: component_baseline
outcome: mixed
tags: ["teacher", "detector-utility", "stage2"]
added: 2026-07-11
---

# Train-only detector-aware teacher utility

## One-line thesis

利用 dense AdaTAD train-only teacher 产生 acquisition utility，训练 selector 后在测试时移除 teacher。

## 为什么提出

p_action 不知道哪些帧真正改善 detector 分类、回归和边界质量。

## 已有证据

责任/utility exporter、ledger 路线与 provenance gate 已实现过，但主要是 staged/offline 证据。

## 当前选择或否定理由

可作为 teacher upper bound 或辅助训练对照；不再把三阶段 teacher pipeline 当最终方法。

## 风险与失败模式

teacher leakage、坐标/fps/stride 映射错误、成本高；GT boundary proxy 被误称 detector utility。

## 下一次允许采取的动作

如使用，必须锁 checkpoint/config/hash、仅 train split，并与真实 detector-derived one-swap utility 区分。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
