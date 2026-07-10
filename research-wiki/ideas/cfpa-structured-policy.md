---
type: idea
node_id: idea:cfpa-structured-policy
title: "CFPA exact-K max-gap structured policy"
stage: archived
outcome: mixed
thesis: "用同一可行状态机产生 hard exact-K/max-gap 路径和其 differentiable relaxation。"
risks: "修复优化正确性但不解决 frame-selection 新颖性；CPU/内存成本和 detector utility 方向仍需审计。"
based_on: []
target_gaps: []
added: 2026-07-11T00:00:00+08:00
updated: 2026-07-11T00:00:00+08:00
---

# CFPA Structured Policy

## 价值

它修复 DUCA 的关键工程错误：hard inference 与 backward surrogate 使用同一 feasible-state definition，exact K 与 max-gap 同时成立，避免独立 soft-resample。

## 限制

这只是 selector 内部优化器，不是新的 TAD 问题。若 hard one-swap finite difference 与梯度方向不一致，仍不能声称 detector utility 学习成功。

## 当前裁决

保留为结构化离散决策和测试资产；不迁入 PhysTime-AdaTAD primary comparison。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
