---
type: idea
node_id: idea:lattice-boundary-replacement
title: "Lattice、move25/move50 与 adaptive radius"
stage: diagnostic
outcome: mixed
tags: ["lattice", "max-gap", "boundary"]
added: 2026-07-11
---

# Lattice、move25/move50 与 adaptive radius

## One-line thesis

从采样骨架出发，在局部用动作性或边界候选替换，并通过固定/学习半径保护上下文。

## 为什么提出

避免纯 top-k 聚集导致大时间空洞，并测试边界附近局部密集是否提升 TAD。

## 已有证据

move25/move50、膨胀和 learned radius 代码与几何分析存在；观察到选择聚集但中心仍偏移。

## 当前选择或否定理由

仅作几何诊断和工程对照，不作为论文最终 selector。

## 风险与失败模式

uniform scaffold 与膨胀消耗预算；repair 可掩盖 score 学习失败；clustered-but-shifted。

## 下一次允许采取的动作

保留 selected-to-boundary distance、repair ratio、gap 和偏移可视化，不再围绕半径搜索主方法。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
