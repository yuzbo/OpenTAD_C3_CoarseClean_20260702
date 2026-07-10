---
type: idea
node_id: idea:cvcr-tad
title: "CVCR-TAD counterfactual value-of-compute routing"
stage: archived
outcome: pending
thesis: "用 time x layer 计算块的 counterfactual detector regret 学习是否执行重计算。"
risks: "counterfactual teacher 成本高；与 MoD/conditional depth 碰撞；packed kernel、成本和跨 detector 泛化未证实。"
based_on: []
target_gaps: ["gap:G7"]
added: 2026-07-11T00:00:00+08:00
updated: 2026-07-11T00:00:00+08:00
---

# CVCR-TAD

## 来源

ResearchClaw 对 DUCA `70aa069` 的 24-idea 发散审查 Top-1。

## 核心

对 time x layer 计算块估计 value of compute，用 train-only counterfactual detector loss 生成 target，再按 measured cost 选择重算。

## 裁决

吸收记录拒绝在三项决定性审计前直接用 CVCR 替代 DUCA。随后真正实现的是相近但更具体的 ChronoTransport，而 CVCR 本身没有独立代码/实验。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
