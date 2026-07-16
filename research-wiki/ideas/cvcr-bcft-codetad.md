---
type: idea
node_id: idea:cvcr-bcft-codetad
title: "CVCR / BCFT / CoDeTAD 替代路线"
stage: proposed
outcome: unknown
tags: ["counterfactual", "routing", "alternative"]
added: 2026-07-11
---

# CVCR / BCFT / CoDeTAD 替代路线

## One-line thesis

通过 counterfactual value、边界条件计算或 time-layer routing，把计算直接分配给有边际价值的位置/层。

## 为什么提出

若 pre-backbone frame dropping 被成本或几何否定，需要更接近真实计算价值的替代问题定义。

## 已有证据

ResearchClaw 发散审查提出；尚无完整实现与匹配结果。

## 当前选择或否定理由

作为 pivot 候选，不视为已证明比 DUCA 更优。

## 风险与失败模式

teacher/counterfactual 成本、packed kernel、novelty collision、跨 head 泛化。

## 下一次允许采取的动作

仅在 DUCA 决定性门槛失败后按失败类型选择最匹配路线。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
