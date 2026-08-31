---
type: idea
node_id: idea:cfpa-streaming
title: "CFPA causal feasible-path acquisition"
stage: out_of_scope
outcome: unknown
tags: ["causal", "streaming", "exact-k"]
added: 2026-07-11
---

# CFPA causal feasible-path acquisition

## One-line thesis

用逐时刻不可撤销的 feasible-path policy 保证 causal exact-K/max-gap 选择。

## 为什么提出

外部审查误把 DUCA 视为在线因果任务后提出，用于严格 prefix invariance。

## 已有证据

存在独立设计/审计任务，但不是当前离线全窗口论文要求。

## 当前选择或否定理由

不接入当前 DUCA；仅作为未来 streaming/Online TAD 路线。

## 风险与失败模式

若强行接入会丢失全局边界证据并改变研究问题。

## 下一次允许采取的动作

只有项目明确转向 streaming 时再恢复。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
