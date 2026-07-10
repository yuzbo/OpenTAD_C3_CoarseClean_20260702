---
type: idea
node_id: idea:duca
title: "DUCA detector-utility-calibrated acquisition"
stage: archived
outcome: mixed
thesis: "在 heavy backbone 前用低成本信号选择任务有用观测，并让下游 TAD utility 监督选择。"
risks: "utility 常退化为 GT boundary proxy；score+top-k+repair 与既有 adaptive sampling 高度相似；全栈成本不明。"
based_on: []
target_gaps: []
added: 2026-07-11T00:00:00+08:00
updated: 2026-07-11T00:00:00+08:00
---

# DUCA

## 最强版本的目标

cache-free、full-window、in-forward、jointly trained pre-backbone selector：coarse hidden features -> transition/start/end/utility-first selector -> exact hard budget -> official AdaTAD/ActionFormerHead，detector loss 反传到 selector/probe。

## 完成的资产

- strict selected positions/no-leak/original-time contracts；
- official head one-step gradient 与 optimizer coverage；
- actionness auxiliary、hidden feature fusion、endpoint proxy；
- exact-K/max-gap structured policy、uint8 bridge、DDP static graph 修复；
- fixed/dynamic accounting 拆分和 claim locks。

## 未跨过的研究门槛

- GT endpoint proxy 不是 detector counterfactual utility；
- hard/soft surrogate 的 hard utility 方向未由 full result 证明；
- scout/decode/H2D 成本可能抵消 savings；
- 多个 proxy/loss 降低单一机制可辨识性；
- 与 adaptive frame sampling/top-k/scaffold 碰撞强。

## 当前裁决

代码和合同保留，论文中心已 pivot。不得把旧 DUCA run 作为 PhysTime 主线证据。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
