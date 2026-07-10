---
type: idea
node_id: idea:duca-must
title: "DUCA-MUST dynamic budget"
stage: archived
outcome: negative
thesis: "预测每个视频/窗口的边际效用并在约束下学习 K(x)。"
risks: "离散 bucket 跳变；expected/hard/unique/padded/backbone K 不统一；padded cap 不产生真实 variable compute。"
based_on: []
target_gaps: []
added: 2026-07-11T00:00:00+08:00
updated: 2026-07-11T00:00:00+08:00
---

# DUCA-MUST

## 目标与现实

目标是用 Lagrangian/dual 约束学习样本相关预算，而不是外部 sweep。现实实现中 detector 常消费固定 padded cap，controller 的 soft expected K 与实际 kernel 长度不一致，并出现 64/384 跳变。

## 论文条件

动态预算必须：

- 实际 backbone/adapter/head compute 随 K 变化；
- 报告 requested/effective/unique/padded/soft expected K；
- 与 matched-average-K fixed curve 比 Pareto；
- controller 不依赖 val/test refit。

## 当前裁决

负面/appendix 记录。PhysTime 当前主线禁止 dynamic K，以免再次混淆几何与采样收益。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
