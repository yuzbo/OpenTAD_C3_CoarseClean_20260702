---
type: idea
node_id: idea:c3-coarse-actionness
title: "C3 low-cost coarse actionness"
stage: archived
outcome: mixed
thesis: "用低成本动作/背景二分类器产生逐时刻 p_action，为间接时序选择提供 deploy-visible 信号。"
risks: "二分类监督过粗；动作内部高分不等于边界效用；probe 计算可能转移成本。"
based_on: []
target_gaps: []
added: 2026-07-11T00:00:00+08:00
updated: 2026-07-11T00:00:00+08:00
---

# C3 Low-Cost Coarse Actionness

## 初心

先用 MobileNet/官方 ASFormer 类粗分类模型在 THUMOS 训练 split 上学习动作/背景二分类，再把 `p_action`、transition 和 hidden features交给间接选择模块。粗分类模型不是最终 detector，也不应单独决定 top-k。

## 保留价值

- 建立了低成本 deploy-visible signal、binary action target 和 no-leak 数据合同。
- 证明粗分类 supervision 与 selector supervision 应分离：前者识别动作性，后者保护边界/状态转换/任务效用。
- 可作为 PhysTime 未来 learned sampler 的输入候选，但不属于当前 K384 head-isolation 实验。

## 失败原因

- 仅看 `p_action` 会偏向动作内部。
- 只传曲线而不传 hidden feature，会退化成复杂 top-k。
- ASFormer“官方实现”不等于加载官方预训练权重，provenance 必须区分。

## 当前裁决

归档为 baseline/可复用输入资产，不再作为论文中心。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
