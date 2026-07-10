---
type: idea
node_id: idea:gas-vt
title: "GAS-VT gap-aware value transport"
stage: archived
outcome: negative
thesis: "用 gap、剩余预算、边界与动作内部代理构造 value-transport 稀疏观测。"
risks: "非真正 sequential；train/apply budget shift；hard repair 可生成隐式 uniform scaffold。"
based_on: []
target_gaps: []
added: 2026-07-11T00:00:00+08:00
updated: 2026-07-11T00:00:00+08:00
---

# GAS-VT

## 设计

使用 `remaining_budget`、`remaining_time`、`budget_pressure`、`gap_urgency`、boundary bracket、action interior、CVaR hole 等特征/损失，然后全局打分、top-k 与约束修复。

## 关键失败

1. 实际 decoder 不是 select-update-state 的顺序 value transport。
2. apply 时可能没有与训练相同的 target budget，导致特征语义漂移。
3. 过强 gap/coverage/action interior 目标会与高-IoU detector utility 冲突。
4. hard repair 可把输出推向 uniform，即使 metadata 声称无 uniform fill/scaffold。
5. fixed384 暴露半数网格，早期 mAP 不能归因给 GAS 机制。

## 当前裁决

失败/诊断 idea，永不从 query pack 删除。不得再通过新增权重继续挽救为主线。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
