---
type: idea
node_id: idea:trainfree-x3d
title: "Train-free frozen X3D actionness"
stage: archived
outcome: negative
thesis: "用 Kinetics 预训练 X3D 的冻结分类置信度/熵产生动作先验，不在 THUMOS 微调。"
risks: "dense inference 慢；类别重叠；max-prob 不等于二分类 actionness；预导出 JSONL 形成 offline pipeline。"
based_on: []
target_gaps: []
added: 2026-07-11T00:00:00+08:00
updated: 2026-07-11T00:00:00+08:00
---

# Train-Free X3D

## 讨论结论

“train-free”只表示不在 THUMOS 训练，不表示无预训练或低计算。X3D formal grid/export 需要长时间密集推理，作为 pre-backbone scout 会使整体 savings 失去意义。

## 当前裁决

停止所有密集 X3D 主实验。只可作为 frozen prior appendix，并必须报告 Kinetics checkpoint、class overlap、完整 latency 和是否使用预计算 JSONL。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
