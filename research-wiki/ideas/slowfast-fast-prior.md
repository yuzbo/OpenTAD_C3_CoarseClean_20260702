---
type: idea
node_id: idea:slowfast-fast-prior
title: "SlowFast Fast-side frozen motion prior"
stage: archived
outcome: mixed
thesis: "用 SlowFast Fast pathway 的高时间分辨率运动特征辅助边界覆盖。"
risks: "Fast 侧不是独立预训练模型；计算仍重；相机运动干扰；Kinetics prior 与类别重叠。"
based_on: []
target_gaps: []
added: 2026-07-11T00:00:00+08:00
updated: 2026-07-11T00:00:00+08:00
---

# SlowFast Fast-Side Prior

## 正确认知

Fast pathway 与 Slow pathway 共同端到端预训练，不能把 Fast 侧描述为独立官方模型。Fast-only 更偏 motion/transition，可用于验证视频运动先验是否比图像 MobileNet 更适合边界，但易被 camera motion 误导。

## 当前裁决

代码/诊断保留，appendix-only。不得替换当前 PhysTime-AdaTAD 的相同无学习采样策略。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
