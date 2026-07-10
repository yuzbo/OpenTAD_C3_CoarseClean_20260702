---
type: idea
node_id: idea:actal
title: "ACTAL compute-to-resolve"
stage: archived
outcome: pending
thesis: "streaming 模型把计算作为动作，主动缩小 endpoint belief 与输出延迟风险。"
risks: "需要因果协议、delay/revision 定义和 POMDP/VoI；偏离当前离线 TAD。"
based_on: []
target_gaps: []
added: 2026-07-11T00:00:00+08:00
updated: 2026-07-11T00:00:00+08:00
---

# ACTAL

## 核心动作

cheap update、heavy refresh、bounded buffer revisit、wait/emit，以 endpoint posterior 风险、计算成本、delay 和 revision penalty 联合决策。

## 当前裁决

这是有潜力的新在线任务，但用户已明确当前项目是完全离线 TAD。保留 idea，不与 PhysTime 混合。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
