---
type: idea
node_id: idea:lattice-center-radius
title: "Lattice replacement and center-radius selection"
stage: archived
outcome: mixed
thesis: "从覆盖骨架出发，用 p_action/transition/utility 种子进行局部替换或半径膨胀。"
risks: "uniform scaffold 或 repair 可能才是有效成分；膨胀可掩盖中心偏移；单位和预算容易失真。"
based_on: []
target_gaps: []
added: 2026-07-11T00:00:00+08:00
updated: 2026-07-11T00:00:00+08:00
---

# Lattice / Center-Radius

## 产生的有效诊断

move25、move50 和 dilation 说明选点有聚集性，但中心位置仍可能偏离 GT 边界。最大间隔骨架有助于 fail-closed 覆盖，却会降低 learned policy 的可归因性。

## 偏移原因候选

- actionness binary label 只区分动作/背景，不能给精细 endpoint；
- probe temporal stride、窗口平滑或 clip center 带来延迟；
- score supervision、radius dilation 和 hard decode 不同构；
- 对所有正分数膨胀会退化成广泛平滑；
- repair 在最后重写选点。

## 当前裁决

仅作 geometry/decoder diagnostic。任何使用 scaffold 的结果必须报告 uniform overlap、replacement/repair count 和 pre/post gap。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
