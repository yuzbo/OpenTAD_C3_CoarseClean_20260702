---
type: paper
node_id: paper:sun2026_liquidtad
title: "LiquidTAD: Efficient Temporal Action Detection via Parallel Liquid-Inspired Temporal Relaxation"
authors: ["Zepeng Sun", "Naichuan Zheng", "Hailun Xia", "Junjie Wu", "Liwei Bao", "Xiaotai Zhang"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2604.18274"
  doi: "10.48550/arXiv.2604.18274"
  s2: null
tags: ["TAD", "continuous-dynamics", "efficient", "sampling-robustness"]
added: 2026-07-11T00:00:00+08:00
---

# LiquidTAD: Efficient Temporal Action Detection via Parallel Liquid-Inspired Temporal Relaxation

## One-line thesis

把 liquid dynamics 的指数松弛先验改写为可并行、线性复杂度的 temporal operator，用于参数高效 TAD。

## Problem / Gap

Transformer TAD 的参数和计算量较大，传统 liquid/ODE 动力学又有串行求解瓶颈。

## Method

parallel liquid-inspired relaxation + hierarchical decay-rate sharing + anchor-free TAD。

## Key Results

数值以 arXiv 2604.18274 最新版本为准。

## Assumptions

其 continuous dynamics prior 和 sampling variation robustness 不等同于 explicit irregular support-integrated geometry。

## Limitations / Failure Modes

PhysTime 不能再宽泛声称首个 continuous-time/physics-inspired TAD；必须比较结构和 robustness。

## Reusable Ingredients

efficient continuous temporal operator baseline、sampling variation comparison。

## Open Questions

LiquidTAD 在相同 raw irregular observations 与 explicit gaps 下，是否能匹配 PhysTime？

## Claims

无。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

是 2026 年最直接的新颖性碰撞之一，必须进入 Related Work 与 Phase 2 baseline。
