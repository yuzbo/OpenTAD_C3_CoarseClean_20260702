---
type: paper
node_id: paper:shihab2025-adaptive-temporal-refinement
title: "Adaptive Temporal Refinement: Continuous Depth Allocation and Distance Regression for Efficient Action Localization"
authors: ["Ibne Farabi Shihab", "Sanjeda Akter", "Anuj Sharma"]
year: 2025
venue: "arXiv"
external_ids: {arxiv: "2511.03943", doi: null, s2: null}
tags: ["tad", "boundary", "continuous-depth", "adaptive-compute"]
added: 2026-07-11
---

# Adaptive Temporal Refinement

## One-line thesis

以 boundary distance regression 提升边界信号，并通过连续深度选择为困难边界分配计算。

## Problem / Gap

TAD 在边界难度高度不均匀时仍均匀分配计算。

## Method

Signed-distance boundary regression 与可微 continuous depth allocation。

## Key Results

论文报告在 THUMOS14 以更低 FLOPs 提升 mAP@0.7，并改善短动作。

## Assumptions

边界难度能够可靠驱动深度分配。

## Limitations / Failure Modes

需进一步核验代码、正式发表状态和实测成本。

## Reusable Ingredients

Boundary-specific compute allocation 强基线。

## Open Questions

其 continuous depth 是否产生真实 GPU execution savings？

## Claims

无本项目 proof-checker claim。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。

## Relevance to This Project

显著削弱“首次为 TAD 边界动态分配深度”的新颖性主张。
