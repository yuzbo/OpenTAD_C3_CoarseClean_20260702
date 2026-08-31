---
type: paper
node_id: paper:cui2026-scope
title: "Not All Frames Deserve Full Computation: Accelerating Autoregressive Video Generation via Selective Computation and Predictive Extrapolation"
authors: ["Hanshuai Cui", "Zhiqing Tang", "Zhi Yao", "Fanshuai Meng", "Weijia Jia", "Wei Zhao"]
year: 2026
venue: "arXiv"
external_ids: {arxiv: "2604.02979", doi: null, s2: null}
tags: ["video", "cache", "predict", "recompute", "stability"]
added: 2026-07-11
---

# SCOPE

## One-line thesis

在自回归视频扩散中用 `cache / predict / recompute` 三模调度，并通过误差传播分析控制预测稳定性。

## Problem / Gap

Binary cache-or-recompute 无法覆盖“直接复用太粗、完整重算又过贵”的中间状态。

## Method

Noise-level Taylor extrapolation、三模 scheduler、stability controls 与 active interval computation。

## Key Results

在 MAGI-1 与 SkyReels-V2 报告最高 4.73× 加速并保持近似质量。

## Assumptions

对象是生成模型的 denoising dynamics，不是判别式 TAD。

## Limitations / Failure Modes

其稳定性分析不能自动转移到 VideoMAE/TAD。

## Reusable Ingredients

三模复用/预测/重算与显式误差稳定性。

## Open Questions

TAD-specific structured regret 是否提供足够的新 delta？

## Claims

无本项目 proof-checker claim。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。

## Relevance to This Project

这是 ChronoTransport 三动作语义最危险的 2026 近邻。
