---
type: paper
node_id: paper:chen2025-progressive-block-drop
title: "Temporal Action Detection Model Compression by Progressive Block Drop"
authors: ["Xiaoyong Chen", "Yong Guo", "Jiaming Liang", "Sitong Zhuang", "Runhao Zeng", "Xiping Hu"]
year: 2025
venue: "arXiv"
external_ids: {arxiv: "2503.16916", doi: null, s2: null}
tags: ["tad", "depth-pruning", "model-compression"]
added: 2026-07-11
---

# Temporal Action Detection Model Compression by Progressive Block Drop

## One-line thesis

在 TAD 中逐步删除影响较小的 block，并通过跨深度对齐恢复性能。

## Problem / Gap

通道剪枝未必带来 GPU 友好的真实加速。

## Method

Block importance、progressive drop 与 cross-depth alignment。

## Key Results

在 THUMOS14 与 ActivityNet 报告约 25% 无损计算降低。

## Assumptions

静态删层足以覆盖主要深度冗余。

## Limitations / Failure Modes

不是输入相关动态调度，也不解决时间非均匀性。

## Reusable Ingredients

同任务强 depth-compression baseline。

## Open Questions

动态 time×depth 是否能显著超过静态 block drop？

## Claims

无本项目 proof-checker claim。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。

## Relevance to This Project

是 ChronoTransport 必须正面对比的同任务深度压缩近邻。
