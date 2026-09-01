---
type: paper
node_id: paper:dutson2023-eventful-transformers
title: "Eventful Transformers: Leveraging Temporal Redundancy in Vision Transformers"
authors: ["Matthew Dutson", "Yin Li", "Mohit Gupta"]
year: 2023
venue: "arXiv"
external_ids: {arxiv: "2308.13494", doi: null, s2: null}
tags: ["video", "token-update", "temporal-redundancy"]
added: 2026-07-11
---

# Eventful Transformers

## One-line thesis

识别随时间显著变化的 token，只重新处理这些 token，以利用视频帧间冗余。

## Problem / Gap

Video Transformer 对相邻帧重复执行大量相似计算。

## Method

在 Transformer block 内选择并更新变化 token，其他状态复用。

## Key Results

在视频目标检测和动作识别上报告约 2–4× 计算节省，精度仅小幅下降。

## Assumptions

相邻输入具有强时序冗余。

## Limitations / Failure Modes

未针对离线 TAD 的边界风险和高 tIoU 设计。

## Reusable Ingredients

变化 token 检测、state flush、runtime budget。

## Open Questions

变化幅度是否等价于定位计算价值？

## Claims

无本项目 proof-checker claim。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。

## Relevance to This Project

直接覆盖 ChronoTransport 的“非均匀重算变化状态”基础机制。
