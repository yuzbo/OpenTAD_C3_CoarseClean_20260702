---
type: paper
node_id: paper:shukla2021_mtan
title: "Multi-Time Attention Networks for Irregularly Sampled Time Series"
authors: ["Satya Narayan Shukla", "Benjamin M. Marlin"]
year: 2021
venue: "arXiv"
external_ids:
  arxiv: "2101.10318"
  doi: null
  s2: null
tags: ["irregular-time", "continuous-time", "attention", "time-series"]
added: 2026-07-11T00:00:00+08:00
---

# Multi-Time Attention Networks for Irregularly Sampled Time Series

## One-line thesis

学习连续时间值嵌入并通过 attention 将可变数量的不规则多变量观测投影为固定表示。

## Problem / Gap

标准序列模型默认规则间隔，难以处理 EHR 等不规则稀疏时间序列。

## Method

continuous-time embeddings + multi-time attention interpolation/aggregation。

## Key Results

数值以 arXiv 原文为准。

## Assumptions

主要任务是 time-series interpolation/classification，不是长视频 TAD 的 endpoint localization 与 high tIoU。

## Limitations / Failure Modes

PhysTime 若只做 timestamp embedding + attention projection，会与 mTAN 非常接近。

## Reusable Ingredients

mTAN-like projection 必须成为直接 baseline。

## Open Questions

support mass、physical query pyramid 和 endpoint head 是否提供超出 mTAN projection 的必要机制？

## Claims

无。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

构成 irregular continuous-time novelty 的最近通用基线。
