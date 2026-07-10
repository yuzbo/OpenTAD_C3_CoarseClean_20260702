---
type: paper
node_id: paper:kim2024_te_tad
title: "TE-TAD: Towards Full End-to-End Temporal Action Detection via Time-Aligned Coordinate Expression"
authors: ["Ho-Joong Kim", "Jung-Ho Hong", "Heejo Kong", "Seong-Whan Lee"]
year: 2024
venue: "CVPR"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["TAD", "actual-time", "query-based", "coordinate"]
added: 2026-07-11T00:00:00+08:00
---

# TE-TAD: Towards Full End-to-End Temporal Action Detection via Time-Aligned Coordinate Expression

## One-line thesis

使用实际时间线坐标和随视频长度调整的 query 数，缓解 query-based TAD 的长度与坐标表达问题。

## Problem / Gap

归一化坐标和固定 query 集在视频时长变化大时会引入偏置，并阻碍 full end-to-end TAD。

## Method

time-aligned coordinate expression + adaptive query selection。

## Key Results

数值以 CVPR 官方页面为准。

## Assumptions

主要关注 query coordinate 与视频时长，不等同于显式 irregular observation support measure。

## Limitations / Failure Modes

不能据此声称“使用 actual time”本身是 PhysTime 的新颖点。

## Reusable Ingredients

actual timeline coordinate、length-aware query baseline。

## Open Questions

TE-TAD 在固定相同不规则观测下是否处理 support gaps 和局部采样密度偏差？

## Claims

无。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

是 PhysTime 新颖性和 physical-grid baseline 必须正面处理的最近 TAD 坐标工作。
