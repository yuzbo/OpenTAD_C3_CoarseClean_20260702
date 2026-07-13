---
type: paper
node_id: paper:wang2022_rcl
title: "RCL: Recurrent Continuous Localization for Temporal Action Detection"
authors: ["Qiang Wang", "Yanhao Zhang", "Yun Zheng", "Pan Pan"]
year: 2022
venue: "CVPR"
external_ids:
  arxiv: "2203.07112"
  doi: null
  s2: null
tags: ["TAD", "continuous-anchor", "temporal-coordinate", "localization"]
added: 2026-07-13T13:30:00+08:00
---

# RCL: Recurrent Continuous Localization for Temporal Action Detection

## One-line thesis

以视频嵌入和连续时间坐标为条件学习可微连续锚表示，并通过尺度无关采样和递归细化完成 TAD。

## Problem / Gap

规则离散锚网格会产生尺度不均衡和短动作覆盖不足。

## Method

Continuous Anchoring Representation、scale-invariant sampling 与 recurrent refinement。

## Key Results

数值以 CVPR 2022 官方论文为准。

## Assumptions

研究重点是连续候选表示，不是原生不规则 raw-video observations 的显式非扩张 support provenance。

## Limitations / Failure Modes

它已占据“连续时间坐标/连续锚”这一宽泛新颖性空间；PhysTime 不能仅凭连续坐标或任意长度 proposal 声称新颖。

## Reusable Ingredients

continuous-coordinate baseline、短动作候选覆盖诊断、尺度无关采样对照。

## Open Questions

RCL 在固定不规则观测和连续空洞下是否仍把输入 token 视为规则时间网格？

## Claims

无。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

它要求项目把贡献限定为 observation-support 到 physical candidate pyramid 的接口，而不是连续锚本身。
