---
type: paper
node_id: paper:liu2024_adatad
title: "End-to-End Temporal Action Detection with 1B Parameters Across 1000 Frames"
authors: ["Shuming Liu", "Chen-Lin Zhang", "Chen Zhao", "Bernard Ghanem"]
year: 2024
venue: "CVPR"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["TAD", "AdaTAD", "VideoMAE", "end-to-end", "adapter"]
added: 2026-07-11T00:00:00+08:00
---

# End-to-End Temporal Action Detection with 1B Parameters Across 1000 Frames

## One-line thesis

通过 temporal-informative adapters 降低端到端长视频 TAD 的训练内存，使大型 VideoMAE backbone 与长输入可联合训练。

## Problem / Gap

端到端 TAD 受 backbone 参数量、输入长度和训练显存限制。

## Method

冻结/低学习率视频 trunk，在 backbone 内插入可训练 temporal adapters，并连接 TAD projection/head。

## Key Results

数值以 CVPR 原文和本仓库 official config 结果为准。

## Assumptions

标准配置仍以规则长帧窗口和既定 temporal geometry 为主要输入。

## Limitations / Failure Modes

adapter 解决训练规模，不自动解决不规则观测的 physical-time metric。

## Reusable Ingredients

Official VideoMAE-S checkpoint、16-frame chunks、adapter optimizer policy、raw-video AdaTAD pipeline。

## Open Questions

当只处理相同 K 个不规则 raw frames 时，哪种检测头最能保护高-IoU 边界？

## Claims

无。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

PhysTime-AdaTAD 必须最大限度复用该 official backbone/adapter，而不能用简化 detector 证明主方法。
