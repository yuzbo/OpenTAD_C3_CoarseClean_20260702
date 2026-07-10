---
type: paper
node_id: paper:zhang2022_actionformer
title: "ActionFormer: Localizing Moments of Actions with Transformers"
authors: ["Chenlin Zhang", "Jianxin Wu", "Yin Li"]
year: 2022
venue: "ECCV"
external_ids:
  arxiv: "2202.07925"
  doi: null
  s2: null
tags: ["TAD", "anchor-free", "transformer", "selected-axis-baseline"]
added: 2026-07-11T00:00:00+08:00
---

# ActionFormer: Localizing Moments of Actions with Transformers

## One-line thesis

用多尺度局部 self-attention 与轻量 anchor-free head 在单阶段完成动作分类和边界回归。

## Problem / Gap

长视频动作区间需要同时建模局部上下文与多尺度持续时间。

## Method

规则 temporal feature sequence -> multiscale local transformer -> point-wise classification 与左右边界距离回归。

## Key Results

本项目不在 Wiki 复制数值；以原论文和 matched repository baseline 为准。

## Assumptions

输入 token 通常代表规则、等间隔时间网格。

## Limitations / Failure Modes

将不规则观测压到 selected rank 后，其 Conv/attention/pyramid 距离不再等于物理时间距离。

## Reusable Ingredients

Official ActionFormerHead、anchor-free assignment、NMS 与 TAD evaluation。

## Open Questions

physical-grid assignment 是否已经足以修复大部分不规则时间问题？

## Claims

无。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

PhysTime-AdaTAD 的 selected-axis 和 physical-grid 两个关键 baseline 均以 ActionFormer 为基础。
