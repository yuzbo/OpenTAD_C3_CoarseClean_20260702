---
type: paper
node_id: paper:zeng2024_temporal_robustness
title: "Benchmarking the Robustness of Temporal Action Detection Models Against Temporal Corruptions"
authors: ["Runhao Zeng", "Xiaoyong Chen", "Jiaming Liang", "Huisi Wu", "Guangzhong Cao", "Yong Guo"]
year: 2024
venue: "CVPR"
external_ids:
  arxiv: "2403.20254"
  doi: null
  s2: null
tags: ["TAD", "robustness", "missing-frames", "FrameDrop", "TRC"]
added: 2026-07-11T00:00:00+08:00
---

# Benchmarking the Robustness of Temporal Action Detection Models Against Temporal Corruptions

## One-line thesis

系统评估 TAD 的时间破坏鲁棒性，发现主要退化来自定位而非分类，并提出 FrameDrop 与 temporal-robust consistency。

## Problem / Gap

TAD 对缺帧、模糊等时间破坏的脆弱性长期缺少统一 benchmark。

## Method

THUMOS14-C/ActivityNet-C corruption benchmark + FrameDrop augmentation + TRC。

## Key Results

数值以 CVPR/arXiv 原文为准。

## Assumptions

corruption protocol 与任意不规则 sampling/support measure 并不完全相同。

## Limitations / Failure Modes

若 PhysTime 只使用随机丢帧增强和一致性 loss，容易被视为该工作的自然延伸。

## Reusable Ingredients

temporal corruption families、localization-vs-classification decomposition、FrameDrop/TRC baselines。

## Open Questions

support-integrated operator 是否在 held-out gap pattern 上优于仅做 robust augmentation？

## Claims

无。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project

定义 PhysTime robustness 评测下限，并提醒 primary comparison 不应先加入额外 consistency supervision。
