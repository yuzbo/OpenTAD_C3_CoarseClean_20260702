---
type: paper
node_id: paper:soldan2025-residualvit
title: "ResidualViT for Efficient Temporally Dense Video Encoding"
authors: ["Mattia Soldan", "Fabian Caba Heilbron", "Bernard Ghanem", "Josef Sivic", "Bryan Russell"]
year: 2025
venue: "ICCV 2025"
external_ids: {arxiv: "2509.13255", doi: null, s2: null}
tags: ["dense-video", "residual", "token-reduction", "tal"]
added: 2026-07-11
---

# ResidualViT for Efficient Temporally Dense Video Encoding

## One-line thesis

通过可学习时序残差与 token reduction 低成本生成 temporally dense frame features。

## Problem / Gap

密集时序任务需要高帧率特征，但逐帧 foundation encoder 成本过高。

## Method

原始 encoder 与快速 residual encoder 交错执行，并通过 distillation 逼近 dense 特征。

## Key Results

跨四类任务和五个数据集报告最高 60% 计算降低与最高 2.5× 加速；包含 TAL 评估。

## Assumptions

相邻帧特征可由残差路径近似。

## Limitations / Failure Modes

不是 VideoMAE 内部 layer-group 调度，TAL 也不是其唯一目标。

## Reusable Ingredients

Dense-output residual encoding、teacher feature distillation。

## Open Questions

TAD-specific regret 是否能显著优于通用 feature approximation？

## Claims

无本项目 proof-checker claim。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。

## Relevance to This Project

是当前最直接的 temporally dense feature reuse 近邻。
