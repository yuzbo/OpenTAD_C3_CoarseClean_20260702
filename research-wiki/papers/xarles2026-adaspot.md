---
type: paper
node_id: paper:xarles2026-adaspot
title: "AdaSpot: Spend Resolution Where It Matters for Precise Event Spotting"
authors: ["Artur Xarles", "Sergio Escalera", "Thomas B. Moeslund", "Albert Clapes"]
year: 2026
venue: "CVPR"
external_ids: {arxiv: "2602.22073", doi: null, s2: null}
tags: ["event-spotting", "spatial-zoom", "roi", "efficient-video"]
added: 2026-07-13
---

# AdaSpot

## One-line thesis

以低分辨率全帧提取全局特征，并在每帧选择一个高分辨率 ROI；使用无需额外训练的
task-aware saliency 与时空平滑，避免弱空间监督下 learnable crop 的不稳定。

## Key Relevance

它比视频分类版 Uni-AdaFocus 更接近本项目：目标是精确事件时间定位，并直接说明空间
降采样会损害细粒度时间判别线索。其 global/local/fused 辅助监督和 ROI temporal
smoothing 是应优先复现的稳定 baseline。

## Limitations for This Project

Precise Event Spotting 与带起止区间的 TAD 仍不同；单 ROI 在人物/球主导数据上成立，
不代表 THUMOS 的全局动作、多人或并发事件也成立。它还使“低分辨率全局 + 高分辨率
局部”本身不再具有新颖性，主方法必须提供 TAD-specific delta。

## Source

- Paper: https://openaccess.thecvf.com/content/CVPR2026/html/Xarles_AdaSpot_Spend_Resolution_Where_It_Matters_for_Precise_Event_Spotting_CVPR_2026_paper.html
- arXiv: https://arxiv.org/abs/2602.22073
- Code: https://github.com/arturxe2/AdaSpot

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
