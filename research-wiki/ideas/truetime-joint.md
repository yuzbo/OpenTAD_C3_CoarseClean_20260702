---
type: idea
node_id: idea:truetime-joint
title: "TrueTime 与 detector-gradient joint selector"
stage: tested
outcome: mixed
tags: ["truetime", "joint-training", "gradient"]
added: 2026-07-11
---

# TrueTime 与 detector-gradient joint selector

## One-line thesis

hard forward 选择 original-time positions，soft/ST backward 接收下游 ActionFormer loss，并映回真实时间。

## 为什么提出

解决多阶段不优雅和 selector 与 TAD 目标脱节的问题。

## 已有证据

TrueTime metadata、后映射、gradient proof 和 joint wrapper 逐步实现；后来并入 DUCA。2026-08-22 的
严格 K384 配对中，RankPack/TrueTime 均完成 seed 3407、60 epoch、6000 次成功更新和 epoch-59 EMA
官方 validation，Avg-mAP 为 `61.57/62.19`；TrueTime 为 `+0.62` 点，tIoU 0.6 为 `+1.69` 点。
保存的 prediction 经冻结官方 evaluator 重算与记录值精确一致。

## 当前选择或否定理由

思想保留并由 DUCA 继承；当前单 seed 配对为物理时间解释提供部分机制证据，但没有 10,000 次逐视频
配对 bootstrap 或多 seed，且 post-run evidence 因 raw/normalized evaluation-config 哈希口径不一致
尚未封存。因此不能升级为稳定或论文级主张。

## 风险与失败模式

selected-axis 把不等物理间隔当等间隔；nonzero gradient 不代表 hard utility 对齐。

## 下一次允许采取的动作

不重训，先修复证据封存口径并对已有 prediction 执行 10,000 次逐视频配对 bootstrap；仅在该门支持后
考虑最小多 seed 确认。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
