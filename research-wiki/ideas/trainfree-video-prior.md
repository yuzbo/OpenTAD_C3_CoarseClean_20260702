---
type: idea
node_id: idea:trainfree-video-prior
title: "X3D / SlowFast frozen video prior"
stage: diagnostic
outcome: negative
tags: ["x3d", "slowfast", "frozen-prior"]
added: 2026-07-11
---

# X3D / SlowFast frozen video prior

## One-line thesis

用 Kinetics-pretrained frozen 视频模型提供无需 THUMOS 训练的动作/运动先验。

## 为什么提出

验证视频预训练 prior 是否比图像 MobileNet 更理解动作变化，并提供 train-free baseline。

## 已有证据

X3D grid/export、JSONL pipeline 与 SlowFast Fast-side boundary diagnostic 已实现或部署过。

## 当前选择或否定理由

不作为主方法 coarse probe。仅保留 appendix diagnostic/upper prior。

## 风险与失败模式

密集推理耗时巨大，吞掉 backbone 节省；类别重叠；JSONL 预提取违背主方法统一 forward 叙事。

## 下一次允许采取的动作

只在明确成本账本下做少量 frozen prior 对照，不再重复密集 grid。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
