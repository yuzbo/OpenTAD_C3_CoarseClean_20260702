---
type: idea
node_id: idea:duca-sqb-block-dk-v1
title: "DUCA-SQB-Block-DK-v1 semantic query-bridge block acquisition"
stage: discussed
outcome: pending
tags: ["ducca", "semantic-query-bridge", "physical-block", "fixed-k-attribution", "dynamic-k"]
added: 2026-08-20
---

# DUCA-SQB-Block-DK-v1

## One-line thesis

Query-Bridge 只输出动作性/起点/终点三类语义 logit；确定性 cell 规则把这些语义转成
连续 16-frame 物理块的非均匀采样；先做 fixed-K 严格归因，再开放真正的 dynamic-K。

## 为什么提出

独立 Pro 审查指出当前 UVT/Fovea 混合了分数、预算、监督、直接索引学习、时间重排和
课程，无法归因；采样单位必须是物理连续 block，不能把孤立帧按选择秩拼成伪连续 clip。

## 关键约束

- 候选单位：48 个原始时间连续的 16-frame RGB block（768 窗口）；
- 禁止 selected index / contribution top-k / K / Gumbel / DPP / MMR / detector utility 从 Query 输出；
- 主损失：balanced action/start/end BCE + query residual 正则，删除当前 Fovea mask/coarse/cycle/diversity/budget loss；
- 前端 P0 与检测器 6000-update 课程分离，uniform warmup 在同一预算内；
- 动态 K 与位置价值解耦；dynamic-K 用 training-CAL 冻结分位阈值，且必须有 DK-Upos / DK-shuffle 对照；
- 旧 direct selector 只保留一个 ablation，不进入主方法。

## 当前状态

`discussed`，尚未实现、测试或部署。UVT/Fovea 的实际负/弱结果与组件复用关系见各自
experiment/idea 节点。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
