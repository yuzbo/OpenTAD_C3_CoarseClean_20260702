---
type: idea
node_id: idea:phystime-tal-1
title: "PhysTime-TAL 1.0"
stage: archived
outcome: negative
thesis: "用 continuous relative-time attention、physical query grid、endpoint hazard 和双视图 consistency 处理不规则观测。"
risks: "与 mTAN/TE-TAD/LiquidTAD 碰撞；normalized time、support width、固定 M、hazard 和 consistency 定义不严。"
based_on: ["paper:shukla2021_mtan", "paper:kim2024_te_tad", "paper:zeng2024_temporal_robustness", "paper:sun2026_liquidtad"]
target_gaps: ["gap:G1", "gap:G2", "gap:G3"]
added: 2026-07-11T00:00:00+08:00
updated: 2026-07-11T00:00:00+08:00
---

# PhysTime-TAL 1.0

## 初始设计

任意 irregular features、timestamps、support widths -> continuous-time attention -> fixed physical query pyramid -> classification/start/end hazard；同一视频生成两个不规则 view 做 resampling consistency。

## HOLD 原因

1. `[0,1]` normalized time 不能替代真实秒尺度。
2. 邻接 timestamp/Voronoi support 会跨真实 gap 虚构质量。
3. 固定 query 数 M 不保证跨视频物理 receptive field。
4. sigmoid/BCE endpoint branch 不自动等于 hazard/intensity。
5. 双视图一致性在不可观测区域不应强制相同。
6. feature drop 不等于 raw-video 真实缺帧。
7. 宽泛 continuous-time/resampling claim 与近邻高度碰撞。

## 当前裁决

被 `idea:phystime-tad-2` 取代。原规格保留为失败设计审计。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
