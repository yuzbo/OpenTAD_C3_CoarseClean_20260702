---
type: idea
node_id: idea:phystime-tad-2
title: "PhysTime-TAD 2.0"
stage: active
outcome: pending
thesis: "把 TAD 定义为物理时间支持测度上的积分算子，使检测在观测网格细化和缺失下比 selected-rank operator 更一致。"
risks: "可能被 timestamp/interpolation/mTAN-like baseline 匹配；feature track 不等于 raw-video；新颖性必须窄而可证。"
based_on: ["paper:shukla2021_mtan", "paper:kim2024_te_tad", "paper:zeng2024_temporal_robustness", "paper:sun2026_liquidtad"]
target_gaps: ["gap:G1", "gap:G2", "gap:G3", "gap:G5"]
added: 2026-07-11T00:00:00+08:00
updated: 2026-07-11T00:00:00+08:00
---

# PhysTime-TAD 2.0

## 单一核心机制

对 observation ownership interval `I_i` 与 physical query cell `R_q`，使用重叠长度 `m_qi = |R_q ∩ I_i|` 作为指数外的 evidence mass；每层直接从原始不规则观测投影，query grid 由 seconds spacing 构造，检测头在秒坐标 assignment、decode、NMS。

## 关键性质

- constant-kernel support split additivity；
- padding invariance；
- gap 不扩张；
- query count 与 K 无关；
- uncovered query 输出有限零；
- endpoint intensity 按 cell width 积分为事件概率。

## 已实现范围

物理几何、measure projection、PhysTimeHead、detector、feature transforms 与 focused tests 已存在。该代码证明软件合同，不证明论文效果。

## 长期角色

这是最终独立 detector 的方法本体。`PhysTime-AdaTAD 1.0` 是它进入 raw-video official AdaTAD 的第一阶段验证。

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
