---
type: experiment
node_id: exp:chronotransport-formal-stage-b
title: "ChronoTransport formal Stage-B gate"
idea: idea:chronotransport
verdict: negative
confidence: high
commit: "92029ea"
jobs: "formal seed-3407 fit/calibration/evaluation artifacts recorded in method log"
updated: 2026-07-11
---

# ChronoTransport Formal Stage-B Gate

## Raw metrics / observations

正式 fit/calibration/evaluation 闭环、EMA、恢复、split 隔离和统计 gate 可运行。P3 总 gate 为 FAIL：risk-regret 排序为负；cell-risk 求和与窗口 regret target 尺度严重错配；transport 相对 HOLD 的局部 detector-regret 改善不能抵消 feature-level 改善不稳定。

实验数字仅保留在 `docs/methods/2026-07-10-chronotransport-implementation-plan.md` 及对应正式 artifact，本页不重复建立第二数字来源。

## Interpretation

该结果支持工程可行性，但否定当前 risk-certified transport 规格直接进入 Stage C/P5 或论文主方法的资格。

## Limitations

单种子负 gate 不能证明所有 feature-transport 思路无效；它只否定当前风险聚合、目标尺度和训练规格。

## Provenance

Commit `92029ea`; `docs/methods/2026-07-10-chronotransport-implementation-plan.md`。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
