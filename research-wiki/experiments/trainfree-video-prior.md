---
type: experiment
node_id: exp:trainfree-video-prior
title: "X3D / SlowFast train-free diagnostic"
idea: idea:trainfree-video-prior
verdict: no
confidence: high
commit: "c95a406/628b9d1/f705dda"
jobs: "1151093, 1151305, 1151868 and related historical jobs"
updated: 2026-07-11
---

# X3D / SlowFast train-free diagnostic

## Raw metrics / observations

X3D dense grid/export 运行数小时且出现多套重复任务；未形成可接受的主方法成本闭环。SlowFast Fast 仅作为边界运动先验诊断。

## Interpretation

否定密集 X3D/SlowFast 作为低成本 pre-backbone 主模块；不否定其作为 frozen prior baseline 的研究价值。

## Limitations

未完成统一成本硬件与 class-overlap audit；部分作业取消或旧逻辑。

## Provenance

main thread record; duca-online-plugin-code-experiment-trainfree-x3d-severe-review-absorption.md

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
