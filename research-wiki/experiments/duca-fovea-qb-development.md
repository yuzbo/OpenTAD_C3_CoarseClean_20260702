---
type: experiment
node_id: exp:duca-fovea-qb-development
title: "DUCA FoveaSampler/Query-Bridge development gate + 7-arm matrix"
idea: idea:duca-foveasampler-query-bridge
verdict: running
confidence: high
commit: "4ae50671"
jobs: "gate 1244850 passed; matrix 1244851 (arms 0-4 running, arms 5-6 pending slot release)"
updated: 2026-08-19
---

# DUCA FoveaSampler/Query-Bridge development gate + 7-arm matrix

## Raw metrics / observations

尚无训练结果。旧矩阵 `1244133`（7529fba6 wv1）与本次新方法矩阵相互独立：
旧矩阵 `_0`..`_7` COMPLETED、`_8`..`_14` RUNNING。

## Interpretation

实现与 focused tests 只能支持 `tested`，不能支持 `empirically_supported`。
GPU one-step gate `1244850` 已通过；7-arm 单 seed 开发矩阵第一波 `1244851` 正在完整训练，剩余两臂等待 MaxSubmit 槽位。

## Limitations

- 没有同提交 exact-uniform/random/dense 基线；
- 没有 trained checkpoint 成本数据；
- 没有第二 detector。

## Provenance

`docs/superpowers/specs/2026-08-19-duca-foveasampler-query-bridge-design.md`；
Slurm sacct/squeue；本地 `tests/test_fovea_query_bridge.py`。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
