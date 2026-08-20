---
type: experiment
node_id: exp:duca-fovea-qb-development
title: "DUCA FoveaSampler/Query-Bridge development gate + 7-arm matrix"
idea: idea:duca-foveasampler-query-bridge
verdict: partial
confidence: high
commit: "4ae50671"
jobs: "gate 1244850 passed; matrix 1244851 (arms 0-4 running, arms 5-6 pending slot release)"
updated: 2026-08-19
---

# DUCA FoveaSampler/Query-Bridge development gate + 7-arm matrix

## Raw metrics / observations

THUMOS14 val（seed 3407，commit `4ae50671`，epoch_59 + EMA，官方 evaluator）：
- baseline_fused: Avg-mAP 42.94（0.3 61.47 / 0.4 54.74 / 0.5 45.21 / 0.6 32.82 / 0.7 20.45）
- query_only: Avg-mAP 45.26（0.3 61.74 / 0.4 55.84 / 0.5 47.65 / 0.6 36.91 / 0.7 24.15）
- query_gt_mask: Avg-mAP 49.16（0.3 66.30 / 0.4 60.57 / 0.5 52.45 / 0.6 40.52 / 0.7 25.95）
- query_cycle: Avg-mAP 54.67（0.3 71.62 / 0.4 66.23 / 0.5 57.52 / 0.6 46.33 / 0.7 31.63）
- query_fovea: Avg-mAP 43.77（0.3 63.11 / 0.4 56.22 / 0.5 45.63 / 0.6 33.76 / 0.7 20.11）

所有 cell exit 0:0，checkpoint epoch_59 已保存。旧矩阵 `1244133` 15/15 完成。旧矩阵 `1244133`（7529fba6 wv1）与本次新方法矩阵相互独立：
旧矩阵 `_0`..`_7` COMPLETED、`_8`..`_14` RUNNING。

## Interpretation

实现与 focused tests 只能支持 `tested`。5/7 arm 单 seed 原始结果只能支持 development 观察；缺少同提交 matched 基线和剩余两臂，不能支持 `empirically_supported`。query_cycle 在单 seed 上明显高于其他臂，但仍是单点观察。

## Limitations

- 没有同提交 exact-uniform/random/dense 基线；
- 没有 trained checkpoint 成本数据；
- 没有第二 detector。

## Provenance

`docs/superpowers/specs/2026-08-19-duca-foveasampler-query-bridge-design.md`；
Slurm sacct/squeue；本地 `tests/test_fovea_query_bridge.py`。

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
