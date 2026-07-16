---
type: experiment
node_id: exp:lattice-move-diagnostics
title: "move25 / move50 / radius lattice geometry diagnostics"
idea: idea:lattice-boundary-replacement
verdict: partial
confidence: medium
commit: "ff85e56..c7b5fb6"
jobs: "historical lattice diagnostics"
updated: 2026-07-11
---

# move25 / move50 / radius lattice geometry diagnostics

## Raw metrics / observations

可视化与 ledger 分析表明选择具有聚集性，但聚集位置存在偏移；膨胀和半径可改善局部覆盖，却会占用预算并增加 uniform-like 风险。

## Interpretation

证明需要边界距离、cluster shift、repair ratio 等诊断；不证明 lattice/radius 是最终选帧方法。

## Limitations

部分 move 实验不是同提交；缺 detector instance matching 与 high-IoU 因果链。

## Provenance

docs/methods/2026-07-07-selector-geometry-analysis-suite.md; 2026-07-08-fbea37b-learned-context-radius-hold-review-absorption.md

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
