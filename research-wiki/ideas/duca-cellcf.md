---
type: idea
node_id: idea:duca-cellcf
title: "DUCA-CellCF：coverage-preserving cell deformation"
stage: tested_diagnostic
outcome: killed_as_adaptive_allocation_main_retained_phase_control
tags: ["duca", "coverage", "counterfactual-utility", "offline-tad"]
added: 2026-07-13
---

# DUCA-CellCF

## 2026-07-19 final role correction

Exact-commit review plus independent verification permanently kills current
one-per-cell CellCF as a boundary-adaptive allocation main method. It cannot
move quota across cells, and for `T=768,K=384` can move each anchor by at most
one dense-grid index. Current acquisition content is also evaluated at fixed
uniform detector anchors with selected-axis GT remapping, so it is not an
actual-position physical-time allocation experiment.

CellCF remains useful only as a `tested_diagnostic` phase/content-correction
control. Its seed-0 Avg-mAP `64.0610` is below transition-beta0 `64.2755`.
This scoped KILL does not refute every DUCA hypothesis. A different
coverage-scaffold plus global-residual family is recorded separately as
`idea:duca-cara`, pending an exact feasible-set audit.

## 2026-07-16 Round-2 design refinement

The `7525efb` method/paper Pro review selected CellCF as the sole DUCA redesign:
exact-uniform nearest-anchor cells, one hard frame and one softmax distribution
per cell, explicit anchor tie-break at zero initialization, and detached
official-detector utility from at most four flips in different cells. Disjoint
cell flips make incidence rows disjoint (`AA^T=2I`), permitting a weighted
signed logistic objective without Gram whitening or forward-time
`autograd.grad`.

Project verdict is `PARTIAL_ACCEPT`. The coverage-preserving feasible family,
hard-action-aligned teacher and experiment DAG are accepted. Specific weights,
teacher cadence, schedules and numerical publication gates remain proposed.
The review's TAPOS substitution is incorrect, the existing direct-gradient C4
claim does not match detached utility, and one seed may kill this registered
configuration rather than prove every DUCA hypothesis false. Local-cell is a
uniform-residual selector and cannot claim global budget reallocation or GT
Oracle reachability.

Status was `designed` at review absorption. A local implementation now exists
on `codex/duca-cellcf-20260716`: exact cells, transition-first scorer,
separate acquisition/detector coordinates, local signed hard-flip utility,
three-arm evidence DAG and cost tooling. Local compile, shell syntax and
contract tests pass. A clean exact commit, Linux Torch tests, real THUMOS CUDA
gate, forced-overflow pilot, mAP and cost evidence are still absent.

## Thesis

以 exact-uniform anchors 划分保序 cells，每个 cell 恰选一帧；只学习 uniform 的局部
时间位移，并用训练期 hard alternative detector loss 蒸馏 cell 内 preference。该路线
删除 global top-k、`G=15` 和未经 hard-swap 对齐的 soft RGB detector bridge。

## Why it is the bounded appeal

- exact uniform 位于可行集内，coverage 不再依赖弱 loss 或 post-hoc repair。
- scorer 只读 action-state 的 deploy-visible temporal changes，保留间接边界定位初心。
- detector-derived utility 来自与推理可行动作一致的 hard alternatives。
- 相比 DUCA-FSU，它进一步把交换空间限制为 one-per-cell local deformation，并提出固定
  detector-grid anchor，以优先保护 uniform 的 coverage 和 detector geometry。

## Unresolved risks

- 实际采集位置 `s_j` 与 detector anchor `u_j` 不同，可能造成 observation-time mismatch。
- 独立 cell alternative loss 可能不能表达多个 cell 联合变化的非加性 detector utility。
- EMA teacher 非平稳，counterfactual 额外训练成本尚未测量。
- 一格一帧无法逼近可跨格聚集的 GT Oracle；合法 claim 只能是 uniform residual learning。
- 当前代码尚未实现 CellCF；review 中代码片段未经本地测试。

## Required gates

1. repo-wide exact-uniform and target-semantics audit;
2. pure delta vs compound proxy vs scorer diagnostic;
3. matched standalone/joint coarse quality;
4. cell coverage and hard one-swap utility alignment;
5. fixed-grid vs actual-selected-axis vs physical-time geometry;
6. same-commit exact-uniform vs CellCF fixed-384 pilot and multi-seed test;
7. trained-checkpoint full-stack accuracy-cost Pareto.

## Kill rule

若机制门失败，或 same-commit fixed-384 pilot 不能稳定优于 exact uniform，则停止 DUCA
主方法，不解锁 dynamic MUST、更多 detector、X3D/SlowFast 或新的 selector loss。

## Status

Current status: `tested_diagnostic` at exact model commit
`1642f265e48391418a7c8a4a087e33e2b7bf6899`. All three matched seed-0 arms
completed. Terminal-EMA Avg-mAP is 63.8594 for exact-uniform, 64.2755 for
transition-beta0 and 64.0610 for CellCF. CellCF is 0.2145 points below
transition-beta0, so the current detached local counterfactual utility has not
shown added value. The repaired cost schema passed real GPU gate `1170940`,
but the formal repeated cost pair has not been rerun. This route is neither
`empirically_supported` nor `paper_ready`.

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
