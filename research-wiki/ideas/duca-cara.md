---
type: idea
node_id: idea:duca-cara
title: "DUCA-CARA: coverage-anchored residual allocation"
stage: designed
outcome: hold_pending_deployable_map
tags: ["duca", "offline-tad", "boundary-allocation", "physical-time", "coverage"]
added: 2026-07-19
---

# DUCA-CARA

## Thesis

在冻结的真实物理坐标上，以 exact-K 和最大物理间隔共同定义全局可行集，再由粗动作
状态变化、不确定性和训练期检测效用学习其中的选择。它与一格一帧 CellCF 的本质
区别是：背景区域可以释放预算，高状态变化区域可以获得多个观测。coverage scaffold
只允许作为已选集合的事后规范分解，不再预先固定为 mandatory slots。

## Why this route exists

`T=768,K=384` 的 CellCF 只能在每个均匀 cell 内移动至多一个 dense-grid index，
不能跨区域转移 quota。matched seed-0 中 CellCF 也没有优于 transition-beta0。
因此需要先改变可行集合，而不是继续叠加 selector loss。

## Candidate contract

- offline full-window TAD, fixed exact-K first;
- branch new work from evidence commit
  `4ce69c852bdbd902046b47bc6019ae11e850dbe4`, which contains immutable
  model commit `1642f265e48391418a7c8a4a087e33e2b7bf6899` as an ancestor;
- binary action/background supervision remains the coarse task;
- selector is transition/boundary first, not actionness top-k;
- global exact-K allocation under an explicitly frozen physical maximum
  interval;
- any scaffold/residual view is a deterministic post-hoc decomposition of a
  selected set, not a fixed feasible-set restriction;
- exact uniform must be an explicit feasible member of the new family, while
  the family must also permit background quota release and multiple residual
  observations near a boundary;
- actual acquisition positions are consumed by the existing physical-grid
  ActionFormer path;
- no selected-axis GT remap;
- detector-derived hard counterfactual utility is a candidate supervision,
  not automatically direct detector backpropagation;
- complete frontend plus detector cost accounting.

The old illustrative `T=768,K=384,G=3,192+192` split is no longer an
authorized model design. It is valid only for a dense-hole-3 convention. If
the contract instead means at most 15 original decoded frames, the stride-4
grid has an effective cap of 12 frames: an arbitrary fixed scaffold needs at
least 255 points, and an exact-uniform-subset scaffold needs 382. The physical
unit and cap remain unfrozen project choices.

## Required first gate

Before training, compare exact-uniform, one-per-cell, fixed
coverage-residual, global exact-K/physical-gap and privileged unrestricted
families with exact geometric constraints. Family D, global exact-K under a
physical cap, is the primary ceiling; fixed scaffold family C is diagnostic.
Report boundary distance/density, any/both endpoint coverage, background
release, short-action coverage, physical max gap and a frozen-detector
physical-grid diagnostic.

The coordinate unit must be frozen before solving. A dense-grid gap, decoded
frame-index gap and seconds gap are not interchangeable; report all three
through the actual decoder timestamp/index mapping. Family, cap and objective
design must use a frozen training-side partition. Current `val` and `test`
both use THUMOS validation, so different overlap settings do not create an
independent test population.

## Risks

- a strict original-frame cap can make a fixed scaffold consume nearly all
  quota;
- dense-index gap and original-frame gap can be confused;
- detached coarse evidence may protect calibration but weaken collaboration;
- allowing selector gradients into coarse features may corrupt binary
  actionness and recreate a direct boundary predictor;
- hard top-k requires auxiliary or estimator-based learning;
- physical-grid ActionFormer is a project extension, so detector generality
  remains unproven;
- the low-cost frontend can erase heavy-backbone savings.

## Kill rule

Do not train the full matrix if privileged global-D has no useful boundary
headroom, deploy-visible low-cost evidence cannot recover that headroom, the
family violates the frozen physical contract, or there is no plausible
full-stack break-even. If a matched fixed-K implementation cannot robustly
beat uniform, keep DUCA as a negative/diagnostic result and leave dynamic MUST
frozen.

## Status

`designed / hold_and_revise_family`. The global feasible-set diagnostic is
tested, but no revised global-D CARA training model, gate,
pilot, mAP or cost result exists.

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
