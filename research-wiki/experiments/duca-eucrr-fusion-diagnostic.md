---
type: experiment
node_id: exp:duca-eucrr-fusion-diagnostic
title: "DUCA exact-uniform coarse residual reuse diagnostic"
status: discussed_conditional_not_authorized
added: 2026-07-22
updated: 2026-07-22
---

# DUCA EU-CRR representation diagnostic

## Question

With hard positions frozen to exact-uniform K=384, does the selected frozen
coarse ASFormer hidden state add useful detector representation after VideoMAE
and before the unchanged ActionFormer projection?

This is not a selection experiment and is not the G23 boundary-burst method.

## Prerequisites

- Seal terminal V8 `63e25eb` evidence first.
- Complete train-split Oracle K/G reachability R0 first.
- Bind U0/U1 to one implementation commit, P0 checkpoint hash, data split,
  seed, 6000 successful updates and terminal-EMA evaluator.
- Keep positions, selector/DP, VideoMAE, projection, neck, head, losses and
  schedule identical.

## Arms

| arm | positions | coarse path | detector reuse |
|---|---|---|---|
| U0 | exact-uniform K384 | same frozen P0 probe executes | hidden discarded |
| U1 | exact-uniform K384 | same frozen P0 probe executes | detached selected hidden through zero-gated post-VideoMAE residual |

Only if this diagnostic and the learned selector independently justify a
second matrix may L0/L1 run. Required contrasts are `U1-U0`, `L1-L0`,
`L0-U0` and `L1-U1`.

## Required contracts

- Zero-gate value equivalence with U0.
- Real wrapper shape and temporal-alignment checks; VideoMAE output is
  `[B,C,K]` after tubelet/chunk/interpolation processing.
- Coarse parameters frozen/eval and `grad is None`.
- Fusion optimizer coverage, no-decay gate and EMA inclusion.
- Two-step full-model gradient gate.
- No GT/teacher/cache leakage at inference.
- Full cost includes decode, transform, host materialization, dense H2D,
  coarse, DP, gather, VideoMAE, fusion, detector and remap/NMS.

## Interpretation

- U1 GO supports a possible acquisition-and-fusion adapter, not a strict
  pre-backbone-only plugin and not learned boundary selection.
- U1 KILL permanently stops this residual-fusion hypothesis only.
- No result from this node can replace G23/R0--R5 terminal-mAP evidence.

