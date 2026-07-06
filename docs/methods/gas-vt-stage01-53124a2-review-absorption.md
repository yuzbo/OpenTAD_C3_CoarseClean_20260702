---
updated: 2026-07-06
status: active
scope: Absorbed Pro review record for commit 53124a2 GAS-VT Stage0/1
out-of-scope: Final detector mAP claims, new experiment results, or paper-ready claims
---

# GAS-VT Stage0/1 Review Absorption

Raw record: `docs/methods/reviews/2026-07-06-gas-vt-stage01-53124a2-pro-review-raw.txt`

Reviewed commit: `53124a2`

Branch/context: `codex/gas-vt-mainline-20260706`

## Core Verdict

The GAS-VT Stage0/1 engineering chain is mostly closed as an offline sparse-ledger pipeline:

`GAS-VT policy checkpoint -> deployable sparse ledger -> AdaTAD full training/eval`

It must not be described as end-to-end training or detector-aware acquisition. The current allowed claim is only that strict GAS-VT sparse ledgers are generated, validated, and connected to AdaTAD full training/eval.

## Claims Not Allowed Yet

- Do not claim end-to-end temporal acquisition.
- Do not claim detector-aware acquisition.
- Do not claim GAS-VT improves AdaTAD mAP until fixed_384, fixed_768, and dynamic full AdaTAD results are compared against fair baselines.
- Do not describe the final selected frames as purely learned if hard gap repair or constrained decoding participates.

## Claims That May Be Used Carefully

- Offline GAS-VT sparse-ledger generation is implemented.
- Strict ledger validation includes no uniform fill/scaffold and p_action provenance checks.
- AdaTAD can consume the generated ledger through the offline `LoadFrames` path.
- Current Stage0/1 is a guarded experimental pipeline, not the final paper method.

## P0 Absorbed Issues

1. End-to-end claim is false for the current commit.
   - Selector is applied offline.
   - AdaTAD consumes a precomputed ledger.
   - Detector loss does not reach selector.

2. Detector-aware claim is false for the current commit.
   - GAS-VT training uses `p_action`, action targets, boundaries, and GT-derived budget supervision.
   - No dense AdaTAD teacher utility or detector-loss feedback is present.

3. Train GT supervision must be stated honestly.
   - GAS-VT is a train-GT-supervised selector pretraining route.
   - It is not pure p_action self-supervision.

4. PC-OT/MRAS bridge cannot be used as proof of hard sparse pre-backbone ST detector-aware selection.
   - It is a soft/continuous token transport route.
   - It can only be used as an ablation or separate route.

## P1 Absorbed Issues

1. `_gap_values` appears off by one relative to contiguous unselected holes.
   - Tail/head gaps should count unselected positions, not endpoint distance.
   - Validator and tests should be aligned with the intended gap definition.

2. `cvar_max_hole_loss` is a window-empty-mass surrogate, not the exact validator max-hole metric.
   - Either rename it honestly or implement a differentiable approximation closer to run-length holes.

3. GAS-VT deploy scoring is not truly autoregressive if `selected_so_far` is not updated during checkpoint application.
   - Either rename to static gap-aware scoring or implement iterative decoding that recomputes features after selections.

4. `constrained_topk` uses deterministic hard gap repair.
   - This is not uniform fill, but it is still hand-coded gap repair.
   - Metadata must expose `uses_hard_gap_repair=True`.
   - Required ablation: pure top-k vs gap-repaired top-k.

5. Paper-ready provenance should fail closed.
   - Avoid default inferred p_action provenance for deploy/paper experiments.
   - Require explicit source name, source hash, split, and no-leak flags.

6. GAS-specific validator wrapper is missing or inconsistent.
   - Add a wrapper name matching the GAS-VT route if the launcher/documentation refers to it.

7. Dynamic selected-count support needs loader smoke tests.
   - Test mixed selected counts in one batch.
   - Verify tensor shape, mask, GT remap, valid_len, and no fallback behavior.

## P2 Absorbed Issues

- Use GAS-specific naming in launchers and validator logs.
- Add selector internal validation/holdout metrics rather than only training split metrics.
- Add tiny real-checkpoint smoke tests for train -> save -> apply -> convert -> validate -> LoadFrames.
- Dynamic budget needs distribution diagnostics, not only nonconstant selected-count checks.

## Required Baselines Before Paper Claims

At minimum:

- dense AdaTAD same config
- uniform fixed_384 and fixed_768
- random fixed_384 and fixed_768 with multiple seeds
- p_action top-k
- delta-p_action top-k
- GAS-VT pure top-k
- GAS-VT with gap repair
- dynamic budget with matched average K fixed baseline
- boundary oracle or teacher-utility upper bound

Primary detector metrics:

- Average mAP over configured tIoU thresholds
- mAP at high tIoU, especially 0.60 and 0.70
- compute/memory/latency

Ledger diagnostics:

- selected count distribution
- boundary support
- action coverage
- max gap and p95 gap
- max hole and p95 hole
- uniform similarity
- no uniform fill/scaffold
- no GT/teacher/cache leakage in deployed val/test ledgers

## Stage2 Direction Absorbed

The review strongly supports moving from p_action-only/GAS-VT supervision to dense AdaTAD teacher utility:

- export train-only dense teacher utility
- lock teacher checkpoint/config SHA
- generate utility targets from detector responsibility, cls/reg loss, saliency, and counterfactual utility
- train selector only on train split utility
- strip all teacher payload from val/test/deploy ledgers
- compare detector-aware selector against p_action-only GAS-VT under matched budgets

## Stage3 Direction Absorbed

True end-to-end requires selector inside detector forward:

- hard forward sparse gather
- straight-through backward path
- original dense time coordinates preserved
- irregular point generation or equivalent true-time remap
- detector loss reaches selector
- forward-test deterministic hard selection
- collapse controls and full detector mAP proof

Toy gradient proof is not sufficient for paper claims.

## Immediate Execution Order

1. Finish current `gas_vt_fixed_384` full AdaTAD run and extract final mAP.
2. Run uniform_384 and p_action_topk_384 with the same AdaTAD config as urgent baselines.
3. If fixed_384 does not beat or match uniform_384 credibly, debug boundary support, short-action recall, max-hole behavior, and ledger-to-loader time remap before expanding the matrix.
4. If fixed_384 is competitive, run fixed_768, dynamic matched-K, pure-topK/no-gap-repair, and random multi-seed baselines.
5. In parallel, implement Stage2 dense AdaTAD teacher utility with train-only selector supervision.

## Paper-Claim Policy

Current status:

- Stage0/1 guarded offline pipeline: pass as engineering pipeline.
- End-to-end claim: fail.
- Detector-aware claim: fail.
- Main-method paper readiness: warn / not ready.

The paper story should only upgrade after detector mAP baselines and Stage2 teacher-utility evidence close.
