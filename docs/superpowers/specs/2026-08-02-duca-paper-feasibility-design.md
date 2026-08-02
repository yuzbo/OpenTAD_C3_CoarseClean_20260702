# DUCA paper-feasibility execution design

## Decision

The next experiment is the paper-facing feasibility matrix, not another
Admission simulation and not H-RIME.  The immutable first-stage scope is
ActionFormer, a dense 768-frame input window, a K=384 acquisition operating
point, all 200 THUMOS14 training videos, all 211 official validation videos,
and seeds 5801, 8123, and 12011.

The current protected DUCA frontend contains a train-only, jointly optimized
ASFormer coarse scanner.  It is not initialized from a separate frozen
checkpoint.  Freezing it would change the method.  Its runtime is therefore
part of full-stack DUCA cost.

## Scientific question

At matched heavy-backbone input cost, can task-aware learned temporal
positions protect official temporal localization better than exact-uniform
positions?  If that fixed-budget premise succeeds, can a train-only calibrated
dynamic controller improve the accuracy/cost frontier without validation or
test labels at decision time?

## Stage A: immediately executable cells

Each row is trained from the same VideoMAE-S initialization for 60 epochs,
100 global optimizer updates per epoch, global batch size two on two GPUs, and
terminal epoch-59 EMA evaluation.  Training evaluation loaders are sealed.

1. `dense`: dense AdaTAD/ActionFormer at T=768.
2. `uniform_fixed_k384`: exact-uniform K384 positions.
3. `uniform_mixed_train_k384_eval`: the registered per-video mixed-K training
   histogram (K in 192/256/384/512, counts 8/12/16/24, mean 384), evaluated at
   exact-uniform K384.  This is a detector-robustness control, not a dynamic
   inference policy.
4. `duca_fixed_k384`: jointly trained ASFormer evidence plus learned K384
   positions, with no dynamic budget controller.

The twelve Stage-A cells are individually paper-eligible only after their
complete official 211-video evaluation receipts exist.  Partial, single-seed,
training-domain, or intermediate metrics are engineering status only.

## Stage B: conditional dynamic-budget cells

Stage B is released only after Stage A yields a complete mixed-K checkpoint
and full-200 train-only counterfactual measurements.  Cross-fitted per-K
utility/risk targets and the mean-K384 protocol must be generated from the
training subset only.  No placeholder targets are permitted.

Stage B adds:

1. `duca_dynamic_mean384`, three seeds;
2. `uniform_same_realized_k`, evaluation-only replay of each dynamic run's
   exact realized-K ledger.

The fixed learned-position comparison isolates position quality.  The
same-realized-K replay isolates content-conditioned budget allocation.

## Official evidence contract

- annotation identity and VideoMAE initialization are SHA-256 bound;
- training set is exactly the 200 `training` keys and is never block-listed;
- evaluation set is exactly the 211 `validation` keys and is never
  block-listed;
- every training cell is exact-commit, clean-worktree, two-process DDP;
- each completed run has exactly 6000 successful optimizer, scheduler, and EMA
  updates;
- only terminal epoch-59 `state_dict_ema` is evaluated;
- evaluation must traverse exactly the 211 validation video identities through
  the standard OpenTAD sliding-window merge, NMS, and mAP evaluator;
- no Admission Monte Carlo output is model-performance evidence.

## Stop/go logic

Stage A is the feasibility decision.  The learned fixed-K method must be
compared against both fixed-uniform and mixed-K-training controls across all
three seeds.  Dynamic-budget work starts only if the fixed learned-position
mechanism is scientifically defensible or its failure analysis identifies a
specific, testable selector defect.  TriDet, K192, H-RIME, and broader ablations
remain deferred until this decision is complete.
