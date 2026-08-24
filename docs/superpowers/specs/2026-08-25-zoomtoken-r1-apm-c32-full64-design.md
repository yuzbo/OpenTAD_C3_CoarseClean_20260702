# ZoomToken R1-APM-C32/FULL64 design

## Objective

Isolate whether causal aligned patch memory can preserve or improve the official-comparable R1 representation without confounding the test with K32 deep execution. This is one performance-first admission experiment, not an efficiency claim.

## Frozen mechanism

- Keep the existing strict pre-backbone R1 8x8 support (K64) and current patch embedding for every selected tubelet patch.
- Reuse `build_apm32_temporal_plan` unchanged: within each eight-tubelet VideoMAE clip, compare only current `t` with detached `t-1`, use radius-two mutual-nearest matching at similarity threshold 0.80, and reset at each clip boundary.
- If fewer than 32 valid matches exist, use current-only K64. Otherwise, the 32 highest-similarity matched current slots use the existing carrier `stopgrad(E_prev) + alpha * (E_current - stopgrad(E_prev))`; all other slots use `E_current`.
- Add the current positional identity after carrier construction.
- Execute all K64 tokens through Q/K/V/output/MLP and the existing Adapter for all 12 VideoMAE-S blocks. The temporal match mask is diagnostic/carrier metadata only and must never become a block execution mask.
- Add no parameter, loss, router, optimizer group, hidden-state cache, cross-clip memory, future access, teacher, GT route, or raw-prediction input.

## Single experiment

- New arm: `R1-APM-C32-FULL64`, seed 42.
- Canonical THUMOS14 train to validation, official test closed.
- Same official initialization, augmentation, AdamW, learning rate, scheduler, AMP, EMA, ActionFormer detector/loss/evaluator/Soft-NMS, two GPUs, global/local batch 2/1, and 60 epochs as the read-only local official and R1 comparisons.
- Full-state recovery every five epochs, latest three plus final; epoch-59 final EMA is the only primary result.
- Do not retrain a current-only FULL64 control: deterministic parity with R1 is required instead.

## Admission and stop rule

The final EMA must satisfy Avg-mAP >= 68.73, mAP@0.6 >= 61.58, and mAP@0.7 >= 47.24, with at least one strict improvement. When a verifiable official scheduler receipt is available, accumulated accelerator-seconds must also not exceed it. Any miss stops this patch-memory route; no cost profiling, second seed, threshold search, or K32 rescue is opened by this run.

## Verification and handoff

The focused known-answer tests must prove the unchanged matcher semantics, detached previous/current-live gradients, FULL64 execution in every block, exact K64 support without padding, the temporal ledger, unchanged parameter inventory relative to R1, five-epoch recovery, and launcher/config binding. A clean immutable candidate then receives an independent read-only Critic review and a result-blind PRE_RUN before the one formal training job is submitted.
