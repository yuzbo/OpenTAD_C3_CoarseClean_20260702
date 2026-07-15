---
title: ChronoTransport CT-P3R-3S-r2 minimal protocol amendment proposal
date: 2026-07-15
status: proposed_unapproved
predecessor_spec: docs/superpowers/specs/2026-07-12-chronotransport-ct-p3r-3s-r2-design.md
predecessor_spec_commit: e4422f5
predecessor_spec_sha256: 87FA305CCAFC3A29176C3971F593489F86EDD23A4C02C1BFBDAE4144FCF34CF8
---

# ChronoTransport r2 minimal protocol amendment proposal

This file is a decision proposal only. It does not amend the approved specification, unlock
registration, authorize a job, or create an experimental result. The frozen predecessor remains
normative until the user explicitly approves exact replacement text and a spec-only review approves
the resulting bytes.

## Why an amendment is required

Four implementation-reachable questions are not uniquely executable from the current approved text:

1. the unsuffixed Gate-1 comparators `random_p2/p4/p8` use a seed-dependent digest, but no single
   comparator seed is fixed;
2. the predecessor requires physical GPU1 and writes `CUDA_VISIBLE_DEVICES=1`, while current governing
   repository rules require a Slurm-assigned GPU, forbid physical-index pinning/visibility overwrite,
   and require the process to use logical `cuda:0`;
3. Stage C explicitly requires overflow rollback of `AnchorFreeHead.loss_normalizer`, but does not
   state whether that train-mode buffer may advance on a successful matched update;
4. the inherited regret definition needs a current-model dense no-grad reference plus a
   counterfactual pass, whereas §13.4 only says that the three optimized losses come from the same
   counterfactual forward. The exact paired-forward and per-window batch-two contract should be stated
   rather than inferred by a runner.

## Recommended exact decisions

### A1 — Gate-1 random-control identity

Freeze `control_seed=3407` for all three unsuffixed comparators `random_p2`, `random_p4`, and
`random_p8`, for every Gate-1 evaluation window and layer group. The three training seeds do not create
three additional random comparator identities. Registration must bind the integer, generated action
hashes, and factory config; missing or different values remain `INVALID_IMPLEMENTATION`.

Reason: this is the smallest repair and preserves the already frozen six-comparator schema. Using a
different random seed per Stage-B seed would multiply or redefine the Gate-1 comparator set and is not
a minimal amendment.

### A2 — Slurm-assigned device semantics

Replace every physical-GPU1 / exact-`CUDA_VISIBLE_DEVICES=1` requirement with all of the following:

- every GPU action runs inside a single-GPU Slurm allocation or step;
- launchers do not set, replace, append to, or otherwise modify Slurm's `CUDA_VISIBLE_DEVICES`;
- the process requires exactly one CUDA-visible device and addresses it only as `cuda:0`;
- precheck records the unmodified `CUDA_VISIBLE_DEVICES`, `SLURM_JOB_ID`, `SLURM_STEP_ID`,
  `SLURM_JOB_GPUS`, `SLURM_STEP_GPUS`, `SLURM_GPUS_ON_NODE`, GPU model and GPU UUID;
- registration freezes the required GPU model plus driver/CUDA/PyTorch/cuDNN/precision contract.
  Each formal artifact binds its observed GPU UUID and Slurm allocation identity. A model or software
  mismatch is `INVALID_ENVIRONMENT`;
- all candidates in a Gate-1 cost profile use the same allocated device; all D/C/S arms in a matched
  Gate-4 timing block use the same allocated device. No result may compare costs collected on mixed
  GPU models.

Reason: this obeys the current repository authority and preserves matched hardware comparisons without
pretending that a Slurm logical ordinal is a physical GPU index. The rejected alternative is to weaken
the governing rule and pin physical GPU1 outside normal Slurm assignment.

### A3 — Stage-C frozen-head state semantics

Keep the detector/head parameters frozen but keep `AnchorFreeHead` in train mode for the detector task
loss. `rpn_head.loss_normalizer` is the only approved success-mutated detector buffer. It must:

- start bitwise equal in CT and matched-dense arms;
- be snapshotted and restored bitwise after every overflow attempt;
- advance exactly once after each successful arm update according to the unchanged OpenTAD formula;
- have identical per-successful-update values in CT and matched-dense arms because they share ordered
  GT/masks/materialized batches;
- be checkpointed and included in resume validation and the matched-arm trace hash.

All other frozen detector buffers must remain unchanged unless separately enumerated by a later
approved amendment. Any trace divergence is `INVALID_IMPLEMENTATION`.

Reason: §13.5 already names this buffer in overflow rollback, and this choice preserves OpenTAD's
training loss normalization. Putting the head in eval mode would silently replace the EMA normalizer
with per-batch positive-count normalization and therefore change the training objective.

### A4 — Stage-C paired forward and per-window regret

For each attempt and the same ordered materialized batch of two windows:

1. snapshot the pre-attempt RNG and mutable state;
2. run one forced-dense reference forward under `torch.no_grad()` and capture per-window detector task
   losses and dense features;
3. restore the paired-forward RNG state required for stochastic equivalence;
4. run exactly one differentiable counterfactual forward with the two registered candidate actions;
5. derive `LD` from the two counterfactual detector task losses using the unchanged batch reduction;
6. derive each risk target as
   `max(L_counterfactual_window - L_dense_window, 0)`, detach both target entries from A/T, and compute
   `LR` from the two predictor rows produced from the same counterfactual runtime signals/actions;
7. derive `LF` from counterfactual versus dense features with the gradient ownership in §13.4.

The detector must expose an exact two-element per-window task-loss vector from those same head logits
and targets; recomputing a second counterfactual head pass, reading a replay ledger, using GT in the
scheduler, or accepting caller-supplied regret targets is forbidden. Audit hooks count one dense
reference and one counterfactual model pass, and exactly one counterfactual risk-predictor pass.

Reason: this makes the inherited paired regret definition executable while preserving the current
model, augmentation and loss provenance. A precomputed or caller-supplied target would not track the
current Stage-C adapters and would reopen a leakage path.

## Approval and stop condition

Recommended approval sentence:

> I approve A1–A4 exactly as written as the minimal CT-P3R-3S-r2 protocol amendment. Create a new
> immutable spec-only commit, obtain an independent spec-only review, then continue implementation and
> registration. Do not reinterpret any other threshold, seed, split, budget, candidate, update count or
> stop condition.

Until that approval and review exist, random controls stay intentionally unconstructable, formal
Slurm launchers stay locked, Stage-C real-runner choices stay unregistered, and no I/R or formal Gate
may be minted.
