# DUCA-RIME Phase-2 Mixed-K Uniform Baseline Design

Status: approved for implementation by the user's 2026-07-28 instruction to
execute the accepted four-stage decision without another planning pause.

## Purpose

Formal O1 asks whether a single detector has useful localization headroom across
multiple heavy-frame budgets. A checkpoint trained only at K=384 cannot answer
that question: evaluating it at K=192/256/512 measures cross-budget transfer,
not mixed-K detector headroom. The required baseline is therefore one detector
trained with a frozen, auditable exposure to every registered K while the frame
positions remain exact uniform.

This baseline is an evidence instrument, not the final RIME method. Its formal
name is `U-mixed-K`; its inference selector is label-free, probe-free, and
batch-composition-free.

## Considered designs

1. Reuse `DucaRimeFrameSelector` and execute the ASFormer probe even though
   positions are uniform. This minimizes code but contaminates the baseline's
   cost and leaves unused learned selection machinery in the checkpoint.
2. Add a probe-free `uniform_mixed_k` path to the existing exact-K RIME
   execution contract. This preserves one physical-time decoder and ledger
   implementation while making the decision path constant and auditable.
3. Drive K from a mutable global optimizer-step counter. This offers an easy
   global histogram but couples allocation to AMP replay and resume bookkeeping.

Design 2 is selected. K is a pure function of immutable sample metadata and a
checkpoint-persistent schedule, so AMP retries reproduce the same decision.

## Frozen exposure

- Candidate budgets: `(192, 256, 384, 512)`.
- Execution quantum: 16 frames.
- Training: one process, batch size 1, 100 detector-training videos, 60 epochs,
  exactly 6000 successful optimizer updates.
- Per-video K counts over 60 epochs: `(8, 12, 16, 24)`.
- Mean heavy-frame exposure: exactly
  `(8*192 + 12*256 + 16*384 + 24*512) / 60 = 384`.
- The 60-entry cycle is deterministically permuted from schedule seed 3407 and
  persisted in the checkpoint. Video/sample index rotates the cycle, preventing
  every video in an epoch from receiving the same K.
- Training and inference fail if requested K cannot be executed exactly; no
  reduction to a shorter effective K and no padding to Kmax are allowed.

The schedule uses only `duca_stateless_epoch`, `duca_stateless_sample_index`,
the frozen cycle, and the requested evaluation budget. It cannot inspect GT,
teacher outputs, raw predictions, cheap action evidence, or other samples in the
test batch.

## Model and data flow

For `uniform_mixed_k`, the selector skips the coarse actionness network. It
constructs a constant potential, selects physical exact-uniform positions at the
scheduled K, gathers exactly K RGB frames, and emits the existing RIME physical
metadata and inference ledger. The heavy VideoMAE backbone, projection, adapter,
detector head, and NMS are unchanged.

At evaluation, one checkpoint is reopened four times with an immutable
evaluation K. Each run must bind the same checkpoint SHA-256, split assignment,
position policy, and no-padding ledger. The O1 source manifest records
`detector_training_exposure=mixed_k_registered_panel`; the formal record builder
rejects the fixed-K diagnostic label.

## Failure handling and evidence levels

- Missing schedule metadata, a non-candidate K, histogram drift, mean-cost
  drift, effective-K shrinkage, mixed-K batch execution, or padding is fatal.
- A deterministic reexecution is reproducibility evidence only, not an
  independent training seed.
- Passing unit tests or the code gate means `implemented/tested`, not
  `empirically_supported`.
- O1 can become empirically supported only after the mixed-K checkpoint and all
  hash-bound budget cells complete on the frozen train-only development role.

## Verification

Focused tests must cover the exact per-video histogram, mean K=384, deterministic
rotation, AMP-replay identity, probe-free construction, exact-K/no-padding
ledger, fixed evaluation K, contaminated/missing metadata rejection, 6000-update
config contract, launcher fail-closed behavior, and fixed-K diagnostic rejection.

Self-review: the design contains no placeholders; the exposure histogram,
selection inputs, failure conditions, claim boundary, and evaluation identity
are explicit and mutually consistent.
