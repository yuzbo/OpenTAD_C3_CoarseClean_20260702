# DUCA-RIME Phase-2 Mixed-K Uniform Baseline Design

Status: amended by the user-approved 2026-08-03 full-data short-window
corrigendum. The original exact-requested-K rule is retained below only where it
does not conflict with the corrigendum; the corrigendum is authoritative.

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

## Frozen requested exposure and realized execution

- Candidate budgets: `(192, 256, 384, 512)`.
- Execution quantum: 16 frames.
- Training: one process, batch size 1, all 200 detector-training videos, 60 epochs,
  exactly 6000 successful optimizer updates.
- Per-video requested-K counts over 60 epochs: `(8, 12, 16, 24)`.
- Nominal requested mean K is exactly
  `(8*192 + 12*256 + 16*384 + 24*512) / 60 = 384`.
- The 60-entry cycle is deterministically permuted from schedule seed 3407 and
  persisted in the checkpoint. Video/sample index rotates the cycle, preventing
  every video in an epoch from receiving the same K.
- For a natural window with valid length `L`, the label-free feasible execution
  is `K_eff = min(K_req, floor(L / 16) * 16)`. `K_req` remains the schedule
  request; it must never be rewritten as realized cost.
- Physical execution must satisfy
  `K_backbone = K_unique = K_eff <= K_req`. Repetition, tail padding, video
  deletion, and length-conditioned request generation are forbidden.
- `L < 16`, a non-quantized result, duplicate gathered positions, or any
  discrepancy between the selector ledger and the actual heavy-backbone input
  fails closed. The full-200 preflight must prove that no registered natural
  training window is sub-quantum.
- The realized mean and histogram are measured facts, not assumed to remain 384.
  Every successful optimizer step records requested, feasible, unique and actual
  heavy-backbone K separately; AMP retries cannot create committed rows.

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
requested evaluation K. Natural windows use the same deterministic cap, so a
fixed requested-K384 run remains a fixed-policy control and is not dynamic
inference. Each run must bind the same checkpoint SHA-256, split assignment,
position policy, and no-padding ledger. The O1 source manifest records
`detector_training_exposure=mixed_k_registered_panel`; the formal record builder
rejects the fixed-K diagnostic label.

## Failure handling and evidence levels

- Missing schedule metadata, a non-candidate requested K, requested-histogram
  drift, unexplained realized-cost drift, mixed-K batch execution, padding,
  repetition, or selector/backbone accounting disagreement is fatal.
- A deterministic reexecution is reproducibility evidence only, not an
  independent training seed.
- Passing unit tests or the code gate means `implemented/tested`, not
  `empirically_supported`.
- O1 can become empirically supported only after the mixed-K checkpoint and all
  hash-bound budget cells complete on the frozen train-only development role.

## Verification

Focused tests must cover the exact per-video histogram, mean K=384, deterministic
rotation, AMP-replay identity, probe-free construction, exact-K/no-padding
ledger on full windows, deterministic short-window aliases, fixed requested
evaluation K, actual heavy-backbone tensor accounting, contaminated/missing
metadata rejection, 6000-update config contract, launcher fail-closed behavior,
and fixed-K diagnostic rejection. Before a fresh Stage-A release, a clean-commit
Slurm gate must run the real dataset decoder, selector, physical gather and heavy
backbone on all four requests including at least one natural short window. Its
immutable receipt is a prerequisite of the matrix manifest.

The scheduler release uses seven jobs: for each of three seeds one sequential
control group executes dense, fixed-uniform and mixed-uniform arms; an isolated
DUCA job runs independently for that seed; one `afterok` job seals all twelve
logical cells. This is scheduler grouping only. It prevents a mixed-control
failure from suppressing the DUCA arm and stays within the cluster's sixteen-job
submission ceiling.

Self-review: the design contains no placeholders; the exposure histogram,
selection inputs, failure conditions, claim boundary, and evaluation identity
are explicit and mutually consistent.
