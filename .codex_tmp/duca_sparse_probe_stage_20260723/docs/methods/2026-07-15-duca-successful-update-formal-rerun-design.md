# DUCA Successful-Update Formal Rerun Design

## Decision

The replacement DUCA P0 suite keeps the audited fixed-384 model and four-arm
scientific comparison unchanged. This change repairs training and evidence
integrity only. It does not add a new selector loss, dynamic budget, external
action prior, detector head, or physical-time geometry.

## Formal Training Contract

- Task: offline temporal action detection.
- Arms: exact uniform, direct-a5, transition beta=0, transition
  counterfactual.
- Shared exposure: 100 sampled batches per epoch, 132 epochs, exactly 13,200
  successful optimizer updates per arm.
- AMP overflow: replay the same in-memory batch after restoring CPU/CUDA RNG
  and every model buffer. Retain GradScaler backoff. Advance optimizer,
  scheduler, EMA, and DUCA schedule exactly once after a successful replay.
- Replay exhaustion: fail closed. A consumed batch without a successful update
  is forbidden in a formal run.
- Resume: only an epoch-boundary checkpoint carrying the formal training state
  and GradScaler state may resume a formal run.
- Checkpoints remain every five epochs. The primary checkpoint is fixed before
  execution to zero-based epoch 131, `state_dict_ema`.

## Evidence Contract

Every checkpoint receives atomic metadata binding the exact commit, source and
resolved config identities, variant, seed, Slurm job, protocol hashes,
successful update counts, scheduler/EMA/DUCA schedule exposure, AMP attempts,
and replay count. A terminal evaluator must load `epoch_131.pth` and
`state_dict_ema`, then write the exact prediction dictionary and structured
metric JSON.

Post-run evidence is accepted only when SHA-256 binds all of:

1. run manifest;
2. terminal checkpoint and sidecar;
3. cumulative training audit;
4. prediction artifact;
5. structured evaluation result;
6. evaluator implementation source.

No intermediate THUMOS test metric can choose a checkpoint. Formal training
disables intermediate test evaluation; the terminal evaluation is the sole
primary result.

## Counterfactual Scope

The counterfactual arm remains detached hard one-swap relative ranking
distillation. It is not renamed signed utility or direct detector-gradient
learning. The exact-commit CUDA gate must expose a deterministic positive
direction check in addition to finite diagnostics. Full mechanism claims still
require train-window alignment evidence after the seed-0 result.

## Deployment DAG

1. Focused CPU tests and exact-uniform/config validators.
2. Exact-commit real AdaTAD CUDA gate.
3. Four-arm, real-batch, ten-successful-update DDP-wrapper pilot including an
   injected AMP replay test.
4. Four seed-0 full runs under Slurm.
5. CPU aggregation job validates four post-run evidence files and emits the
   matched terminal table.
6. Only a positive seed-0 high-tIoU decision may authorize seeds 1 and 2.

## Acceptance Criteria

- No missed successful update and no replay exhaustion.
- Per arm: successful updates, scheduler updates, EMA updates, and DUCA schedule
  step all equal 13,200.
- Exact K=384 and max-hole contract remain valid.
- All four terminal artifacts pass byte-level validation under one shared
  protocol hash.
- C3/C4 remain unproven until the sealed terminal table exists.
