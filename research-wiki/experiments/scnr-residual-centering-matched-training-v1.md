---
type: experiment
node_id: exp:scnr-residual-centering-matched-training-v1
title: "SCNR residual-centering matched development training v1"
stage: implemented
status: implemented_pending_exact_commit_remote_precheck
outcome: pending
added: 2026-08-06
updated: 2026-08-06
---

# SCNR residual-centering matched development training v1

## Purpose

Test whether the offset-only calibration that restored role reachability in the
frozen-checkpoint probe improves high-IoU development accuracy after fresh
training. This is the first performance test of the repair, not a proof that the
complete ROI+TokenSelect Hybrid is effective.

## Cells

- `none_control`: fresh G1 `native_1cell_main` training with branch calibration
  disabled.
- `residual_window_center`: the same fresh G1 training with only the valid
  full-window residual modifier mean subtracted.

Both use seed 3407, 60 epochs, exactly 9,600 successful updates, exact
`B=24576`, fully dynamic roles/`K_t` including zero, true ragged execution,
masked-zero carrier, and identical data, initialization, optimizer, AMP, EMA,
evaluator, and NMS. Old M2 checkpoints are not reused.

## Implementation status

The study now has a fail-closed contract, per-arm runner, atomic two-arm Slurm
deployer, and `afterany` finalizer. The contract derives both cells from one
complete G1 training config and compares a normalized complete-recipe hash, so
optimizer, losses, augmentation, detector/head, scheduler, data and all other
shared fields cannot drift while only work-directory/receipt identity and the
registered calibration mode vary. The deployment receipt binds both held stage
Jobs and the held finalizer before releasing any Job.

Local Python compilation and Bash syntax checks pass. Pure contract, inherited
M2, and required C3 regressions pass `57/57`; the new matched-training suite is
`10/10`. Torch-backed collection is deferred to the clean N16R4 Linux snapshot
because the local Windows installation stops at the known `torch/c10.dll`
loader boundary. No Slurm Job, checkpoint, prediction, or metric exists yet.

## Integrity and decision

Each final EMA checkpoint receives two same-GPU strict math-SDPA Gate replays.
Raw predictions, route payload, population, and metrics must match exactly within
the cell. Every provenance, no-leak, exact-B, ragged, sidecar, and matchedness
gate must pass before contrasts exist.

The centered cell must retain nonzero selected context and ROI counts and a
residual selected fraction below one. The accuracy screen requires centered
`mAP@0.6 > none`, centered `mAP@0.7 > none`, and centered Avg-mAP `>= none`.
Any incomplete cell yields empty contrasts; any tie/crossing/failure yields HOLD.

A pass authorizes only a separately frozen ABBA+BAAB paired full-stack cost
study. Only a later accuracy-cost Pareto pass may open seeds 3408/3409. M3,
official test, floor selection, complementarity, efficiency, and paper claims
remain closed.

Full design:
`docs/superpowers/specs/2026-08-06-scnr-residual-centering-matched-training-v1-design.md`.
