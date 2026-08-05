---
type: experiment
node_id: exp:scnr-residual-centering-matched-training-v1
title: "SCNR residual-centering matched development training v1"
stage: experiment_running
status: both_p0_passed_both_fresh_trainings_running
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
loader boundary.

## Running evidence

Exact runtime `16137484c5ccad422e017e67a81c1a07d1ed2fbb` passed clean N16R4
Linux/Torch regression `93/93`. The atomic deployment root is
`/data/run01/sczc063/yuzibo/scnr_residual_centering_matched_training_16137484_s3407_20260806_061352`;
deployment SHA-256 is
`71b10681118c57a845deb33a3f0f98d269ae05ac4b6b9e0e0114182ee1998b59`.
Fresh training Jobs are `1223819` (`none_control`) and `1223820`
(`residual_window_center`); after-any finalizer `1223821` depends on exactly
those two Jobs. Both no-performance P0 gates passed, with report SHA-256 values
`0f24c8710436bdb0090079b8a6b68e1e93d166519e0ac21145e3194c5685db56`
and `edd5617a167b198eb5a7ab924df86cbabb5004dfc04b79685c213c4c66eecc6d`,
respectively. Both fresh trainings entered epoch 0 on `g0059`. No final
checkpoint, duplicate accuracy result, contrast, or metric is yet valid.

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
