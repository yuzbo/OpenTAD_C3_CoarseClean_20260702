---
type: experiment
node_id: exp:scnr-residual-centering-matched-training-v1
title: "SCNR residual-centering matched development training v1"
stage: tested
status: accuracy_screen_pass_paired_cost_authorized
outcome: single_seed_centering_accuracy_supported_cost_pending
added: 2026-08-06
updated: 2026-08-09
---

# SCNR residual-centering matched development training v1

## Purpose

Test whether the offset-only calibration that restored role reachability in the
frozen-checkpoint probe improves high-IoU development accuracy after fresh
training. This is the first performance test of the repair, not proof that the
complete ROI+TokenSelect Hybrid is already paper-ready.

## Frozen comparison

- `none_control`: fresh G1 `native_1cell_main` training with branch calibration
  disabled.
- `residual_window_center`: the same fresh G1 training with only the valid
  full-window residual modifier mean subtracted.

Both use seed 3407, 60 epochs, exactly 9,600 successful updates, exact
`B=24576`, fully dynamic roles/`K_t` including zero, true ragged execution,
masked-zero carrier, and identical data, initialization, optimizer, AMP, EMA,
evaluator, and NMS. Old M2 checkpoints are not reused. The normalized complete
training-recipe SHA-256 shared by both cells is
`34defbdbc30e7fff10bbb05d7e6665dd29b8128f8f03cd389250bca9e3e7493c`.

## Terminal execution evidence

Exact runtime `16137484c5ccad422e017e67a81c1a07d1ed2fbb` passed clean N16R4
Linux/Torch regression `93/93`. Atomic root:
`/data/run01/sczc063/yuzibo/scnr_residual_centering_matched_training_16137484_s3407_20260806_061352`.
Jobs `1223819` (`none_control`), `1223820` (`residual_window_center`), and
after-any finalizer `1223821` all completed `0:0`.

Both cells completed 60 epochs and 9,600 successful updates, published one
epoch-59 `state_dict_ema` checkpoint, and passed two serial same-GPU strict
math-SDPA accuracy replays. Within each cell the two prediction files are
byte-identical across 40 videos/80,000 candidates; route payload, population,
branch summary, and metrics also match exactly.

- control checkpoint SHA-256:
  `5350a03b1584ab8e0023b6c212fc2a3b8526c45de169b5660d827762a9dd6ff4`;
- centered checkpoint SHA-256:
  `e45a37708c68e1ea02bec02bc14dfbe199ea8303f562088469d98a8fe45c7028`;
- common accuracy population SHA-256:
  `35aaa9192b4dfd4bd03599450fbbceed6c7d60e98d8df7f582bd39df26f40aa8`;
- finalization SHA-256:
  `2a9351a3c21c850f28aab4bd162f7b69f3ca40921a97304431a8a760d6ebbe8a`.

## Accuracy result

| Variant | Avg-mAP | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | high-IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| `none_control` | 10.52 | 13.31 | 12.40 | 10.99 | 8.90 | 6.98 | 7.94 |
| `residual_window_center` | 12.57 | 16.19 | 14.66 | 12.82 | 11.04 | 8.14 | 9.59 |
| centered minus control (pp) | +2.05 | +2.88 | +2.26 | +1.83 | +2.14 | +1.16 | +1.65 |

All preregistered signs pass: mAP@0.6 and mAP@0.7 are strictly higher and
Avg-mAP is non-lower. The terminal status is
`PASS_ACCURACY_SCREEN_PAIRED_COST_AUTHORIZED`, decision
`RUN_SAME_GPU_ABBA_BAAB_FULL_STACK_COST`, with empty errors.

## Structural and dynamic-route result

The uncentered control remains residual-only: selected
context/ROI/residual counts are `0/0/3,342,336`. Its `K_t` spans `0..215`, mean
64, with 754 zero-tubelet assignments.

The centered model selects context/ROI/residual
`210,925/1,613,683/1,517,728`, or `6.31%/48.28%/45.41%`, and passes all three
reachability conditions. Its maximum absolute valid post-centering residual
mean is `2.6075e-7`. Its observed `K_t` spans `5..206`, mean 64, with no zeros
on this Gate replay. This observation does not remove protocol support for
`K_t=0`; no minimum-K constraint was introduced.

Equal B does not imply equal system cost: the changed `K_t` distribution can
change ragged attention pairs. Cost therefore remains an empirical gate.

## Integrity audit and evidence boundary

Independent read-only integrity review classifies provenance/no-leak, score
handling, artifact exactness, and critical-path reachability as PASS. Scope and
evaluation are valid only under a single-seed, development-only ceiling. The
result-to-claim review supports the narrow preregistered accuracy screen with
high confidence, but not generalization or efficiency.

Allowed wording:

> On THUMOS14 Gate development, one fresh seed-3407 matched G1 comparison at
> B=24576 passed the preregistered residual-centering accuracy screen,
> improving Avg-mAP by 2.05 pp, mAP@0.6 by 2.14 pp, and mAP@0.7 by 1.16 pp.
> This authorizes, but does not establish, a counterbalanced full-stack cost
> study.

Not supported: multi-seed significance, official-test performance, complete
Hybrid complementarity, floor causality, superiority to dense/fixed/random/free
TokenSelect, efficiency, second-detector transfer, or a final paper method.

## Next gate

The authorized successor is
`exp:scnr-residual-centering-paired-cost-v1`: one Slurm Job, one physical GPU,
one continuous 20-ms NVML sidecar, and eight serial `ABBA+BAAB` passes over the
full decode-to-NMS Gate path. It reuses these audited checkpoints without
training. Only accuracy plus cost non-inferiority may open fresh seeds
3408/3409. Official test and paper claims remain closed.

Designs:

- `docs/superpowers/specs/2026-08-06-scnr-residual-centering-matched-training-v1-design.md`;
- `docs/superpowers/specs/2026-08-09-scnr-residual-centering-paired-cost-v1-design.md`.
