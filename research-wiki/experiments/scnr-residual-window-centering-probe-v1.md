---
type: experiment
node_id: exp:scnr-residual-window-centering-probe-v1
title: "SCNR residual-window centering frozen-checkpoint probe v1"
stage: tested
status: pass_structural_reachability_only_matched_training_design_authorized
outcome: residual_offset_centering_restores_context_and_roi_role_reachability_in_both_frozen_m2_arms
added: 2026-08-06
updated: 2026-08-06
---

# SCNR residual-window centering frozen-checkpoint probe v1

## Purpose

Test the smallest offset-identifiability repair suggested by the authorized
categorical role evidence. This is a no-training mechanism probe on the frozen
M2 G1/G2 checkpoints, not a performance experiment.

## Single intervention

For each complete 384-tubelet window, subtract the differentiable all-valid
candidate mean from `delta_residual` before unchanged Scheme-A role argmax and
global top-B. Leave `q_base`, signed `delta_roi`, exact-zero context, exact
`B=24576`, fully dynamic `K_t` including zero, true ragged execution, masked-zero
carrier, and VideoMAE two-frame tubelets unchanged. Default mode remains `none`.

Per-tubelet centering is deferred because it adds a temporal-uniformity prior.
RMS/temperature/bounding and ROI-conditioned complement are deferred because
they alter a second property or mechanism. Fixed role quotas, target fractions,
post-hoc reassignment, and independent `q_ctx` remain forbidden.

## Gate

The strict math-SDPA duplicate replay must be deterministic, every window must
select exactly B valid native candidates, and all audit/provenance/no-leak fields
must validate. Both G1 and G2 must have nonzero aggregate valid context wins,
nonzero aggregate valid ROI wins, and a nonzero selected non-residual count in at
least one window. Failure stops before training. Pass authorizes only a newly
frozen matched `none` versus centering development-training protocol, never mAP,
cost, floor, M3, official-test, operational-Hybrid, or paper claims.

Full design:
`docs/superpowers/specs/2026-08-06-scnr-residual-window-centering-probe-v1-design.md`.

## Implementation status

The opt-in `residual_window_center` mode is implemented immediately after the
ROI modifier and before the unchanged Scheme-A role argmax/global selector. It
uses the differentiable valid-candidate mean, preserves invalid candidates,
and leaves the default mode at `none`. The wrapper publishes calibration
telemetry, while the new M2-bound runner executes two strict math-SDPA
`--not_eval` replays per arm, verifies raw-prediction and route-payload
determinism, and classifies only the preregistered structural gate. It emits no
metric evaluation and cannot authorize training by itself.

Local Python compilation, shell syntax, whitespace, pure-contract tests, and
standalone result-classification checks pass. The complete Torch-backed suite
cannot load the local Windows `c10.dll`; the authoritative clean Linux/Torch
regression and frozen G1/G2 Slurm outcomes are recorded below.

## Tested result

Exact implementation source `091f9f9b57e68a4706a91d8b3b9176ddc88d0c6c`
passed the clean N16R4 GeoRoute/probe suite `90/90` and required C3 regressions
`20/20`. Frozen-checkpoint Jobs `1223783/1223784` completed `0:0` under
`/data/run01/sczc063/yuzibo/scnr_residual_window_centering_probe_091f9f9b_s3407_20260805_211124`.
Each arm ran `centered_a -> centered_b` serially on one Slurm-visible GPU with
strict math SDPA, TF32 disabled, no metric evaluation, and exact raw prediction
plus route-payload parity.

G1 selected context/ROI/residual counts are
`168,733/421,121/2,752,482`; G2 counts are
`186,976/429,896/2,725,464`. All-valid role counts are respectively
`5,682,153/566,244/5,240,883` and
`5,373,479/577,817/5,537,984`. Both arms therefore pass every registered
structural condition. The maximum absolute post-centering residual mean is
`2.81e-7` for G1 and `3.04e-7` for G2, below the frozen `1e-4` tolerance.
Exact B, unique/ragged execution, 136-window population, and no-leak guards
validate; `K_t` remains fully induced (`4..215` G1, `4..207` G2 in this frozen
probe) and no padding is introduced.

G1 result internal/file SHA-256 values are
`4688a95063f1320527cf6de7e4638c1379ec946d3d45a4aa1a288607564edac2` /
`f95a04a967d4dfe93f9f5a0a6881b8b22e27b0c1e4eaa2c0316e6d633646c9db`;
G2 values are
`1dc87ffdda1dab5095f83a18e27e24f3713ab22cde585e53079a4b141d51312f` /
`7d54430e6ff044d7ac65e1bda0bdc0496b253e15c2eeb8882cc493ba53ab8920`.

## Decision

The frozen structural gate passes and authorizes designing one matched
development-only `none` versus `residual_window_center` training protocol. It
does not establish accuracy, complementarity, cost, floor selection, M3,
official-test, or paper claims. No Pro discussion is needed before freezing
that single-variable protocol.
