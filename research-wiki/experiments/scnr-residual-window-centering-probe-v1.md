---
type: experiment
node_id: exp:scnr-residual-window-centering-probe-v1
title: "SCNR residual-window centering frozen-checkpoint probe v1"
stage: designed
status: preregistered_pending_implementation
outcome: pending
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

