---
type: experiment
node_id: exp:scnr-dynamic-role-calibration-diagnostic-v1
title: "SCNR dynamic role-calibration diagnostic v1"
stage: implemented
status: local_compile_pass_remote_tensor_tests_pending
outcome: pending
added: 2026-08-06
updated: 2026-08-06
---

# SCNR dynamic role-calibration diagnostic v1

## Purpose

Explain the terminal M2 hard-role collapse before changing the model or launching
M3. The diagnostic distinguishes two mechanisms:

1. `delta_residual` already wins over both the signed geometry-derived
   `delta_roi` and exact-zero context modifier on almost every valid candidate;
2. all-valid roles remain diverse, but shared `q_base` plus global top-B causes
   the selected subset to become residual-only.

This is a result-blind mechanism replay, not a new performance arm. It must not
force role usage, choose a floor, read GT into routing, retrain a checkpoint, or
alter the selected support.

## Frozen diagnostic contract

- Reuse the immutable G1/G2 epoch-59 EMA checkpoints and exact 136-window Gate
  population from M2; no training, resume, optimizer, scheduler or EMA state.
- Preserve the hard route, exact `B=24576`, VideoMAE two-frame tubelet, true
  ragged executor, masked-zero carrier and all model/config parameters.
- Enable the new telemetry only in the accuracy replay, after route construction;
  it is excluded from timed cost and reports
  `changes_route_or_execution=false`.
- Require the complete replay's prediction file SHA-256 to match the corresponding
  frozen M2 prediction artifact exactly. Any mismatch invalidates the diagnostic.
- Bind model runtime `6ee97336` separately from the future clean diagnostic
  execution commit and publish a new namespace; never edit the M2 root.
- Keep role target fractions, fixed quotas, post-hoc reassignment and independent
  `q_ctx` absent.

## Added observations

For every window and for `valid`, `selected`, and `unselected` scopes, record
compact distributions of:

- shared `q_base`;
- `delta_roi` and `delta_residual`;
- `delta_residual-delta_roi`;
- ROI/residual minus the exact-zero context modifier;
- winning modifier and top1-minus-top2 role margin.

Also record all-valid, selected and unselected role counts/fractions, missing
selected roles, dominant-role fraction and selected/valid role-fraction ratio.
These are observed statistics only; no balance threshold is a pass criterion.

## Interpretation

| Pattern | Diagnosis | Authorized next step |
| --- | --- | --- |
| Residual dominates all-valid and selected candidates with large positive margins | branch bias/scale non-identifiability | freeze a minimal bounded or centered residual-calibration ablation |
| All-valid roles are diverse but selected roles collapse | interaction between `q_base`, modifier magnitude and global ranking | inspect selected-vs-unselected utility coupling; do not add quotas |
| Margins are near zero with stable residual wins | argmax tie/initialization sensitivity | freeze a symmetry/tie-handling calibration ablation |
| ROI/context are present but only in isolated windows | temporal/data-dependent collapse | stratify window statistics before choosing an intervention |

No outcome above alone authorizes M3, official test or a Hybrid claim. A model
intervention must be separately trained and compared with the unchanged Scheme-A
baseline under exact B. Candidate interventions such as per-tubelet modifier
centering/RMS matching, bounded signed residual logits, or ROI-conditioned
residual complement are `discussed`, not approved or implemented.

## Current implementation state

The wrapper now has an opt-in
`georoute_role_calibration_telemetry_enabled` flag. Default and all frozen M2
configs keep it false. The diagnostic path computes no gradients, changes no
route token, and emits nested schema
`scnr_dynamic_role_calibration_window_v1`. A standalone analyzer validates
population/no-leak fields, reconstructs exact role partitions and produces
weighted branch statistics without assuming JSON key order or role quotas.

Python compile and whitespace checks pass locally. Windows Torch tests remain
environment-blocked by the known `c10.dll` load failure; clean N16R4 Linux/Torch
focused tests and the two frozen-checkpoint replays are pending.
