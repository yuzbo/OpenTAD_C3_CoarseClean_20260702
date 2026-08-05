---
type: experiment
node_id: exp:scnr-dynamic-role-calibration-diagnostic-v1
title: "SCNR dynamic role-calibration diagnostic v1"
stage: implemented
status: prediction_parity_failed_pair_neutrality_implemented_pending_remote
outcome: no_valid_role_calibration_result
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

Implementation commits through `2c39ce58791704a29745e9172565df42fba4723b`
separately bind source model runtime `6ee97336775a09611f10423e07cafcea375e191a`,
the original config/checkpoint/prediction receipts, source population SHA-256 and
source dataset count. Clean N16R4 Linux/Torch validation passed `51/51` focused
dynamic tests and `20/20` required C3 regressions.

## Terminal replay attempts

No attempted replay currently supplies valid role-calibration evidence:

- Jobs `1223595/1223596` (`c4815388`) failed before inference because generic
  Phase-M supplied forbidden `--not_eval`.
- Jobs `1223601/1223602` (`469dfe63`) failed before inference because generic
  development profiling violated frozen M2 accuracy's `profile=false`.
- Jobs `1223615/1223616` (`b6c792fc`) completed inference but the runner looked in
  the wrong output directory, so its required artifact contract could not close.
- Jobs `1223625/1223626` (`0f97307d`) completed inference but the runner expected
  the earlier diagnostic schema after the execution path had moved to the formal
  schema. They are not salvageable under that frozen contract.
- Jobs `1223640/1223641` (`2c39ce58`) completed inference and schema validation,
  but failed the frozen raw prediction-SHA parity gate. Their role telemetry is
  sealed and must not be inspected or interpreted.

The `2c39ce58` replay root is
`/data/run01/sczc063/yuzibo/scnr_dynamic_role_calibration_diag_2c39ce58_s3407_20260806_0240_defaultmem`.
Its G1/G2 replay prediction SHA-256 values are respectively
`3fa61cbcb1722aebfda04c3a87b9e8436fddff4af2ec7cea3ce70fd0c912496b`
and `92c3e3dea171e2afb12e4845069513b6ae6c3f2edcdc3bdd91731f0a7c2c5b9c`,
which differ from source `79c407c4...` and `727694fa...`. Both arms preserve the
same ordered 40-video key set and 80,000 prediction records. Exact
`(video,label,start,end)` identity overlap is `76,660/80,000` for G1 and
`78,387/80,000` for G2; therefore this is not merely JSON key order or formatting.
The original M2 jobs ran on `g0024`, whereas these replays ran on `g0044/g0048`,
so cross-node rerun drift and instrumentation effect remain confounded.

## Registered successor

The only authorized successor is
`exp:scnr-role-instrumentation-neutrality-pair-v1`. It executes source-formal
telemetry with role calibration OFF, then ON, serially in one Slurm job on one
visible GPU. Evaluation path, config, checkpoint, seed, 136-window population,
exact `B`, formal telemetry and `profile=false` are common; only the
role-calibration extension and its provenance/output path differ. OFF/ON raw
prediction SHA equality is a hard gate. Source parity remains required before the
original frozen role diagnostic can be interpreted. Local Python/Bash/whitespace
checks pass; remote tests and execution are pending.
