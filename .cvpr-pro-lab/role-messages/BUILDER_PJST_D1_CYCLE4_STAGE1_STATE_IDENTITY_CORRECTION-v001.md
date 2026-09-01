# Builder focused correction 2 — PJST-D1 Stage-1 state identity

Work only in the clean Builder worktree:

`C:/Users/skywalker/.codex/worktrees/duca-pjst-cycle4-builder-20260826`

Frozen parent: `dc260fad42be4fe8e3a77459b2142a115c4866c7`.

## Observed deterministic failure

Matched formal jobs `1256317` and `1256318` passed the full Linux focused suite and canonical dataset binding, then both failed before any optimizer update during strict Stage-1 model initialization:

`unexpected=['backbone.model.backbone.blocks.0.relative_physical_time_scale']`

The frozen epoch-29 Stage-1 checkpoint contains exactly one related scalar in both state and EMA state. It is scalar-shaped and exactly zero. The historical H65-OFF audit records it as a registered architecture-identity scalar with `single_clock_admission=False`, not an active SingleClock intervention. The accepted PJST-D1 decision forbids enabling SingleClock, ignoring unknown keys, using `strict=False`, changing checkpoint, or adding a different state identity.

## Exact scope

Make the smallest common OFF/ON config correction:

1. In both `configs/adatad/thumos/duca_pjst_d1_stage2_off.py` and `..._on.py`, explicitly keep `model.single_clock_admission=False`.
2. In both configs, set the existing VideoMAE backbone constructor field `model.backbone.backbone.relative_physical_time_residual=True` solely so block 0 registers the zero scalar required by the frozen Stage-1 state dict.
3. Do not pass SingleClock physical-coordinate metadata. Do not enable any SingleClock detector route, gate, residual computation, optimizer override, or new trainable mechanism. With `single_clock_admission=False`, no relative physical time tensor reaches the backbone; the zero scalar is architecture identity only.
4. Keep OFF/ON resolved configs identical except `work_dir` and `model.backbone.custom.pjst_derivative_only`.
5. Add the smallest focused assertion (test and/or existing validator) proving both configs share `single_clock_admission=False` and `relative_physical_time_residual=True`, while the allowed OFF/ON diff set stays unchanged.
6. Do not change checkpoint loader strictness, reset-key whitelist, model implementation, launcher, data, selector, PJST formula, optimizer, schedule, seed, evaluator, split, NMS, checkpoint path/SHA, or scientific documents.

Run feasible focused checks. Commit and push the existing branch. Report commit, changed files, exact checks, and next_owner=Critic. No remote run or experiment.
