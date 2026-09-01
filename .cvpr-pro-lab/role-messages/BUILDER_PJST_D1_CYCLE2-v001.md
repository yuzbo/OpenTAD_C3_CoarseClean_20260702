# BUILDER_PJST_D1_CYCLE2-v001

## Binding

- role: Builder
- workspace: `C:/Users/skywalker/.codex/worktrees/duca-pjst-cycle2-builder-20260826`
- branch: `codex/duca-pjst-cycle2-builder-20260826`
- frozen parent: `b2ccfccab5b4912b59954afcc9b0364955327f7c`
- access: writable only in the bound worktree; durable role returns belong in the canonical project's `.cvpr-pro-lab/role-returns/` and must not be committed.

## Frozen scientific contract

Implement only `PJST-D1` (Derivative-Only Physical-Jacobian Scaled Tubelet) as fixed-selector representation attribution on the historical H65 semantic indirect nonuniform per-frame selection path. Preserve K=384, selected RGB frames/order/mask, VideoMAE-S, Adapter, ActionFormer, head/loss/NMS, data/split/evaluator, optimizer, and schedule. Do not change the selector or introduce support-aware transport, dynamic K, cliplet selection, new parameters, or a new scientific route.

The matched training contract is the existing 30-epoch uniform Stage-1 (`duca_sampling_rate_curriculum_stage1_uniform384.py`) followed by the existing 60-epoch/6000-successful-update Stage-2 (`duca_sampling_rate_curriculum_stage2_joint384.py`). Do not use the unrelated 20+40 compression configs.

## Mandatory process

1. Before editing, write `BUILDER_PJST_D1_CYCLE2_MINIMAL_CHANGE_PLAN-v001.md` to the canonical role-return directory and send the Coordinator a compact `MINIMAL_CHANGE_PLAN` message.
2. In the same turn, implement and test the plan. Do not stop after prose.
3. Commit one clean candidate and write `BUILDER_PJST_D1_CYCLE2_IMPLEMENTATION-v001.md` with parent/head, exact diff, commands/results, evidence class, and next owner=independent Critic.

## Required implementation

- Add a pure PJST metadata helper. Input is global tubelet-pair coordinates `[B,24,8]`; packed metadata is `[B*24,8]`. Never flatten it as `[B,192]` or repeat the whole row per clip.
- For each adjacent tubelet pair, in float32 compute canonical spacing `delta_can`, actual physical spacing `delta_act`, valid-pair mask, and `scale=delta_can/delta_act`. Exact-uniform rows must bypass before any cast or division. Invalid/padded pairs must be identity.
- Apply exactly one original `PatchEmbed`, then pairwise midpoint/difference reconstruction: `m=(x_minus+x_plus)/2`, `v=scale*(x_plus-x_minus)/2`, `y_minus=m-v`, `y_plus=m+v`; cast back to the input token dtype. No second convolution and no learnable state.
- Wire metadata through `BackboneWrapper` and VideoMAE. For checkpointing along dimension 0, slice all PJST metadata with the exact same chunk boundaries. If PJST is enabled with temporal chunking on dimension 2, fail closed with a clear error rather than silently misaligning metadata.
- Preserve the exactly-once selected-axis to physical-time segment remap before threshold/top-k/NMS.
- Produce matched `MATCHED_OFF` and `PJST_D1_ON` configs sharing the same historical 30+60 contract; OFF must execute the identical code/data path with PJST disabled. Add one project-local launcher/validator surface only if the existing launcher cannot express this pair.

## Required discriminating checks

Cover at least: `[B,24,8] -> [B*24,8]` packing; B>1 no cross-sample leakage; checkpoint dim-0 chunk slicing; rejection of dim-2 PJST checkpointing; exact-uniform bitwise identity; invalid/padded identity; nonuniform numerical formula; float32 work and dtype restoration; gradient finiteness/nonzero flow without new parameters; one PatchEmbed call; state-dict key identity OFF vs ON; matched config invariants; and pre-NMS exactly-once remap.

Run focused local checks that are available without data. Do not access official validation data, submit Slurm/GPU jobs, train, evaluate mAP, or make efficacy claims.

You are not alone in this repository. Do not revert or overwrite other users' work. Restrict edits to the frozen allowed surfaces and accommodate the clean parent state.
