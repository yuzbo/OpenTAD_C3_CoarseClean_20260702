# BUILDER_PJST_D1_CYCLE3-v001

## Binding and status

- role: sole Builder for a new clean implementation cycle
- workspace: `C:/Users/skywalker/.codex/worktrees/duca-pjst-cycle3-builder-20260826`
- branch: `codex/duca-pjst-cycle3-builder-20260826`
- clean parent: `b2ccfccab5b4912b59954afcc9b0364955327f7c`
- previous packages `877d893f/84325205` and `987f4811/c8faf96b` are terminal negative implementation evidence. Do not cherry-pick or copy their patches.

Read the complete accepted contract in `.cvpr-pro-lab/pro-reviews/runs/duca-pjst-derivative-causal-freeze-v002/PRO_DUCA_PJST_DERIVATIVE_CAUSAL_FREEZE-v002.md` from the canonical project.

## Mandatory process

1. Before edits, write `BUILDER_PJST_D1_CYCLE3_MINIMAL_CHANGE_PLAN-v001.md` to the canonical role-return directory and notify the Coordinator with `MINIMAL_CHANGE_PLAN`.
2. Implement, test, and commit in the same turn. A prose-only return is invalid.
3. Write `BUILDER_PJST_D1_CYCLE3_IMPLEMENTATION-v001.md` with exact parent/head/diff/commands/results/evidence and `next_owner=independent Critic`.

## Frozen science

Only implement PJST-D1 as a fixed-selector representation attribution on H65: same semantic indirect nonuniform per-frame positions, selected RGB, ordered K=384 input, VideoMAE-S/Adapter/ActionFormer, loss/NMS/evaluator/data/seed and 30+60 schedule. PJST changes only the first 2-frame Conv3D mixing by scaling the odd/derivative mode. No support-weighted appearance, SingleClock, Query, dynamic K, cliplet selection, new learnable parameter/buffer or selector change.

## Exact production implementation

### Metadata

- Add a pure helper in `opentad/models/utils/temporal_grid.py` taking global int64 selected positions `[B,384]`, dense valid length and prefix mask.
- Reuse the sole `exact_uniform_positions(T_b,n_b)` implementation. Pair global ranks `(16c+2r,16c+2r+1)` and return `actual_delta`, `canonical_delta`, `pair_valid`, and audit-only support as `[B,24,8]`; packed scale/valid is `[B*24,8]` in batch-major clip order.
- For each sample, compare `P_valid == U_valid` before any cast/division. Exact-uniform samples expose an explicit bypass mask and must not execute float arithmetic. Valid irregular pairs use float32 `delta_can/delta_act`; invalid/partial pairs have scale 1 and never change RGB. Strictly increasing valid positions imply positive actual delta; fail closed otherwise.

### Runtime reachability and OFF identity

- `SingleStageDetector.forward_train/test` must pass `metas` to `BackboneWrapper` when accepted by its signature. Preserve all other backbone callers.
- `BackboneWrapper` reads a real config flag (e.g. `custom.pjst_derivative_only`, default false). OFF must not call the PJST helper, construct PJST scale/valid metadata, or pass PJST kwargs. ON extracts the existing selector physical positions from per-sample metas (`irregular_selected_positions`, with the existing documented bridge fallback only), validates length/mask, creates `[B,24,8]`, packs to `[B*24,8]`, and passes it with flattened frames.
- With temporal checkpointing, ON permits only `chunk_dim=0`; slice every packed metadata tensor with exactly the same precomputed chunk boundary as frames. ON with `chunk_dim=2` fails before checkpoint execution. OFF retains the unmodified base checkpoint behavior for both supported dimensions.

### PJST input transform

- Add a pure, testable helper (in the allowed VideoMAE/temporal-grid surface) that maps `X[Bclips,3,16,H,W]` and packed metadata to `Y`.
- View `X` as `[Bclips,3,8,2,H,W]`. Start from `Y=X.clone()`. Only rows not marked exact-uniform and pairs marked valid execute float32:
  `m=(x_minus+x_plus)/2`, `v=scale*(x_plus-x_minus)/2`, `y_minus=m-v`, `y_plus=m+v`; cast written values back to X dtype. Uniform rows and invalid pairs remain the original bytes.
- `VisionTransformerAdapter.forward` accepts PJST kwargs at the model-level signature, applies the pure transform immediately before the existing `self.patch_embed(x)[0]`, and calls PatchEmbed exactly once. Do not add PJST args/state to Transformer blocks and do not add a convolution/parameter/buffer/key.

### Physical decode

- In `single_stage.py`, move the existing `_remap_selector_segments_for_post_processing` call to immediately after raw per-video `segments/scores` extraction and before confidence filtering, top-k, IoU and NMS. Remove the old later call. Do not modify the mapping function, scores, labels, thresholds, ordering, NMS or output format. OFF and ON share this claim-neutral correction.

## Matched configs and real launcher

- Add matched OFF and ON Stage-2 configs inheriting `duca_sampling_rate_curriculum_stage2_joint384.py`. Both require the same Stage-1 epoch-29 terminal checkpoint/hash, seed 3407, 60 epochs/6000 successful updates, every-5-epoch resumable checkpoints, final/final-EMA and official evaluator. Only ON sets the backbone PJST flag.
- Freeze/replay selector behavior in both configs: fixed learned policy alpha, zero detector-to-selector and auxiliary-selector update weights, no ASFormer adaptation; both arms must resolve to identical selector/exposure contracts. Do not edit selector code.
- Adapt the real `scripts/run_duca_h65_matched_cycle4_n16r4.sbatch` pattern, not an environment-only stub. New launcher supports `PRECHECK_ONLY=1` and formal `MODE=STAGE2_OFF|STAGE2_ON`, validates the same Stage-1 checkpoint/hash/epoch 29, invokes `tools/train.py <matched-config> --seed 3407` under Slurm, uses distinct new roots, and preserves the canonical N16R4 data/annotation/category/pretrain paths and distributed environment bindings.
- Add a focused validator derived from the cycle4 validator that loads both configs and proves their shared scientific/training contract, only-one-flag distinction, selector freeze, checkpoint policy and fresh roots.

## Required tests

Add one focused test file (module-level skip on local Windows Torch load failure, but fully runnable on Linux/N16R4) covering: `[B,24,8] -> [B*24,8]`; global pair order and B>1 isolation; all-uniform and mixed-batch byte identity; invalid/padded identity; explicit formula, constant-pair and gap scaling; dtype restoration; finite/nonzero input gradient; exactly one PatchEmbed; OFF/ON state-dict keys/parameter counts equal; real config distinction and selector freeze; dim-0 checkpoint slicing and dim-2 ON rejection with OFF unchanged; K384; model-level metadata reachability; and pre-filter exactly-once remap ordering. Include syntax/config/launcher checks and `git diff --check`.

Do not access data, browser, GPU or Slurm, train/evaluate, or claim efficacy. Do not commit `.cvpr-pro-lab` coordination files. You are not alone in this repository: do not revert unrelated edits.
