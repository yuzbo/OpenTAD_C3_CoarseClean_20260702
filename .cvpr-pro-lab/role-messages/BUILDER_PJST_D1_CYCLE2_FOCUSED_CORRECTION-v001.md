# BUILDER_PJST_D1_CYCLE2_FOCUSED_CORRECTION-v001

## Binding

- role: focused Builder correction
- workspace: `C:/Users/skywalker/.codex/worktrees/duca-pjst-cycle2-builder-20260826`
- frozen input candidate: `987f48113784295d80e8edc2bd91ff69ec895756`
- access: writable only in the bound worktree.

## Scope

Perform one claim-preserving focused correction of all deterministic blockers in `CRITIC_PJST_D1_CYCLE2-v001.md`. Do not change the scientific route, selector, selected RGB/K/order/mask, data/split, head/loss/NMS/evaluator, optimizer, 30+60 schedule, or add learnable state.

1. Wire `pjst_derivative_only` through the actual model/backbone config and runtime. OFF must pass no PJST metadata and execute raw PatchEmbed; ON must pass the packed `[B*24,8]` metadata. Verify the `VisionTransformerAdapter.forward` signature accepts the runtime kwargs; do not place PJST arguments on Transformer blocks.
2. Preserve exact-uniform identity per sample/clip in a mixed batch: only nonuniform valid rows may enter float32 midpoint/difference arithmetic. Uniform and invalid/padded rows remain the original tensor bytes. Still call PatchEmbed exactly once.
3. Move the existing selected-axis-to-physical segment remap to immediately after raw per-video segment extraction and before confidence filtering/top-k/NMS, exactly once, without changing scores/labels/thresholds/NMS.
4. Add minimal focused tests for config reachability and arm distinction, `[B,24,8] -> [B*24,8]`, B>1 isolation, mixed uniform byte identity, padding identity, numerical formula, dtype restoration, finite/nonzero gradients, exactly one PatchEmbed, OFF/ON state-key identity, dim-0 checkpoint metadata slicing, dim-2 rejection, and pre-filter exactly-once remap.
5. Add or minimally adapt one executable H65 30+60 OFF/ON PRECHECK/launcher path. It must bind the same Stage-1 terminal checkpoint to both Stage-2 arms, seed 3407, 6000 successful updates, checkpoint every 5 epochs, terminal final/final-EMA rule, official evaluator, and separate result roots. Do not submit it.

Run the focused no-data suite and syntax/config instantiation checks. Commit one clean correction. Write `BUILDER_PJST_D1_CYCLE2_FOCUSED_CORRECTION-v001.md` to the canonical role-return directory with exact commit/diff/commands/results/evidence and next_owner=the same independent Critic for one focused recheck. Do not commit coordination artifacts.

You are not alone in this repository. Do not revert other work. No browser, data, GPU, Slurm, training, evaluation, metrics, or efficacy claims.
