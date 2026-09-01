# BUILDER_PJST_D1_CYCLE3_FOCUSED_CORRECTION-v001

## Role, binding, and evidence boundary

- You are the sole implementation Builder for the first focused correction of the PJST-D1 Cycle-3 candidate.
- Work only in `C:/Users/skywalker/.codex/worktrees/duca-pjst-cycle3-builder-20260826`.
- Required starting branch: `codex/duca-pjst-cycle3-builder-20260826`.
- Required starting commit: `a367063f58746a87314e60cedcd7165bf992cc0f` with a clean worktree.
- The clean scientific parent is `b2ccfccab5b4912b59954afcc9b0364955327f7c`.
- Do not cherry-pick or copy patches from the terminal Cycle-1/Cycle-2 packages. They are negative implementation evidence only.
- Do not access any data, browser, GPU, Slurm scheduler, remote host, training, inference, evaluation, or metric. This task is production code/config/launcher/test implementation plus no-data verification only.
- You are not alone in the repository. Preserve unrelated work and do not revert other people's edits.

Read completely before editing:

1. Worktree `AGENTS.md`, `RTK.md`, `research-wiki/query_pack.md`, and `research-wiki/anti_repetition.md`.
2. Canonical accepted decision: `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/pro-reviews/runs/duca-pjst-derivative-causal-freeze-v002/PRO_DUCA_PJST_DERIVATIVE_CAUSAL_FREEZE-v002.md`.
3. Original Cycle-3 task: `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/role-messages/BUILDER_PJST_D1_CYCLE3-v001.md`.
4. Original Cycle-3 return: `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/role-returns/BUILDER_PJST_D1_CYCLE3_IMPLEMENTATION-v001.md`.

## Confirmed blocking defects in `a367063f...`

These are implementation defects, not a scientific-route change.

1. `SingleStageDetector.forward_train` and `forward_test` still call `self.backbone(inputs, masks)` without `metas`; therefore the ON runtime never receives selector positions.
2. `BackboneWrapper.forward` requires both `metas` and the explicit `irregular_selected_positions` argument when ON, but it never extracts positions, valid length, or selected mask from the real per-sample selector metadata. The ON config therefore fails before the intended transform.
3. Temporal checkpointing is invalid: `_inner_forward` accepts five tensor arguments while `cp.checkpoint` supplies eight; PJST metadata is captured unsliced by the closure. ON must support only chunk dimension 0 and slice all packed metadata at the identical frame boundaries. OFF must preserve base behavior.
4. `apply_pjst_derivative_only` rewrites every pair in an irregular row. Invalid or partial pairs must retain the original bytes; only `(not exact_uniform) AND pair_valid` pairs may be written.
5. The OFF/ON configs set `policy_alpha=0`, which produces the uniform fallback rather than the frozen learned H65 nonuniform acquisition. Both arms must use the same frozen/replayed learned H65 selector exposure and differ only in `custom.pjst_derivative_only`. Selector/detector-to-selector/asformer update routes must be disabled exactly as the accepted decision specifies.
6. No focused tests or validator were added. The launcher is a ten-line shell stub, not the actual N16R4 Stage-2 launcher and cannot establish the frozen checkpoint/config/update/checkpoint-resume contract.
7. The raw-segment selected-to-physical remap was moved before filtering/top-k in `a367063f...`; preserve that correction and prove exactly-once ordering. Do not change its mapping function, thresholds, scores, labels, NMS, evaluator, or output format.

## Required focused correction

Implement the entire runnable path in one bounded correction. Do not stop after a plan or after fixing only the first exception.

### A. Production metadata reachability and OFF identity

- Add the smallest signature-aware call in both `SingleStageDetector` paths so a backbone that accepts `metas` receives the selector-updated `metas`; preserve compatibility with other backbone callables.
- In `BackboneWrapper`, OFF must not construct PJST pair metadata and must not pass PJST-only kwargs to the VideoMAE backbone.
- ON must extract each sample's existing selector positions from `meta["irregular_selected_positions"]`, using only the already documented `meta["pc_ot_mras_bridge"]["selected_positions"]` fallback if needed. Extract/derive the corresponding selected-prefix mask and dense valid length from real existing metadata fields; do not invent a second selection path. Fail closed on missing or inconsistent data.
- The helper's scientific metadata remains global `[B,24,8]`; pack it batch-major to `[B*24,8]` exactly aligned with flattened 24 clips per sample.
- Metadata is detached and creates no parameter, buffer, state-dict key, or selector gradient.

### B. Correct checkpoint execution

- Repair `_inner_forward` so its explicit arguments match `cp.checkpoint` exactly.
- Precompute chunk boundaries once. For ON and `chunk_dim=0`, slice frames, physical coordinates, valid lengths/masks, PJST pair scale/valid, and exact-uniform bypass with the same start/end.
- Reject ON with `chunk_dim=2` before checkpoint execution. Preserve the unmodified OFF checkpoint path for both supported dimensions.
- Do not capture full-batch PJST tensors in a closure used for a mini-batch.

### C. Byte-preserving derivative transform

- Start with `Y=X.clone()` and write only valid irregular pairs.
- Uniform rows must bypass every PJST cast/division before the transform and remain byte-identical.
- Invalid/partial/padded pairs in irregular rows must remain byte-identical; never rewrite them with an ostensibly identity floating calculation.
- Valid irregular pairs alone use float32 `m`, `v`, and `s=delta_can/delta_act`, then cast the two written frames back to the original dtype.
- Apply this immediately before exactly one existing `self.patch_embed(x)[0]` invocation. Do not add a convolution, parameter, buffer, learned gate, clipping, epsilon, or support-weighted appearance.

### D. Frozen matched configs, validator, and real launcher

- Derive two Stage-2 configs from `duca_sampling_rate_curriculum_stage2_joint384.py`. They must resolve identically except for the PJST flag and separate output roots.
- Preserve the learned H65 semantic indirect nonuniform K384 positions while freezing/replaying selector exposure. Do not switch to exact-uniform acquisition. Freeze all selector learning/adaptation routes that could make OFF/ON positions differ.
- Add a focused validator derived from `tools/bata/validate_duca_h65_first_singleclock_cycle4.py`. It must instantiate/resolve both configs and prove: sole PJST distinction, shared selector/acquisition/data/model/loss/evaluator/optimizer/schedule/seed contract, 60 epochs/6000 successful updates, every-five-epoch resumable checkpoints, fixed final/final-EMA rule, clean new roots, and required Stage-1 epoch-29 checkpoint binding.
- Replace the stub launcher by adapting `scripts/run_duca_h65_matched_cycle4_n16r4.sbatch`. Preserve canonical N16R4 module/conda setup, repository and data/annotation/category/pretrain bindings, Slurm-provided GPU visibility, distributed rank/rendezvous variables, real validator/PRECHECK_ONLY flow, distinct OFF/ON modes and roots, and `tools/train.py <config> --seed 3407`. Do not submit it.

### E. Focused regression tests

Add a focused test file, runnable on Linux/N16R4 and allowed to module-skip only for the known local Windows Torch DLL load issue. It must distinguish at least:

- global `[B,24,8]` to packed `[B*24,8]`, correct global pair order, and B>1 isolation;
- exact-uniform mixed-batch byte identity before any PJST arithmetic;
- invalid/partial-pair byte identity inside an irregular row;
- formula, constant-pair, gap scaling, dtype restoration, finite nonzero input gradient;
- exactly one PatchEmbed call and unchanged OFF/ON state keys/parameter count;
- real SingleStage selector metadata reaching the ON backbone in train and test;
- OFF not constructing/passing PJST metadata;
- chunk-dim-0 metadata slicing and chunk-dim-2 ON rejection while OFF remains unchanged;
- K384, config sole-difference/selector freeze, checkpoint policy, validator, launcher syntax;
- raw-segment remap occurs exactly once before threshold/top-k/NMS.

## Required process and return

1. Verify starting HEAD/worktree before editing.
2. Write a concise focused plan to `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/role-returns/BUILDER_PJST_D1_CYCLE3_FOCUSED_CORRECTION_PLAN-v001.md`, then implement immediately in the same invocation.
3. Run focused pytest, config loading/validator, `py_compile`, launcher `bash -n`, and `git diff --check`. If local Windows cannot load Torch, still run every static check locally and make the exact Linux focused command launch-ready; do not pretend it passed.
4. Inspect the final diff for forbidden scientific changes. Commit only repository implementation/config/test/launcher changes on the existing Cycle-3 branch. Worktree must end clean.
5. Write `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/role-returns/BUILDER_PJST_D1_CYCLE3_FOCUSED_CORRECTION-v001.md` with parent/head, exact changed files, commands and verbatim outcomes, remaining limitations, evidence class `STATIC_NO_DATA_IMPLEMENTATION`, and `next_owner=independent Critic`.

No efficacy claim is permitted. The scientific route remains active but PRE_RUN is blocked until an independent Critic passes this frozen commit and an Evaluator subsequently admits it.
