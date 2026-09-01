# BUILDER_PJST_D1_CYCLE3_FOCUSED_CORRECTION_PLAN-v001

## Status

- role: sole implementation Builder (focused correction)
- worktree: `C:/Users/skywalker/.codex/worktrees/duca-pjst-cycle3-builder-20260826`
- branch: `codex/duca-pjst-cycle3-builder-20260826`
- starting HEAD: `a367063f58746a87314e60cedcd7165bf992cc0f`
- clean scientific parent: `b2ccfccab5b4912b59954afcc9b0364955327f7c`
- evidence class: `STATIC_NO_DATA_IMPLEMENTATION` (no data/browser/GPU/Slurm/training/eval/metrics)

## Confirmed defects (from the focused-correction role message)

1. `SingleStageDetector.forward_train/forward_test` call `self.backbone(inputs, masks)` without `metas`; the ActionFormer detector (which overrides both paths and is the real runtime) also drops `metas`/`masks` in `_forward_backbone_with_single_clock`.
2. `BackboneWrapper.forward` requires `metas` *and* the explicit `irregular_selected_positions` argument when ON, but never extracts positions/valid-length/prefix-mask from per-sample selector metadata.
3. `_inner_forward` accepts 5 tensor args while `cp.checkpoint` supplies 8; PJST metadata is captured unsliced by the closure.
4. `apply_pjst_derivative_only` rewrites an entire irregular clip row; only `(not exact_uniform) AND pair_valid` pairs may be written, and non-written bytes must survive.
5. OFF/ON configs set `policy_alpha=0` (uniform fallback) instead of the frozen learned H65 nonuniform acquisition; selector update routes must be frozen.
6. No focused tests/validator; the launcher is a 9-line stub.
7. The pre-filter remap reorder (already in `a367063f`) must be preserved and proven exactly-once.

## Correction plan (bounded, one pass)

### A. Metadata reachability + OFF identity
- Add a signature-aware `_call_backbone(inputs, masks, metas)` helper on `SingleStageDetector` (reuses the existing `inspect`-based `_callable_accepts_metas`); use it in `SingleStageDetector.forward_train`/`forward_test`.
- In `ActionFormer._forward_backbone_with_single_clock`, route the non-admission path through the same helper (passes `metas` only; preserves the existing no-`masks` OFF call so OFF stays byte-identical).
- In `BackboneWrapper.forward`, when `custom.pjst_derivative_only` is ON, extract per-sample `irregular_selected_positions` (with the documented `pc_ot_mras_bridge.selected_positions` fallback), dense valid length, and selected-prefix mask from `metas` (via a pure helper in `temporal_grid.py`); OFF constructs none of it and passes no PJST kwargs.

### B. Correct checkpoint execution
- Make `_inner_forward`'s explicit args match `cp.checkpoint` exactly (8 tensor args).
- Precompute chunk boundaries once; for ON with `chunk_dim=0`, slice frames + all packed PJST metadata at the same start/end; reject ON with `chunk_dim=2` before checkpointing; keep the unmodified OFF path for both dims.

### C. Byte-preserving derivative transform
- `Y = X.clone()`; write only `(not exact_uniform) AND pair_valid` pairs in float32 `m`,`v`,`s=delta_can/delta_act`, casting the two written frames back to the original dtype; uniform rows and invalid/partial pairs stay byte-identical; exactly one `self.patch_embed(x)[0]` call; no conv/param/buffer/gate/clip/eps/support term.

### D. Frozen matched configs, validator, real launcher
- Rewrite the two Stage-2 configs so the *only* differences are `model.backbone.custom.pjst_derivative_only` and `work_dir`; freeze selector routes (`policy_alpha` fixed 1.0 learned, `detector_gradient`/`detector_contribution`/`asformer_adapt` all 0); keep the shared H65 stage-1 checkpoint/epoch-29 binding and official-60 60-epoch/6000-update contract.
- Add `tools/bata/validate_duca_pjst_d1_derivative_only.py` (derived from the cycle4 validator) proving: sole-PJST-distinction, shared selector/acquisition/data/model/loss/evaluator/optimizer/schedule/seed, 60/6000, every-5 resumable checkpoints, final/final-EMA, fresh distinct roots, stage-1 epoch-29 binding.
- Replace the stub with `scripts/run_duca_pjst_d1_matched_cycle3_n16r4.sbatch` adapted from the canonical `run_duca_h65_matched_cycle4_n16r4.sbatch` (module/conda, data/annotation/category/pretrain bindings, GPU visibility, distributed env, real validator + PRECHECK_ONLY flow, distinct OFF/ON modes/roots, `tools/train.py <cfg> --seed 3407`); never submitted.

### E. Focused regression tests
- Add `tests/test_duca_pjst_d1_derivative_only.py` (module-level skip only on the local Windows Torch c10.dll load failure) covering all required bullets: `[B,24,8]->[B*24,8]`, global pair order + B>1 isolation, uniform/mixed-batch byte identity, invalid/partial-pair byte identity, formula/constant-pair/gap-scaling/dtype-restoration/finite-nonzero-input-gradient, single PatchEmbed call + unchanged state keys/param count, selector metadata reaching ON backbone in train+test, OFF not constructing/passing PJST metadata, chunk-dim-0 slicing + chunk-dim-2 ON rejection with OFF unchanged, K384, config sole-difference/selector-freeze/checkpoint policy, validator, launcher syntax, pre-filter exactly-once remap.

## Verification (no-data)

- `python -m py_compile` on all modified python + configs (configs via `py_compile` only when standalone-compilable; otherwise validated by the validator's `Config.fromfile`).
- `python -m pytest tests/test_duca_pjst_d1_derivative_only.py -q` (expected: local Windows torch module-skip, verbatim recorded; Linux command launch-ready).
- Validator run (config resolution path) recorded verbatim.
- `bash -n scripts/run_duca_pjst_d1_matched_cycle3_n16r4.sbatch`.
- `git diff --check`.

## Commit & returns

- Commit only repository implementation/config/test/launcher changes on the existing branch; worktree ends clean (no `.cvpr-pro-lab` coordination files committed).
- Write both durable returns to `.cvpr-pro-lab/role-returns/` on `E:`.
