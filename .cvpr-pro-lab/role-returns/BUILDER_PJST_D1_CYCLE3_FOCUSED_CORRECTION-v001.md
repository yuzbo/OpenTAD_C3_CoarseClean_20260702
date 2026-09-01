# BUILDER_PJST_D1_CYCLE3_FOCUSED_CORRECTION-v001

## Identity

- role: sole implementation Builder (focused correction)
- worktree: `C:/Users/skywalker/.codex/worktrees/duca-pjst-cycle3-builder-20260826`
- branch: `codex/duca-pjst-cycle3-builder-20260826`
- parent: `a367063f58746a87314e60cedcd7165bf992cc0f` (the Cycle-3 package that carried the 7 confirmed defects)
- clean scientific parent (unmodified reference): `b2ccfccab5b4912b59954afcc9b0364955327f7c`
- head: `5a4fef786f71f191cd9bb4fe3cb32334a0ba61b5`
- evidence class: `STATIC_NO_DATA_IMPLEMENTATION`
- next_owner: independent Critic

No data, browser, GPU, Slurm scheduler, remote host, training, inference, evaluation, or metric was accessed. No efficacy claim is made.

## Scope of the correction (the 7 confirmed defects)

1. `SingleStageDetector.forward_train/forward_test` called `self.backbone(inputs, masks)` without `metas`; the ActionFormer detector (the real runtime, which overrides both paths) also dropped `metas`/`masks`.
2. `BackboneWrapper.forward` required `metas` *and* the explicit `irregular_selected_positions` argument when ON, but never extracted positions/valid-length/prefix-mask from per-sample selector metadata.
3. `_inner_forward` accepted 5 tensor args while `cp.checkpoint` supplied 8; PJST metadata was captured unsliced.
4. `apply_pjst_derivative_only` rewrote an entire irregular clip row instead of only `(not exact_uniform) AND pair_valid` pairs.
5. OFF/ON configs set `policy_alpha=0` (uniform fallback) instead of the frozen learned H65 nonuniform acquisition.
6. No focused tests/validator; the launcher was a 9-line stub.
7. The pre-filter remap reorder (already present in `a367063f`) was preserved and is proven exactly-once.

## Changed files (exact)

- `opentad/models/utils/temporal_grid.py` (M) — `pjst_selector_positions_from_metas`, `pjst_pair_metadata` (global `[B,24,8]`), `pack_pjst_pair_metadata`, byte-preserving/differentiable `apply_pjst_derivative_only`.
- `opentad/models/backbones/backbone_wrapper.py` (M) — ON extracts positions from metas; OFF passes no PJST kwargs; `_backbone_forward` helper; `temporal_checkpointing` with correct `_inner_forward` signature, precomputed chunk boundaries, chunk-dim-0 slicing, chunk-dim-2 ON rejection.
- `opentad/models/detectors/single_stage.py` (M) — signature-aware `_call_backbone` used in `forward_train`/`forward_test` (pre-filter remap reorder untouched).
- `opentad/models/detectors/actionformer.py` (M) — `_forward_backbone_with_single_clock` non-admission path forwards `metas`.
- `configs/adatad/thumos/duca_pjst_d1_stage2_off.py` (M), `configs/adatad/thumos/duca_pjst_d1_stage2_on.py` (M) — `policy_alpha` pinned to learned 1.0, selector adaptation routes pinned 0; checkpoint policy fields added; only `work_dir` + `pjst_derivative_only` differ.
- `tools/bata/validate_duca_pjst_d1_derivative_only.py` (A) — read-only, torch-free config validator.
- `scripts/run_duca_pjst_d1_matched_cycle3_n16r4.sbatch` (M) — real N16R4 launcher adapted from `run_duca_h65_matched_cycle4_n16r4.sbatch`.
- `tests/test_duca_pjst_d1_derivative_only.py` (A) — focused regression tests.

`vit_adapter.py` was not modified (its `forward` already calls `apply_pjst_derivative_only` exactly once before `self.patch_embed(x)[0]`, and accepts the PJST kwargs). The pre-filter remap mapping function, thresholds, scores, labels, NMS, evaluator, and output format were not changed.

## Verification — verbatim outcomes

All static/no-data checks were run locally. Local Windows Torch cannot load `c10.dll` (WinError 1114), so the Torch-dependent pytest module-skips (recorded verbatim below) and the full focused suite is launch-ready for Linux/N16R4.

1. `python -m py_compile` on all modified Python + config files: **PASS** (no output, exit 0).
2. `python tools/bata/validate_duca_pjst_d1_derivative_only.py`:
   ```
   PASS PJST-D1 matched configs: sole_distinction=[work_dir, pjst_derivative_only] seed=3407 epochs=60 updates=6000 checkpoint_interval=5 roots=[exps/thumos/adatad/duca_pjst_d1_matched_off, exps/thumos/adatad/duca_pjst_d1_matched_on]
   ```
3. `python -m pytest tests/test_duca_pjst_d1_derivative_only.py -q`:
   ```
   1 skipped in 1.21s
   ```
   (exit code 5 = "no tests collected" because the whole module skipped on the local Windows Torch `c10.dll` load failure; this is the documented, allowed module-skip. The exact Linux command is `python -m pytest tests/test_duca_pjst_d1_derivative_only.py -q`, where all tests run.)
4. `bash -n scripts/run_duca_pjst_d1_matched_cycle3_n16r4.sbatch`: **PASS** (no output, exit 0).
5. `git diff --check`: **PASS** (no output, exit 0).

The Torch-dependent tests were NOT executed locally (Torch does not import on this Windows host). The static/config/launcher/source-order checks were executed and passed. This is not a claim that the Torch-dependent tests passed.

## Remaining limitations (for the Critic)

- Full pytest execution requires the Linux/N16R4 OpenTAD environment (Torch); it was not run here, and no test result is fabricated.
- No real model instantiation (`build_detector`) was performed (torch-free validator resolves configs only); the Critic/Evaluator must still establish the frozen Stage-1 epoch-29 checkpoint, selected RGB, and exposure identity ledger on the real environment before PRE_RUN.
- The scientific route remains active but `PRE_RUN` is blocked until an independent Critic passes this frozen commit `5a4fef786f71f191cd9bb4fe3cb32334a0ba61b5` and an Evaluator subsequently admits it.

## Return documents

- Plan: `.cvpr-pro-lab/role-returns/BUILDER_PJST_D1_CYCLE3_FOCUSED_CORRECTION_PLAN-v001.md`
- This return: `.cvpr-pro-lab/role-returns/BUILDER_PJST_D1_CYCLE3_FOCUSED_CORRECTION-v001.md`
