# CRITIC_PJST_D1_CYCLE3_LINUX_GATE_RECHECK-v001

status: PJST_D1_FOCUSED_STATIC_PASS
verdict: PASS
evidence_class: READ_ONLY_STATIC_RECHECK_NO_LINUX_EXECUTION

## Commit / cleanliness

- reviewed_commit: `a16a67c4f74ce19de640704c357850c0e7b85ba3`
- required_parent: `cbefa51563adce5c512403695259f2fcb3da16fa`
- worktree: `C:/Users/skywalker/.codex/worktrees/duca-pjst-cycle3-builder-20260826`
- `git rev-parse HEAD`: `a16a67c4f74ce19de640704c357850c0e7b85ba3`
- `git status --porcelain=v1`: empty (clean)
- `git diff --name-only cbefa51563adce5c512403695259f2fcb3da16fa a16a67c4f74ce19de640704c357850c0e7b85ba3`: exactly
  `opentad/models/utils/temporal_grid.py` and `tests/test_duca_pjst_d1_derivative_only.py`.

## Focused evidence

1. `opentad/models/utils/temporal_grid.py:163-201`, `apply_pjst_derivative_only`: input and metadata checks remain `[Bclips,3,16,H,W]` / `[Bclips,8]`; `write = (~exact_uniform_identity)[:, None] & pair_valid` is unchanged; `write5 = write[:, None, :, None, None]` is exactly 5-D `[Bclips,1,8,1,1]` and broadcasts over `[Bclips,3,8,H,W]`. `torch.where` preserves `y_view` for uniform/invalid pairs, while valid irregular pairs use the existing float32 formula and reshape/cast path.
2. `tests/test_duca_pjst_d1_derivative_only.py:293-320`: both `SingleStageDetector.__new__` constructions call `nn.Module.__init__(det)` before assigning `_RecordingBackbone`; tests still invoke production `det._call_backbone(...)` and `det.forward_train(...)`, with metadata identity assertions.
3. `tests/test_duca_pjst_d1_derivative_only.py:365-374`: remap test uses `inspect.getsource(SingleStageDetector.post_processing)`, counts exactly one `_remap_selector_segments_for_post_processing(`, and asserts `remap_line < threshold_line < nms_line` for `keep_idxs1` and `batched_nms`. This scopes inspection to the production method and does not alter production remap code.

## Commands

- `python -m py_compile opentad/models/utils/temporal_grid.py tests/test_duca_pjst_d1_derivative_only.py` -> exit 0.
- `python -m pytest tests/test_duca_pjst_d1_derivative_only.py -q -rs` on this Windows host -> exit 5, `1 skipped`, documented Torch `c10.dll` load failure at test line 25. This is not counted as PASS and does not substitute for Linux/N16R4 execution.
- `git diff --name-only cbefa51563adce5c512403695259f2fcb3da16fa a16a67c4f74ce19de640704c357850c0e7b85ba3` -> exactly two authorized files.

## Boundary / next owner

No efficacy, data, GPU, Slurm, remote, training, inference, evaluation, metric, or checkpoint claim. No code/config/launcher/validator edits were made by Critic. The three prior deterministic Linux blockers are closed in the reviewed snapshot and it is ready for the Evaluator's Linux/N16R4 focused rerun; the Evaluator remains next owner.
