# BUILDER_PJST_D1_CYCLE3_LINUX_GATE_CORRECTION-v001

## Binding and final focused-correction scope

- You are the implementation Builder for the third and final bounded focused correction of the PJST-D1 Cycle-3 package.
- Work only in `C:/Users/skywalker/.codex/worktrees/duca-pjst-cycle3-builder-20260826`.
- Required branch: `codex/duca-pjst-cycle3-builder-20260826`.
- Required clean starting commit: `cbefa51563adce5c512403695259f2fcb3da16fa`.
- Read the N16R4 Evaluator receipt fully: `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/role-returns/EVALUATOR_PJST_D1_CYCLE3_N16R4_PRERUN-v001.md`.
- Do not modify configs, launcher, validator, selector, detector behavior, checkpoint protocol, data, losses, evaluator, optimizer, schedule, or scientific design.
- No browser/data/GPU/Slurm/remote access, training, inference, evaluation, metrics, or claims.

## Exact evidenced failures

On exact N16R4 Linux commit `cbefa515...`, the focused suite executed (not skipped) and returned `20 passed, 10 failed`:

1. **Production transform broadcast bug** at `opentad/models/utils/temporal_grid.py:197`: the pair write mask has six dimensions while `y_minus/y_plus` have five, so `torch.where` fails for seven formula/identity/gradient tests.
2. **Test-only module construction bug** at `tests/test_duca_pjst_d1_derivative_only.py:294,305`: `SingleStageDetector.__new__` is assigned a child module before `torch.nn.Module.__init__`.
3. **Test-only exactly-once counting bug** at `tests/test_duca_pjst_d1_derivative_only.py:371`: the test splits the file at `def post_processing` but includes later helper definitions, so it counts the helper definition plus the actual call. Production `post_processing` has one call at raw segments before filtering/top-k/NMS.

## Authorized correction only

1. In `apply_pjst_derivative_only`, broadcast the boolean pair mask to exactly the `[Bclips,3,8,H,W]` shape of each minus/plus tensor. Preserve the existing formula, float32 valid-pair computation, original-dtype restoration, uniform bypass, invalid-pair byte identity, gradients, and exactly one PatchEmbed contract. Do not use in-place indexed writes that would sever the input gradient.
2. In the two detector reachability tests, initialize the `nn.Module` base correctly before assigning `_RecordingBackbone`; do not weaken assertions or bypass the production `_call_backbone`/`forward_train` paths.
3. Make the remap-order test inspect only `SingleStageDetector.post_processing` (for example `inspect.getsource`) and count the actual `self._remap_selector_segments_for_post_processing(...)` call. It must still prove exactly one call and order before threshold/top-k/NMS. Do not change production remap code to satisfy a faulty source count.
4. Add/retain a regression that would fail with the old six-dimensional mask and prove mixed uniform/irregular, invalid pair, formula, dtype and gradient behavior.
5. Expected diff is only `opentad/models/utils/temporal_grid.py` and `tests/test_duca_pjst_d1_derivative_only.py`. Any broader change is forbidden.

## Verification and return

- Run local `py_compile`, focused pytest (record Windows Torch skip honestly), validator positive/negative cases, `bash -n`, and `git diff --check`.
- Commit the exact bounded correction; end with a clean worktree and do not commit ARIS temp/session files.
- Write `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/role-returns/BUILDER_PJST_D1_CYCLE3_LINUX_GATE_CORRECTION-v001.md` with parent/head/diff, exact commands/results, evidence class `STATIC_NO_DATA_IMPLEMENTATION`, no-efficacy boundary, and `next_owner=independent Critic final focused recheck`.
- Implement, verify and commit in this invocation; do not stop at analysis.
