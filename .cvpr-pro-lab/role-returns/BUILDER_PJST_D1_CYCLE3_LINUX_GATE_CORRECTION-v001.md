# BUILDER_PJST_D1_CYCLE3_LINUX_GATE_CORRECTION-v001 (RETURN)

status: DONE
evidence_class: STATIC_NO_DATA_IMPLEMENTATION

## Commit binding

- parent (required clean start): cbefa51563adce5c512403695259f2fcb3da16fa
- head (correction commit):  a16a67c4f74ce19de640704c357850c0e7b85ba3
- branch: codex/duca-pjst-cycle3-builder-20260826
- worktree: C:/Users/skywalker/.codex/worktrees/duca-pjst-cycle3-builder-20260826

## Diff (authorized scope only, exactly two files)

```
 opentad/models/utils/temporal_grid.py      | 4 +++-
 tests/test_duca_pjst_d1_derivative_only.py | 9 ++++++---
 2 files changed, 9 insertions(+), 4 deletions(-)
```

1. `opentad/models/utils/temporal_grid.py` — `apply_pjst_derivative_only`:
   `write5 = write[:, None, :, None, None, None]` (6D) ->
   `write5 = write[:, None, :, None, None]` (5D `[Bclips,1,8,1,1]`), which
   broadcasts against the `[Bclips,3,8,H,W]` shape of `y_minus`/`y_plus` in
   `torch.where`.  No in-place indexed writes; formula, float32 valid-pair
   computation, original-dtype restoration, uniform bypass, invalid-pair byte
   identity, and gradient flow are all preserved.
2. `tests/test_duca_pjst_d1_derivative_only.py`:
   - added `import inspect`.
   - `test_single_stage_metas_reach_backbone` and
     `test_single_stage_forward_train_passes_metas`: call
     `nn.Module.__init__(det)` after `SingleStageDetector.__new__` and before
     assigning the `_RecordingBackbone` child module.  Assertions and the
     production `_call_backbone` / `forward_train` paths are unchanged.
   - `test_remap_occurs_exactly_once_before_filtering`: now inspects only
     `SingleStageDetector.post_processing` via `inspect.getsource(...)`
     instead of `src.split("def post_processing", 1)[1]` (which swallowed the
     later `_remap_selector_segments_for_post_processing` helper def and broke
     the exactly-once count).  Still proves exactly one remap call ordered
     before `keep_idxs1` threshold and `batched_nms`.  Production remap code
     was not changed.

Regression retained: the pre-existing tests `test_uniform_mixed_batch_byte_identity`,
`test_invalid_partial_pair_byte_identity`, `test_explicit_formula`,
`test_constant_pair_invariance`, `test_gap_scaling_halves_derivative`,
`test_dtype_restoration`, `test_finite_nonzero_input_gradient` all fail under the
old 6D mask and pass under the 5D broadcast (they were the seven evaluator-listed
failures).

## Commands and results (no data / no GPU / no remote)

- `python -m py_compile opentad/models/utils/temporal_grid.py tests/test_duca_pjst_d1_derivative_only.py`
  -> exit 0 (both files compile).
- `python -m pytest tests/test_duca_pjst_d1_derivative_only.py -q -rs`
  -> `SKIPPED [1] tests\test_duca_pjst_d1_derivative_only.py:25: local Windows
     Torch c10.dll load failure (known); full run on Linux/N16R4` — `1 skipped`.
     This is the documented honest Windows skip; the module is designed to run
     fully on Linux/N16R4 (the evaluator already confirmed Linux executes the
     suite rather than skipping).
- Validator positive/negative cases (driver script, `tools/bata/validate_duca_pjst_d1_derivative_only.py`):
  - POSITIVE correct fixture (epoch 29 + matching sha256): rc=0, `PASS PJST-D1 matched configs`.
  - NEG missing checkpoint file: rc=1.
  - NEG directory (not regular file): rc=1.
  - NEG malformed digest: rc=1.
  - NEG wrong digest: rc=1 (`sha256 mismatch`).
  - NEG wrong epoch (30): rc=1 (`epoch must be exactly 29`).
  - NEG no args: rc=2 (argparse error).
  - Result: ALL_VALIDATOR_CHECKS_OK.
- `bash -n scripts/run_duca_pjst_d1_matched_cycle3_n16r4.sbatch` -> exit 0 (no output).
- `git diff --check` -> exit 0 (no whitespace errors).

## No-efficacy boundary

This is a structural/static implementation correction.  No config, launcher,
validator, selector, detector behavior, checkpoint protocol, data, loss,
evaluator, optimizer, schedule, or scientific design was modified.  No training,
inference, evaluation, or metrics were run, and no efficacy claim is made.
Windows could not execute the Torch-dependent suite (c10.dll), so the Linux/N16R4
focused-suite re-run remains the Evaluator's admission gate.

## Next owner

next_owner: independent Critic final focused recheck
dependency: rerun `python -m pytest tests/test_duca_pjst_d1_derivative_only.py -q`
on Linux/N16R4 against commit a16a67c4... and confirm `30 passed` with zero failures
before the Evaluator re-admits the successor commit.
