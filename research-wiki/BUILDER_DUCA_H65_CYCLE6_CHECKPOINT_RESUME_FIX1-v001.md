# DUCA H65 Cycle-6 Builder Fix 1 Receipt

## MINIMAL_CHANGE_PLAN

1. Add one production helper that requires the formal checkpoint top-level `successful_optimizer_updates` field to be a strict integer matching the restored audit counter.
2. Invoke that helper from the existing formal resume path after `restore_training_state`; reuse it in the focused checkpoint round-trip test for match, missing, and mismatch cases.
3. Preserve all model, sampling, loss, split, metric, NMS, seed, epoch, launcher, and experiment definitions; run only py_compile and the focused test.

## Changed files and symbols

- `tools/bata/duca_p0_training.py`: `validate_checkpoint_successful_optimizer_updates` enforces strict top-level integer presence and equality with restored audit state; `restore_training_state` remains the production metadata restoration helper.
- `tools/train.py`: formal resume invokes the shared production consistency helper immediately after restoring training state.
- `tests/test_duca_checkpoint_resume_roundtrip.py`: reuses the production helper and covers match, missing, and mismatch while retaining RNG round-trip coverage.

## Verification

- `python -m py_compile tools/train.py opentad/utils/checkpoint.py tools/bata/duca_p0_training.py tests/test_duca_checkpoint_resume_roundtrip.py` — passed.
- `python -m pytest tests/test_duca_checkpoint_resume_roundtrip.py -q` — blocked during collection by local PyTorch DLL initialization failure (`WinError 1114`, `c10.dll`).

No data, GPU, remote, or experiment-definition changes were made.

Status: implemented.

next_owner=independent Critic focused recheck
