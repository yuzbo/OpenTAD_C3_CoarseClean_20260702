# DUCA H65 Cycle-6 Checkpoint Resume Fix2

## MINIMAL_CHANGE_PLAN

- Change only the checkpoint round-trip test's RNG sequencing.
- Capture Python, NumPy, and Torch CPU RNG state before drawing `expected`.
- Perturb all three generators, restore the captured state, and compare the
  next draws against `expected`.
- Preserve the existing successful-update match/missing/mismatch contract test.

## exact defect

The test drew `expected` before `capture_global_rng_state()`, then restored the
post-draw state and incorrectly required the already-consumed values to be
reproduced. This makes the round-trip assertion fail even when the production
helper restores the captured state correctly.

## changed lines

`tests/test_duca_checkpoint_resume_roundtrip.py`: move the snapshot before the
expected draws and add Python/NumPy/Torch CPU perturbation draws after them.

## verification boundary

Run `python -m py_compile tools/train.py tools/bata/train_lowres_action_probe.py`
and the focused checkpoint-resume test. CUDA RNG is not exercised in this local
test; it remains a GPU PRE_RUN verification boundary.

## next_owner

Critic focused recheck
