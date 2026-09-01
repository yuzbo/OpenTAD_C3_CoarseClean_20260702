# CRITIC_PJST_D1_CYCLE3_FOCUSED_REVIEW-v001

## Identity

- reviewed worktree: `C:/Users/skywalker/.codex/worktrees/duca-pjst-cycle3-builder-20260826`
- reviewed commit: `5a4fef786f71f191cd9bb4fe3cb32334a0ba61b5`
- `git status --porcelain=v1`: clean (empty output)
- parent comparison: `a367063f58746a87314e60cedcd7165bf992cc0f..5a4fef786f71f191cd9bb4fe3cb32334a0ba61b5`
- evidence class: `STATIC_NO_DATA_IMPLEMENTATION`

## Findings

### BLOCKING — validator accepts a nonexistent/unbound Stage-1 checkpoint and malformed SHA

The required Stage-1 binding is not fail-closed in the validator. `_resolve()` installs
`/nonexistent/duca_stage1_epoch29.ckpt` and an all-zero SHA as defaults
(`tools/bata/validate_duca_pjst_d1_derivative_only.py:57-61`), and `main()` only copies
CLI values into environment variables without checking that the checkpoint exists/is readable,
that the SHA is hexadecimal, or that the supplied path/hash are actually bound
(`tools/bata/validate_duca_pjst_d1_derivative_only.py:71-83`). Thus the recorded successful
validator invocation can pass while binding no real epoch-29 checkpoint. The launcher likewise
checks only that `DUCA_STAGE1_CHECKPOINT` and SHA are nonempty and that SHA length is 64
(`scripts/run_duca_pjst_d1_matched_cycle3_n16r4.sbatch:47-57`); it does not require `-r` or
validate `[0-9a-fA-F]{64}` before PRECHECK_ONLY can succeed. This violates the accepted
Stage-1 epoch-29 checkpoint binding contract (`PRO_DUCA_PJST_DERIVATIVE_CAUSAL_FREEZE-v002.md:494-496`,
`531-555`, and builder task `BUILDER_PJST_D1_CYCLE3_FOCUSED_CORRECTION-v001.md:62-65`).

Smallest claim-preserving correction: make validator/launcher fail closed unless the supplied
checkpoint is a readable regular file, SHA is exactly 64 hexadecimal characters, and the
resolved config retains the exact supplied path/SHA/epoch 29. Re-run validator and the focused
Linux tests on the corrected clean commit.

### Non-blocking evaluator gate

The focused pytest module did not execute locally because Windows Torch failed to load
`c10.dll`: `python -m pytest tests/test_duca_pjst_d1_derivative_only.py -q` returned
`1 skipped in 1.31s`. This is explicitly reported by the Builder return
(`BUILDER_PJST_D1_CYCLE3_FOCUSED_CORRECTION-v001.md:39-62`), not counted as PASS. It must run
on Linux/N16R4 before PRE_RUN. Static inspection found the production metadata path, global
`[B,24,8]` pairing/packing, byte-preserving transform, checkpoint slicing, ActionFormer
forward reachability, and pre-filter remap ordering implemented at the cited symbols; no
additional deterministic model-code blocker was established from read-only inspection.

## Commands/results

- `git status --porcelain=v1`: PASS, empty.
- `git rev-parse HEAD`: `5a4fef786f71f191cd9bb4fe3cb32334a0ba61b5`.
- `git diff --check a367063f58746a87314e60cedcd7165bf992cc0f 5a4fef786f71f191cd9bb4fe3cb32334a0ba61b5`: PASS.
- `python tools/bata/validate_duca_pjst_d1_derivative_only.py`: printed `PASS PJST-D1 matched configs: sole_distinction=[work_dir, pjst_derivative_only] seed=3407 epochs=60 updates=6000 checkpoint_interval=5 ...`; this pass is insufficient because of the binding defect above.
- `python -m pytest tests/test_duca_pjst_d1_derivative_only.py -q`: `1 skipped in 1.31s` (Torch DLL load; no tests executed).
- `python -m py_compile` on modified implementation/validator files: PASS.
- `bash -n scripts/run_duca_pjst_d1_matched_cycle3_n16r4.sbatch`: PASS.

## Verdict

`BLOCKED_PRE_RUN`

The deterministic validator/launcher checkpoint-binding defect must be corrected. After that,
`next_owner: clean Builder`; then an independent Critic focused recheck must consume the new
clean commit, and Evaluator must run the Linux/N16R4 Torch-dependent focused tests and PRE_RUN.
No efficacy claim is made. No correction is authorized by this review itself.
