# BUILDER_PJST_D1_CYCLE3_CHECKPOINT_BINDING_CORRECTION-v001

## Binding and scope

- You are the same implementation Builder performing the second bounded, claim-preserving correction of the clean PJST-D1 Cycle-3 package.
- Work only in `C:/Users/skywalker/.codex/worktrees/duca-pjst-cycle3-builder-20260826`.
- Required branch: `codex/duca-pjst-cycle3-builder-20260826`.
- Required clean starting commit: `5a4fef786f71f191cd9bb4fe3cb32334a0ba61b5`.
- Read the independent review completely: `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/role-returns/CRITIC_PJST_D1_CYCLE3_FOCUSED_REVIEW-v001.md`.
- Do not modify model, selector, detector, data, loss, optimizer, schedule, evaluator, scientific mechanism, or experiment matrix.
- No data/browser/GPU/Slurm/remote access, training, inference, evaluation, metric, or claim.

## Exact deterministic defect

The current validator and launcher can report a successful Stage-1 binding while the checkpoint does not exist and the digest is malformed:

- `tools/bata/validate_duca_pjst_d1_derivative_only.py` supplies `/nonexistent/duca_stage1_epoch29.ckpt` plus an all-zero digest by default and never proves a real readable regular file or a valid supplied digest.
- `scripts/run_duca_pjst_d1_matched_cycle3_n16r4.sbatch` checks only nonempty values and digest length.

This is the only authorized correction.

## Required fix

1. Validator CLI must require an explicit Stage-1 checkpoint path and explicit SHA-256 for any admission-capable validation. No fabricated fallback path or all-zero digest may create a PASS.
2. Before config admission, require the checkpoint to be a readable regular file and the supplied digest to match exactly `^[0-9a-fA-F]{64}$`.
3. Compute the file's actual SHA-256 using a streaming read in the validator and require case-insensitive equality with the supplied digest. A merely well-formed but incorrect digest must fail.
4. Require epoch exactly 29 and verify both resolved configs retain the exact supplied checkpoint path, digest, and epoch without coercion or override.
5. The launcher must independently fail closed before validator/PRECHECK/training unless the checkpoint is a readable regular file, epoch is exactly 29, and SHA matches the same 64-hex rule. It must call the validator with the explicit path/digest/epoch. Do not calculate or replace the user's declared digest in the launcher; the validator performs authoritative content equality.
6. Extend focused tests to distinguish: missing file; unreadable/non-regular path where practical; malformed digest; well-formed wrong digest; correct small fixture file/digest; config retention; launcher precheck fail-closed. Do not weaken the existing model-path tests.
7. Preserve all other files byte-for-byte unless a focused test file requires the new assertions. The expected production diff should be limited to validator, launcher, and focused tests.

## Verification and return

- Run `py_compile`, validator negative cases and one positive temporary-file case, focused pytest (record a Windows Torch module skip honestly), `bash -n`, and `git diff --check`.
- Commit the bounded correction and end with a clean worktree. Do not commit ARIS session/temp files.
- Write `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/role-returns/BUILDER_PJST_D1_CYCLE3_CHECKPOINT_BINDING_CORRECTION-v001.md` with parent/head/diff, exact commands/results, evidence class `STATIC_NO_DATA_IMPLEMENTATION`, no-efficacy boundary, and `next_owner=the same independent Critic focused recheck`.
- Do not stop at analysis; implement, verify, commit, and return in this invocation.
