# Fixes and tests

## CT-DP-BAMoD correction

Local branch: `codex/duca-ctdp-admission-fix-20260902`
Commit: `d62cab763c8e0478e73c6c47a4c185db45164dda`

The frozen G0/G1 configs disabled the mechanisms that their names claimed. G0 now fixes the Dual-Phase + CT-Tubelet baseline; G1 changes only B-AMoD; G2 changes only the physical-grid head; G3 composes G1 and G2. A factorization test was added at `tests/test_ctdp_matrix_factorization.py` and passed (`1 passed`). CUDA-dependent tests remain blocked by the Windows Torch DLL failure.

## DUCA Unified fail-closed correction

Local branch: `codex/duca-unified-admission-gates-20260902`
Commit: `98d559ee414504caaa480294ce4d066276cdebe6`

`tools/bata/generate_duca_unified_fullmatrix.py` lines 19-35 define implementation blockers; generated rows carry `admission_status` and `admission_blockers`; the manifest records a blocked cost gate; and the generated submitter refuses formal submission while any affected arm or cost gate is blocked. `tests/test_duca_unified_admission_gate.py` passed. Generator output (`41` rows), `--check`, Python compilation, shell syntax validation, and `git diff --check` passed. OpenTAD tests cannot collect on this Windows host because torch fails with WinError 1114.

## Exact-head static checks

Across the six detached audit worktrees: `git diff --check`, Python compilation, and shell syntax checks passed for applicable files. Focused tests passed only to the extent stated in `00_IDENTITY.json`; skipped CUDA tests are recorded as blockers, not passes. No formal training Slurm task was submitted.

## H65 signature-routing admission fix

The exact-SHA P0 proof exposed an x-only backbone compatibility failure. Local commit `78cde6aa` changes `SingleStageDetector._call_backbone_forward` to inspect the signature and pass `masks` only when accepted. The patch was applied remotely as `7f90a48d`; job `1266408` reran the complete P0 list and passed (`15 passed`). The repair is not the frozen SHA and does not itself authorize strict-60 training.

## BAFDR loader and PRECHECK repair

Legacy BAFDR jobs failed during validation-loader construction because generated `LoadFrames` transforms received unsupported `window_size/window_overlap_ratio` arguments. Branch `codex/zoomtoken-bafdr-admission-fix-20260903`, commit `c750cdae1b7dce8ab1ef9b3d2fa04ebb9853926a`, keeps geometry on the dataset, binds the canonical VideoMAE checkpoint through `BAFDR_PRETRAIN`, and lets the documented single-process PRECHECK use a non-DDP model. Remote validation passed for all 21 configs; D160 seed 4407 PRECHECK passed with 200 training videos and 792 evaluation windows. World-size-two CUDA admission and the screen gate remain pending.

## Failure handling

ET-TRC job `1266185` was read from stderr and classified as a non-finite-loss numerical failure (S1 batch 17); paired job `1266186` was cancelled before a terminal checkpoint. CT-DP jobs `1265704-1265705` were read from stderr and classified as old-launcher path errors. These jobs are outside the frozen SHA contract and are not relabeled as current results. The full diagnosis is in `07_REMOTE_CUDA_RECEIPTS/remote_failure_diagnosis_20260903.json`.

The first BAFDR exact-SHA PRECHECK invocation (`j-k0vrhb`) used the wrong non-interactive SSH environment and failed before model execution with a Python syntax error. After explicitly loading the documented CUDA/Miniforge/Conda stack, the exact `c750cdae` worktree passed the 21-cell validator and D160/seed4407 PRECHECK. A non-fast-forward bundle synchronization was also recorded and repaired with a forced refspec; the remote worktree is now exact and clean.
