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

Across the six detached audit worktrees: `git diff --check`, Python compilation, and shell syntax checks passed for applicable files. Focused tests passed only to the extent stated in `00_IDENTITY.json`; skipped CUDA tests are recorded as blockers, not passes. No remote Slurm task was submitted.
