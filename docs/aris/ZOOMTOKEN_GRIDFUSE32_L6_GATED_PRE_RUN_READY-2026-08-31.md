# ZoomToken GridFuse32-L6 gated pre-run receipt

## Frozen identity

- Task: `ZOOMTOKEN-GRIDFUSE32-L6-GATED-v001`
- Base: `2d945e64bdccd09ae2e2916524562e3f388c5a2a`
- Branch: `codex/zoomtoken-gridfuse32-l6-v001`
- Candidate: `3f1e7961720ceb7c7fa4a6276b6767a42adff94c`
- GitHub repository: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- GitHub branch: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-gridfuse32-l6-v001>
- GitHub exact commit: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/3f1e7961720ceb7c7fa4a6276b6767a42adff94c>
- Remote source: `/data/run01/sczc063/yuzibo/projects/zoomtoken_gridfuse32_l6_src_3f1e7961_v3`

The local branch was pushed, freshly fetched, and matched by `git ls-remote` at the exact commit. The remote source is a clean detached checkout at the same commit. Two direct remote HTTPS clone attempts ended with `GnuTLS recv error (-110)`; the final source therefore used the previously clean GitHub-derived `a4e90d8c…` snapshot plus an incremental Git bundle containing the already GitHub-verified `3f1e7961…` commit. Its `origin` remains the repository URL above. No prior source directory was mutated or treated as the new candidate.

## Minimal implementation

The candidate changes only the authorized GridFuse surfaces. Blocks 0--5 remain dense. Blocks 6--11 use deterministic horizontal/vertical per-tubelet mean pairs, execute complete Q/K/V and MLP on N256, broadcast the merged residual update to the two distinct native residual members, restore the N512 carrier, and execute the unchanged dense Adapter. No router, top-k, cross-clip state, teacher, auxiliary loss or new trainable parameter is introduced.

The only post-Critic correction binds G2 to the exact G1 artifact: the G1 terminal receipt records the checkpoint canonical path, SHA256, epoch 59 and `state_dict_ema`; G2 recomputes and asserts all four fields before profiling.

## Verification

- Local `py_compile`, launcher `bash -n`, and `git diff --check`: pass.
- Local pytest is unavailable because the Windows Torch installation fails to load `c10.dll` with `WinError 1114`; this is an evaluator-host environment limitation, not candidate evidence.
- N16R4 exact checkout, independent process: `tests/test_zoomtoken_gridfuse32_l6.py` — `8 passed in 38.57s`.
- N16R4 exact checkout, independent process: `tests/test_zoomtoken_r1_refresh_carry_k32.py` — `12 passed in 40.97s`.
- N16R4 exact checkout, independent process: `tests/test_strict_rectangle_r234.py` — `8 passed in 0.65s`.
- Fresh independent Critic: `PASS`.
- Initial result-blind Evaluator: one blocker, missing G1-to-G2 checkpoint lineage binding.
- Fresh result-blind Evaluator after the minimal correction: `PRE_RUN_READY`; no remaining execution- or evidence-changing blocker.

## Authorization boundary

This receipt establishes implementation and pre-run readiness only. It is not G0 performance evidence. The next action is one Slurm `PRECHECK_ONLY=1` execution of the frozen launcher. Only `PRECHECK_READY` may open exactly one G0 action. G1 and G2 remain conditional on their preceding frozen gates. Any terminal outcome returns to one fresh exact-Project Pro discussion whose prompt must include the repository, branch and exact commit URLs above.
