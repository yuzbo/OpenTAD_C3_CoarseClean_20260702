# Multi-branch findings

This record distinguishes source-document instructions from observed repository facts. The attached task book defines the acceptance contract; it does not turn skipped CUDA tests, missing checkpoints, or missing remote receipts into passes.

## Identity

- All six requested SHAs are present as local Git commits and were checked out into clean detached audit worktrees under `E:/DeskTop/TAD/_duca_audit_worktrees/`.
- H65 `b419f2b..cfb7041d` has an empty code-tree diff under `opentad configs tools scripts tests`; its implementation receipts are eligible for exact-head review.
- Unified `1c4baebe..89b9ea3e` is non-empty in `tools/bata/write_duca_review_bundle.py`. Therefore implementation receipts from `1c4baebe` are not automatically valid for frozen `89b9ea3e`; fresh exact-head checks are required.
- The supplied Linux evidence attempted a clone and detached worktrees but failed because the environment could not resolve `github.com` and `/mnt/data` was absent. Local object availability was independently established; these bootstrap failures do not prove remote branch state.

## Blocking findings

1. **H65-Pro: conditional only and currently P0-blocked.** Exact-SHA GPU job `1266268` on `g0024` passed `15` focused tests, but job `1266323` found a real failure in `test_duca_official_optimizer_coverage.py`: `_call_backbone_forward` passed `masks` to an x-only proof backbone. Local fix `78cde6aa` and remote patch commit `7f90a48d` implement signature-aware routing, but the rerun was blocked by `AssocMaxSubmitJobLimit`; the fix is therefore unverified. No physical-coordinate reduction receipt, strict60/6000 terminal receipt, Slurm concurrency, or measured end-to-end cost receipt exists.
2. **DUCA Unified: implementation-blocked.** Taylor P0/P1 is not wired to the real detector objective; H65 original retention/transition is represented by a legacy surrogate; cost is not measured. The correction commit `98d559ee414504caaa480294ce4d066276cdebe6` makes generation and `submit_all.sh` fail closed for `D1`, `F11`, `H0`, `G10`, and `G11` and marks cost blocked.
3. **Evidence: parity-only.** Exact-SHA GPU job `1266269` on `g0024` passed `35` focused tests and the seed-8261 path precheck passed. The required C0 must still compare indices, physical positions, features, logits, losses, decode and final predictions at exact historical H65 identity; no terminal EMA/mAP receipt is present.
4. **CT-DP-BAMoD: geometry-only at frozen SHA.** Exact-SHA GPU job `1266271` on `g0024` passed `7` focused tests, but the frozen G0/G1 factorization did not match the declared G0=Dual-Phase+CT-Tubelet and G1=G0+B-AMoD contract. Correction branch `codex/duca-ctdp-admission-fix-20260902` commit `d62cab763c8e0478e73c6c47a4c185db45164dda` restores that factorization in `duca_ctdp_geometry_g0.py` lines 3-22, `g1.py` lines 3-16, and `g2.py` lines 1-3. It still needs finite-difference, full batch/DDP and complete geometry release receipts.
5. **BAFDR: screen-only.** Exact-SHA job `1266317` passed the static protocol suite (`11 passed`), but the terminal D160 epoch59 EMA Teacher identity and disjoint selection split receipt are absent. The official 211-video held-out split must not open the 21-cell matrix.
6. **ET-TRC: checkpoint/DDP-only.** Exact-SHA job `1266317` passed `10` protocol tests, but no verified VideoMAE checkpoint coverage, one-GPU load receipt, or real two-GPU global-batch-2 DDP/resume receipt exists.

## Scientific invariants applied

All released cells must satisfy the task-book P0 contract: one physical time coordinate through CT/anchors/decode/seconds, `completed_epochs == 60` and `successful_optimizer_updates == 6000` for strict-60 cells, exact optimizer parameter coverage, diagnostic non-interference, Slurm-provided GPU visibility and collision-free rendezvous, checkpoint numel/module coverage, evaluator-rerun bootstrap, held-out discipline, and measured full end-to-end cost.

No non-terminal or invalid cell has a result. In particular, no mAP, speedup, bootstrap interval, or cost claim is recorded by this audit.
