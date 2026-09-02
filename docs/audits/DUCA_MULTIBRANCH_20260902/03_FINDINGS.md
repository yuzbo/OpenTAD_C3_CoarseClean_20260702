# Multi-branch findings

This record distinguishes source-document instructions from observed repository facts. The attached task book defines the acceptance contract; it does not turn skipped CUDA tests, missing checkpoints, or missing remote receipts into passes.

## Identity

- All six requested SHAs are present as local Git commits and were checked out into clean detached audit worktrees under `E:/DeskTop/TAD/_duca_audit_worktrees/`.
- H65 `b419f2b..cfb7041d` has an empty code-tree diff under `opentad configs tools scripts tests`; its implementation receipts are eligible for exact-head review.
- Unified `1c4baebe..89b9ea3e` is non-empty in `tools/bata/write_duca_review_bundle.py`. Therefore implementation receipts from `1c4baebe` are not automatically valid for frozen `89b9ea3e`; fresh exact-head checks are required.
- The supplied Linux evidence attempted a clone and detached worktrees but failed because the environment could not resolve `github.com` and `/mnt/data` was absent. Local object availability was independently established; these bootstrap failures do not prove remote branch state.

## Blocking findings

1. **H65-Pro: P0 repair verified, frozen SHA still closed.** Exact-SHA GPU job `1266268` passed `15` focused tests, while job `1266323` exposed `_call_backbone_forward` passing `masks` to an x-only proof backbone. The signature-aware repair was rerun on remote job `1266408` and passed `15` tests. This only validates the repair worktree; a new exact freeze plus physical-coordinate, strict60/6000 and measured-cost receipts are still required.
2. **DUCA Unified: implementation-blocked.** Taylor P0/P1 is not wired to the real detector objective; H65 original retention/transition is represented by a legacy surrogate; cost is not measured. The correction commit `98d559ee414504caaa480294ce4d066276cdebe6` makes generation and `submit_all.sh` fail closed for `D1`, `F11`, `H0`, `G10`, and `G11` and marks cost blocked.
3. **Evidence: parity-only.** Exact-SHA GPU job `1266269` on `g0024` passed `35` focused tests and the seed-8261 path precheck passed. The required C0 must still compare indices, physical positions, features, logits, losses, decode and final predictions at exact historical H65 identity; no terminal EMA/mAP receipt is present.
4. **CT-DP-BAMoD: geometry-only at frozen SHA.** Exact-SHA GPU job `1266271` on `g0024` passed `7` focused tests, but the frozen G0/G1 factorization did not match the declared G0=Dual-Phase+CT-Tubelet and G1=G0+B-AMoD contract. Correction branch `codex/duca-ctdp-admission-fix-20260902` commit `d62cab763c8e0478e73c6c47a4c185db45164dda` restores that factorization in `duca_ctdp_geometry_g0.py` lines 3-22, `g1.py` lines 3-16, and `g2.py` lines 1-3. It still needs finite-difference, full batch/DDP and complete geometry release receipts.
5. **BAFDR: screen-only, with a diagnosed legacy protocol failure.** Exact-SHA job `1266317` passed the static protocol suite (`11 passed`), but the terminal D160 epoch59 EMA Teacher identity and disjoint selection split receipt are absent. Legacy jobs `1266328-1266330` and `1266402` failed because `LoadFrames` was given unsupported `window_size/window_overlap_ratio` constructor arguments. Correction branch `codex/zoomtoken-bafdr-admission-fix-20260903` (`c750cdae`) removes those arguments, binds the canonical pretrained checkpoint, fixes single-process PRECHECK optimizer handling, and passes the 21-cell validator plus D160 PRECHECK (`train_len=200`, `eval_windows=792`). The official 211-video held-out split remains closed.
6. **ET-TRC: checkpoint/DDP-only, with a diagnosed numerical failure.** Exact-SHA job `1266317` passed `10` protocol tests, but no verified VideoMAE checkpoint coverage, one-GPU load receipt, or real two-GPU global-batch-2 DDP/resume receipt exists. A later non-frozen OFF job `1266185` failed at S1 batch 17 with non-finite `cls_loss/reg_loss/cost`; its paired ON job `1266186` was cancelled. The frozen SHA keeps this fail-closed behavior; its CUDA gate could not be allocated in this check because of `AssocMaxSubmitJobLimit`.

## Scientific invariants applied

All released cells must satisfy the task-book P0 contract: one physical time coordinate through CT/anchors/decode/seconds, `completed_epochs == 60` and `successful_optimizer_updates == 6000` for strict-60 cells, exact optimizer parameter coverage, diagnostic non-interference, Slurm-provided GPU visibility and collision-free rendezvous, checkpoint numel/module coverage, evaluator-rerun bootstrap, held-out discipline, and measured full end-to-end cost.

No non-terminal or invalid cell has a result. In particular, no mAP, speedup, bootstrap interval, or cost claim is recorded by this audit.

Remote queue inspection also found pre-existing BAFDR, ET-TRC, CT-DP and Evidence jobs. Their worktrees resolve respectively to `6ae16954`, `be330c07`, `679b7121` and dirty `647151fa`, not to the frozen SHAs audited here. They are therefore not adopted as results and were not cancelled.

Failure details, log paths, classifications, repairs and revalidation receipts are recorded in `07_REMOTE_CUDA_RECEIPTS/remote_failure_diagnosis_20260903.json`. No failure is silently dropped; non-owned legacy jobs remain diagnostic evidence only.
