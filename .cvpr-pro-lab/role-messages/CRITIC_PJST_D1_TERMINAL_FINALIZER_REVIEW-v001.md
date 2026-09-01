# Critic — PJST-D1 terminal-finalizer independent review

Review exact clean commit `45496b8a4355243091eddb1ffc316c435e779bf5` in `C:/Users/skywalker/.codex/worktrees/duca-pjst-cycle4-builder-20260826`, parent `c73e8418de31cdcb2a445ff58a1e33ab9ab6a508`. This is a read-only terminal review; do not edit code, run data/GPU/Slurm, or alter the scientific protocol.

Review the complete diff and the actual production interfaces it calls. Return exactly `PJST_D1_TERMINAL_FINALIZER_STATIC_PASS` or `BLOCKED`.

Verify, with file:line evidence:

1. OFF/ON terminal evaluation loads only each frozen `epoch_59.pth` using `state_dict_ema`, with no fallback/latest/intermediate selection and no training/resume/update.
2. The production `tools/test.py` CLI/config override is valid in this repository and serializes the official per-video prediction format to the path the bootstrap engine consumes. It must evaluate the canonical 211-video validation population with unchanged model, NMS, evaluator, annotation and class map.
3. OFF/ON differ only in frozen arm config/checkpoint/output bindings; no SingleClock, selector, K, data, loss, NMS or evaluator drift is introduced.
4. HEAD/clean tree, arm, checkpoint, state key, output collision and required input checks fail closed before expensive execution.
5. Bootstrap uses OFF as baseline and ON as candidate, exactly 10,000 paired whole-video draws, a fixed nonce/namespace, and the existing official-evaluator recomputation. The 16 shards are disjoint and cover `[0,10000)` exactly once; merge cannot start until every shard succeeds.
6. The merge output preserves the frozen nearest-rank no-interpolation CI convention (one-based ranks 250/9750), and the launchers do not invent thresholds, strata, checkpoint selection or tuning.
7. Literal Slurm command/dependency shapes are runnable on N16R4 and do not hard-code physical GPUs or overwrite `CUDA_VISIBLE_DEVICES`.
8. The focused tests are discriminating enough to catch a wrong checkpoint/state key, missing prediction serialization, incorrect sample/rank/shard contract and premature merge; plainly identify any missing test that corresponds to a reachable failure.

If blocked, list only deterministic defects that would invalidate or prevent the frozen finalizer, grouped into one focused correction. If pass, hand to the existing Evaluator for immediate submission.

- current_scientific_question: whether derivative-only PJST-D1 improves H65 first-mixing representation under frozen matched selection
- next_owner: existing DUCA Evaluator on PASS, focused Builder correction on BLOCKED
- next_action: submit frozen re-inference/bootstrap only on PASS
- dependency: exact clean `45496b8a` diff and production interface review
- expected_return_at: `2026-08-27T11:30:00+08:00`
- single_recovery: one focused Builder correction if and only if a deterministic defect is found

