# Builder — PJST-D1 c73e8418 evaluation-only terminal-finalizer wiring

Work only in the clean Builder worktree `C:/Users/skywalker/.codex/worktrees/duca-pjst-cycle4-builder-20260826`, starting from exact revision `c73e8418de31cdcb2a445ff58a1e33ab9ab6a508`. Verify HEAD and empty porcelain before editing; create a new `codex/` branch if the current branch is not suitable. You are not alone in the repository: preserve unrelated work and never revert another agent's edits.

## Objective

Implement the smallest runnable evaluation-only path needed to close the already completed PJST-D1 OFF/ON experiment. No model, training, data, checkpoint-selection, NMS, evaluator, split, seed, K, or scientific mechanism may change.

The existing jobs/checkpoints are immutable:

- OFF job `1256372`, checkpoint `/data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/off/gpu1_id0/checkpoint/epoch_59.pth`
- ON job `1256373`, checkpoint `/data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/on/gpu1_id0/checkpoint/epoch_59.pth`
- state key `state_dict_ema`, seed `3407`, fixed `K=384`, canonical THUMOS14 211-video validation, unchanged official evaluator/NMS

## Owned surfaces

Prefer additive, PJST-specific files only:

- `scripts/run_duca_pjst_d1_terminal_eval_n16r4.sbatch`
- `scripts/run_duca_pjst_d1_terminal_bootstrap_shard_n16r4.sbatch`
- `scripts/run_duca_pjst_d1_terminal_bootstrap_merge_n16r4.sbatch`
- one focused test file under `tests/` for these launchers

You may minimally reuse or parameterize an existing generic evaluation/bootstrap helper only if additive launchers cannot call its existing CLI. Do not edit `opentad/models/**`, the frozen OFF/ON model configs, training code, loss, post-processing, or the official evaluator core.

## Required behavior

1. The terminal-eval launcher must fail closed on wrong repository HEAD/dirty tree, wrong arm, wrong checkpoint path/epoch/state key, missing canonical annotation/class map/video root, and an existing nonempty output root.
2. It must invoke the production test entrypoint for exactly one arm, load that arm's explicit epoch-59 checkpoint using `state_dict_ema`, enable prediction serialization only, evaluate all 211 canonical validation videos, and write a stable per-video prediction path plus an evaluation sidecar. The sidecar must record arm, commit, config, checkpoint path/SHA/state key/epoch, annotation/class-map/evaluator identities, video/prediction counts, metrics and prediction path.
3. OFF and ON launch shapes must differ only in their frozen arm config/checkpoint/output bindings. No inference-time GT/teacher/cache or alternate checkpoint is permitted.
4. The bootstrap launchers must consume the two serialized prediction paths and the existing `tools/bata/bootstrap_duca_h65_official_map.py` / merge engine. Use OFF as baseline, ON as candidate, exactly 10,000 paired whole-video draws, PCG64 fixed nonce/namespace, and the engine's nearest-rank no-interpolation ranks 250/9750. Use 16 non-overlapping shards plus one merge/finalizer job if that is the existing supported execution path.
5. Produce exact future `sbatch` commands/dependencies for OFF eval, ON eval, 16 bootstrap shards after both evals, and merge after every shard. Builder must not submit them.
6. Add focused tests that distinguish: wrong state key/epoch/arm; missing serialization; output collision; non-211 contract; accidental model/config/evaluator mutation; wrong sample count/ranks; overlapping/missing shard ranges; merge before all shards.

Run focused compile/shell/tests. Commit a clean candidate and return:

- `MINIMAL_CHANGE_PLAN` followed by completed implementation in the same turn
- branch, parent revision, clean commit, exact changed files
- tests/commands/results
- literal N16R4 submission commands and expected artifact paths
- `next_owner=independent Critic`
- `next_action=read-only review of the frozen finalizer wiring`
- `dependency=clean commit and focused tests`
- `expected_return_at=2026-08-27T11:00:00+08:00`
- `single_recovery=one focused correction only for a deterministic finalizer defect`

No browser/Pro/Sources, no training, no data traversal, no GPU/Slurm submission, no metric interpretation, and no Wiki/PAPER_PROGRESS edit.

