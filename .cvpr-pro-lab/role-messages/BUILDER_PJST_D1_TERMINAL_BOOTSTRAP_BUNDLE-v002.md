# Builder — bundle PJST-D1 terminal bootstrap into one Slurm allocation

Consume the accepted static PASS for clean commit `45496b8a4355243091eddb1ffc316c435e779bf5` and the Evaluator START in which terminal re-inference jobs `1257240` (OFF) and `1257241` (ON) were accepted, while the 16-element bootstrap array was rejected by `AssocMaxSubmitJobLimit` before any bootstrap job was accepted.

This is one bounded scheduler-transport correction, not a scientific or evaluator change. In the existing clean Builder worktree and branch, add the smallest fail-closed launcher that runs the already frozen 16 non-overlapping bootstrap shards inside one Slurm allocation and merges only after all 16 succeed. Preserve exactly:

- OFF baseline / ON candidate prediction inputs produced by the accepted terminal-evaluation jobs;
- 10,000 paired whole-video draws, PCG64 engine, fixed nonce/namespace, shard ranges covering `[0,10000)` exactly once, and nearest-rank `250/9750` convention;
- official evaluator, canonical 211-video population, output-collision checks, exact clean commit and output identity;
- no model, config, training, checkpoint, prediction, evaluator-core, metric, gate, threshold, seed, split, or scientific-contract change.

Use one job allocation with bounded CPU parallelism; do not submit jobs. A shard failure must fail the allocation and prevent merge. Add a focused test that distinguishes: exact 16-shard coverage, bounded worker/process launch, wait/return-code propagation, and merge only after all shards exist and succeed. Run only the focused test, shell syntax checks, `git diff --check`, and prove the worktree clean after one commit.

Return `MATERIAL_READY` or `NEEDS_ATTENTION` with parent/clean commit, exact changed files, checks, one-job submission shape and artifacts, plus:

- `current_scientific_question`: can derivative-only PJST-D1 improve H65 first-mixing representation under the frozen matched selector and official evaluation contract?
- `next_owner`: same independent Critic
- `next_action`: one focused static review of only the bundled scheduler transport
- `dependency`: exact clean commit and checks
- `expected_return_at`: `2026-08-27T10:30:00+08:00`
- `single_recovery`: none beyond this scheduler-only correction

Do not change or reinterpret the already negative point estimates; no efficacy claim is authorized.

