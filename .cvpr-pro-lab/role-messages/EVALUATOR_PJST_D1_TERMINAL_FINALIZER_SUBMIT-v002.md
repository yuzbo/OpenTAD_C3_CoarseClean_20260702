# Evaluator — submit static-passed PJST-D1 terminal finalizer

Consume independent PASS `CRITIC_PJST_D1_TERMINAL_FINALIZER_REVIEW-v001` and exact clean commit `45496b8a4355243091eddb1ffc316c435e779bf5`. Reuse your registered evaluation-only identity and workspace; do not register another role.

Deploy a fresh clean remote checkout of exact `45496b8a` under `/data/run01/sczc063/yuzibo/projects/` using the existing bundle/SCP pattern. Prove HEAD and empty porcelain. Do not mutate the completed training roots or checkpoints.

Submit immediately:

1. OFF terminal evaluation from job `1256372`'s explicit `epoch_59.pth`, `state_dict_ema`, into fresh `off/terminal_eval` output.
2. ON terminal evaluation from job `1256373`'s explicit `epoch_59.pth`, `state_dict_ema`, into fresh `on/terminal_eval` output.
3. After both evaluation jobs complete successfully, submit the 16 exact non-overlapping bootstrap shards using the new shard launcher, OFF baseline, ON candidate, fixed frozen nonce/namespace and 10,000 total paired whole-video draws.
4. Submit the merge/finalizer only after all 16 shards complete successfully.

Use N16R4 Slurm, logical `cuda:0`, canonical environment/data paths, and no hard-coded physical GPU. Do not train/resume/update, change any model/config/evaluator/NMS semantics, inspect another checkpoint, tune, add a seed, or launch a new scientific arm.

Before submission run the focused test and `PRECHECK_ONLY`/`sbatch --test-only` equivalent supported by the additive launchers. Stop on a deterministic failure. Unknown submitted state permits one read-only scheduler reconciliation only; never duplicate an accepted job.

Return immediate `START` with:

- exact remote checkout/revision/porcelain
- focused checks and preflight
- OFF/ON evaluation job IDs and output/prediction targets
- bootstrap shard job IDs/array identity and dependency
- merge/finalizer job ID and dependency
- current_scientific_question
- next_owner: Evaluator
- next_action: monitor this exact DAG to terminal and return metrics/CI/gate
- dependency: all exact jobs complete and artifacts validate
- expected_return_at: `2026-08-27T18:00:00+08:00`
- single_recovery: one same-task infrastructure recovery before unknown output state; no duplicate accepted job

Then monitor the exact DAG to terminal and write the durable final receipt required by `EVALUATOR_PJST_D1_CYCLE4_TERMINAL_FINALIZER-v001.md`.

