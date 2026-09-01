# Critic — focused review of one-allocation PJST-D1 terminal bootstrap

Independently review exact clean commit `4204937a933c7a48854b623efefc7fd662e98805` in the existing read-only Critic workspace. Compare it with parent `45496b8a4355243091eddb1ffc316c435e779bf5`, the accepted finalizer static review, the frozen PJST-D1 Pro contract, and the Builder receipt `BUILDER_PJST_D1_TERMINAL_BOOTSTRAP_BUNDLE-v002.md`.

Scope is only the scheduler-transport correction introduced because the 16-element array was rejected before acceptance by `AssocMaxSubmitJobLimit`. Verify:

1. one Slurm allocation runs exactly 16 non-overlapping shard ranges covering `[0,10000)` once;
2. OFF remains baseline and ON candidate; nonce/namespace, canonical inputs, PCG64 bootstrap engine, official evaluator, 10,000 draws, and nearest-rank `250/9750` convention are unchanged;
3. CPU/process parallelism is bounded by the allocation and cannot silently oversubscribe or skip a shard;
4. every shard exit status is propagated fail-closed, and merge cannot run unless all 16 exact shard artifacts exist and succeeded;
5. commit/clean-tree/input/output-collision identity remains fail-closed;
6. no model, config, checkpoint, prediction, evaluator-core, metric, seed, split, gate, threshold, or science changed.

Do not execute inference/bootstrap, access data, submit Slurm, edit code, or broaden the review. Return exactly `PJST_D1_TERMINAL_BOOTSTRAP_BUNDLE_STATIC_PASS` or `BLOCKED`, with one concrete deterministic defect if blocked. Include `next_owner`, `next_action`, `dependency`, `expected_return_at`, and `single_recovery`.

