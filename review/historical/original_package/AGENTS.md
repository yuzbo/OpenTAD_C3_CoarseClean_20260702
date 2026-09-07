# GeoSparse-TAD: master agent execution contract

Read in order: docs/MODEL_SPEC.zh.md, docs/EXPERIMENT_PROTOCOL.zh.md,
docs/INTEGRATION_CONTRACT.zh.md, your role prompt, and manifests/experiments.jsonl.

## Mandatory outcome

Implement A/B/C and registered baselines in the actual repository; materialize every
registered job as an explicit queue item. Run all ready items within the user's
existing, authorized compute allocation. A negative result is a completed scientific
result, not a reason to cancel the other arms. Do not stop after a design document,
a skeleton module, or a smoke test. Do not invent unavailable data or GPU results.

## Non-serial operation

Use the platform's real parallel delegation capability when available. Assign the
role prompts concurrently, using separate branches/worktrees and file ownership.
Do NOT pretend to spawn agents when the platform cannot do so. A local job dispatcher
is not an agent service. Still implement independent components against frozen
interfaces and register the whole matrix, rather than requesting scientific
permission to proceed after each result.

No train job may depend on another experiment's accuracy, Oracle score, dense
baseline reproduction success, or a teacher checkpoint produced by another job.
The only legal waits are actual input assets, implementation correctness,
checkpoint provenance, and physical resource availability. A failed route must not
block runnable routes. Each evaluation depends only on its own artifacts.

## Integrity

- Main route A is preregistered; B/C are fully implemented challengers, not deferred.
- 60 epochs and seeds 0/1/2 for every registered training configuration.
- No independent dense TAD teacher for proposed methods. Recognition pretraining allowed.
- Dense AdaTAD weights/config/reference must be read, not guessed.
- Sparse heavy execution must shrink QKV/MLP tensors; masks alone are not speedup.
- Native pairing, PE indices, parent-clip attention partitions, and TIA scope are preserved.
- A-full and C-all-fine must be numerically tested against the actual base graph.
- Loss-based best-found or finite-menu diagnostics are not theoretical mAP upper bounds.
- No test-label tuning; do not hide missing assets, OOMs, failed jobs or negative results.
- Never change resolution/epochs/seed/budget silently to fit hardware.
- Do not rent compute, delete data, kill others' jobs, or leak credentials.
- Explicit mocks only for unit tests; no mock-generated scientific metrics.

## Common outputs

Each role commits implementation + unit tests + supported-variant list + a concise
handoff. A runnable component writes its own capability receipt atomically. Every
real run stores immutable code/protocol/split/weight hashes, actual shapes and
metrics. The report displays the entire registered denominator, including BLOCKED.

## Task ownership

00 orchestrator/integration; 01 data+baselines; 02 A; 03 B; 04 C; 05 router+training;
06 geometry+correctness; 07 ROI+budget extensions; 08 evaluation+diagnostics;
09 kernels+hardware; 10 scheduler+reproducibility; 11 paper+statistics.
