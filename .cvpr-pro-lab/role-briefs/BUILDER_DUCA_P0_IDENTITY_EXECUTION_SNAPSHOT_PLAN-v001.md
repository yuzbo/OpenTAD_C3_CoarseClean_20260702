# Builder plan — P0 identity-gate execution snapshot

Authority: accepted `PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001`, after the
durable Critic closure
`CRITIC_DUCA_P0_TYPED_FAILURE_INTERFACE_FOCUSED_RECHECK-v001`.

This is the next bounded dependency only: prepare a minimal, clean execution
snapshot plan for the already-reviewed production projector. It must not run
the P0 identity/optimality gate, any Python or test command, data/model access,
metric, CPU/GPU/Slurm work, browser operation, Git push, or scientific-route
change.

The currently registered Builder worktree is at the frozen revision
`63a726a4aaf48ecbf6780bb196de43a890c6b4df` but has unrelated dirty entries.
Preserve them all. Do not reset, stash, clean, commit, alter, or overwrite the
development worktree in this task. The required future formal execution
snapshot must instead be a separately clean, revision-bound workspace carrying
only the already reviewed production paths:

- `opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py`;
- `tests/test_duca_p0_projection_policy.py`.

Return exactly `BUILDER_DUCA_P0_IDENTITY_EXECUTION_SNAPSHOT_PLAN-v001.md` with
the `MINIMAL_CHANGE_PLAN` headings required by the role contract, and state:

1. the existing projector entry point and the smallest planned one-shot
   production-matrix invocation to reuse;
2. how a clean execution snapshot can carry only the two reviewed paths while
   preserving the dirty development workspace;
3. the production-only artifact/receipt interface needed by the Evaluator's
   frozen reference package, without creating it or executing it now;
4. every focused check that must occur later in the clean remote snapshot, all
   explicitly `NOT_EXECUTED` here; and
5. why no second decoder, fallback, tolerance, clipping, deduplication,
   scheduler, framework, configuration, launcher, dataset, metric, or model
   change is needed.

The plan serves the sole current falsifier: a production/reference discrepancy
on a sealed P0 fixture must stop the gate before any scientific performance
claim. If the required execution snapshot would necessarily change the frozen
projection math, route, fixture domain, claim, split, metric, threshold, or
budget, return `NEEDS_ATTENTION` with the exact conflict instead of proposing a
workaround.
