# PJST-D1 cycle 2 minimal change plan

- Add a pure metadata helper for `[B,24,8]` global tubelet-pair deltas/scales/validity with exact-uniform short-circuit semantics.
- Wire detached metadata through BackboneWrapper and VideoMAE, transforming packed inputs before exactly one PatchEmbed; reject temporal chunk checkpointing and slice dim-0 metadata identically.
- Move selected-axis remap before filtering/top-k while preserving mapping/NMS behavior.
- Add matched 30+60 OFF/ON configs and focused no-data tests for shape, identity, formula, checkpointing, gradients, state-dict, and remap ordering.
- Run focused tests and syntax checks, commit only allowed implementation surfaces, and return evidence to independent Critic.
