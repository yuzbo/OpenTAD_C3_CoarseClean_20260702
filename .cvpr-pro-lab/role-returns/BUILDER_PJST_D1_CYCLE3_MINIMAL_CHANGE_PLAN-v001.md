# PJST-D1 Cycle 3 minimal change plan

- Add pure temporal-grid pair metadata helper reusing `exact_uniform_positions`.
- Add pure VideoMAE input transform and wire metadata through detector/backbone/model signatures, preserving OFF identity and checkpointing contracts.
- Move selector physical remap before confidence/top-k processing.
- Add matched Stage-2 OFF/ON configs, real Slurm launcher, focused validator, and focused tests.
- Run no-data static/config/shell checks, commit one clean implementation, and record exact evidence.
