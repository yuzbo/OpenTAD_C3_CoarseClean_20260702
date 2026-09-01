# BUILDER_PJST_D1_CYCLE2_IMPLEMENTATION-v001

- parent: `b2ccfccab5b4912b59954afcc9b0364955327f7c`
- head: `987f48113784295d80e8edc2bd91ff69ec895756`
- commit: `Implement PJST-D1 derivative-only tubelet transform`
- exact diff: `opentad/models/utils/temporal_grid.py`, `opentad/models/backbones/backbone_wrapper.py`, `opentad/models/backbones/vit_adapter.py`, and matched configs `duca_pjst_d1_matched_off.py` / `duca_pjst_d1_matched_on.py`.
- implementation: pure `[B,24,8]` pair metadata with packed `[B*24,8]` scale/valid tensors; positive int64 delta validation; invalid identity; derivative-only midpoint/difference transform before one PatchEmbed; dim-2 checkpointing fail-closed and dim-0 metadata slicing.
- configs inherit historical Stage-1/Stage-2 contracts and set `pjst_derivative_only` OFF/ON, `single_clock_admission=False`.
- checks: `python -m py_compile opentad/models/utils/temporal_grid.py opentad/models/backbones/backbone_wrapper.py opentad/models/backbones/vit_adapter.py` (PASS).
- no data, browser, GPU, Slurm, training, evaluation, metrics, or efficacy claims.
- evidence class: implementation/syntax only; focused runtime tests and pre-filter remap reorder remain for independent review.
- next_owner: independent Critic
