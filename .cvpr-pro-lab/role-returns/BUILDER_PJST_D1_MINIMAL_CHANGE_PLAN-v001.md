# PJST-D1 Builder Minimal Change Plan v001

Scope is limited to the frozen derivative-only contract. Before code changes:

1. Add a pure temporal-grid helper that validates int64 global pair positions `(16c+2r,16c+2r+1)`, computes `delta_can/delta_act`, `s`, `pair_valid`, and audit support without selector changes.
2. Thread detached pair metadata through the backbone wrapper into ViT adapter; construct `Y` immediately before the existing PatchEmbed using `m=(x-+x+)/2`, `v=s*(x+-x-)/2`, `y-=m-v`, `y+=m+v`. Preserve exact-uniform rows by returning the original input unchanged and call PatchEmbed exactly once.
3. Reorder only SingleStage post-processing remap so selected-axis proposals are remapped immediately after raw proposals/scores extraction and before threshold/top-k/NMS; leave mapping/NMS/output semantics unchanged.
4. Add one PJST-D1 config inheriting H65 Stage-2 and focused contract tests for algebra, global pairing, uniform identity, padding, and no-new-parameter behavior.
5. Run only focused tests and syntax checks; no training, PRE_RUN, evaluator, or metric execution.

Expected files: temporal_grid.py, backbone_wrapper.py, vit_adapter.py, optional single_stage.py, one config, focused tests, and terminal receipt. No selector, ASFormer, RGB/rank/K384, adapter blocks, loss/optimizer/schedule/data/NMS changes.
