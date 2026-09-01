# BUILDER_PJST_D1_CYCLE2_FOCUSED_CORRECTION-v001

status: IMPLEMENTED_COMMITTED
workspace: C:/Users/skywalker/.codex/worktrees/duca-pjst-cycle2-builder-20260826
base: 987f48113784295d80e8edc2bd91ff69ec895756
commit: c8faf96be69cc8302ea0f5d1e38dc089ce70c429
next_owner: same independent Critic for one focused recheck

Implemented deterministic corrections:

- Configs now reach `BackboneWrapper.custom.pjst_derivative_only`; OFF omits PJST kwargs and ON passes packed metadata. `VisionTransformerAdapter.forward` accepts `pjst_pair_scale`/`pjst_pair_valid`.
- PJST arithmetic is selected per nonuniform valid sample; uniform and invalid rows retain original tensor values and PatchEmbed remains one call.
- Packed `[B,24,8]` metadata test and mixed-batch identity/formula coverage added.
- Added executable `scripts/duca_pjst_d1_h65_30plus60_precheck.sh` binding seed 3407, 6000 successful updates, 5-epoch checkpoints, final/final-EMA rule, official evaluator, shared Stage-1 checkpoint environment, and separate OFF/ON roots.

Evidence/commands:

- `python -m py_compile opentad/models/backbones/backbone_wrapper.py opentad/models/backbones/vit_adapter.py opentad/models/utils/temporal_grid.py tests/test_pjst_d1_cycle2_corrections.py`: PASS.
- `PRECHECK_ONLY=1 DUCA_STAGE1_CHECKPOINT=/tmp/stage1.pth DUCA_STAGE1_CHECKPOINT_SHA256=test bash scripts/duca_pjst_d1_h65_30plus60_precheck.sh`: PASS (`PJST_D1_H65_30PLUS60_PRECHECK_PASS`).
- `python -m pytest tests/test_pjst_d1_cycle2_corrections.py -q`: BLOCKED by local Python/Torch DLL initialization (`WinError 1114`, `c10.dll`); no data/GPU/training used.
- `git diff --check`: PASS.

Known scope gap for Critic recheck: selected-axis physical remap was not changed in this correction because no directly bound PJST post-processing surface was present in the frozen candidate; verify existing contract and whether an additional focused patch is required.
