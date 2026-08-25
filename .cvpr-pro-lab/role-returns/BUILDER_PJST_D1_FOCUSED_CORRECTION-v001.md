# Builder PJST-D1 focused correction receipt v001

status: IMPLEMENTED_SYNTAX_CHECKED_TEST_BLOCKED_LOCAL_TORCH_DLL
scope: one claim-preserving deterministic correction; no training/evaluation/PRE_RUN

Changes:
- `VisionTransformerAdapter.forward` now accepts `pjst_pair_scale` and `pjst_pair_valid`; normal and temporal-checkpointing wrapper paths pass both keywords.
- `SingleStageDetector` passes selector `metas` to backbone train/test. `BackboneWrapper` accepts bridge metadata, converts positions and dense lengths to int64 tensors, constructs the PJST pair tensors only when explicit `pjst_enabled=True`, and OFF remains disabled by default.
- `build_pjst_pair_metadata` now calls `exact_uniform_positions` exclusively.
- Focused tests cover K384/192 pair layout, partial padding, mixed-batch identity, signature and checkpoint keyword reachability; existing tests remain unchanged.

Verification:
- `python -m py_compile opentad/models/utils/temporal_grid.py opentad/models/backbones/backbone_wrapper.py opentad/models/backbones/vit_adapter.py opentad/models/detectors/single_stage.py tests/test_pjst_d1_contract.py`: PASS.
- `python -m pytest tests/test_pjst_d1_contract.py -q`: BLOCKED during collection by local Windows Torch DLL initialization (`OSError: [WinError 1114] ... Error loading ...\\torch\\lib\\c10.dll or one of its dependencies.`); no PASS claim.
- `git diff --check`: PASS.

No-data/no-training/no-metric attestation: true.
next_owner: coordinator
next_action: inspect this commit and dispatch the independent Critic focused recheck.
