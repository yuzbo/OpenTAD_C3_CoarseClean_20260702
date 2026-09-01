# PJST-D1 independent implementation review v001

verdict: PJST_D1_IMPLEMENTATION_BLOCKED
snapshot: 877d893f61b754c76e402fd4be743b9707649845
diff: b2ccfccab5b4912b59954afcc9b0364955327f7c..877d893f61b754c76e402fd4be743b9707649845
role: independent Critic (read-only)

## Deterministic implementation blocker (bounded correction set)

1. **Backbone-to-ViT forward interface is broken.** `BackboneWrapper.forward` passes `pjst_pair_scale=` and `pjst_pair_valid=` to `self.model.backbone` in both normal paths (`opentad/models/backbones/backbone_wrapper.py:135-144,157-165`) and through checkpointing (`:235-243`). The modified `VisionTransformerAdapter.forward` signature has no `pjst_pair_scale` or `pjst_pair_valid` parameters (`opentad/models/backbones/vit_adapter.py:868-877`), so any PJST invocation raises `TypeError: ... unexpected keyword argument 'pjst_pair_scale'` before PatchEmbed. This also means the temporal-checkpointing contract is not executable.

2. **The matched Stage-2 training call chain does not transport metadata or enable PJST.** `SingleStageDetector.forward_train` calls `self.backbone(inputs, masks)` only (`opentad/models/detectors/single_stage.py:90-93`), while `BackboneWrapper.forward` accepts no `metas` and only computes PJST metadata when `irregular_selected_positions` is explicitly supplied (`opentad/models/backbones/backbone_wrapper.py:66-74,99-119`). The new config contains only top-level scalar names (`configs/adatad/thumos/duca_pjst_d1_stage2.py:5-8`) and does not configure the backbone or a metadata bridge. Thus a real matched training exposure cannot reach the PJST branch; if patched only by adding the missing signature, it silently remains OFF. This is one bounded integration correction: make the existing train/test call chain pass the frozen selector metadata to the wrapper and instantiate PJST-D1 in both matched configs without changing selector/RGB/K/optimizer.

3. **Canonical generator contract is violated in the PJST pair helper.** `build_pjst_pair_metadata` independently constructs canonical positions with `torch.div(..., rounding_mode="floor")` (`opentad/models/utils/temporal_grid.py:26-41`) instead of calling the sole frozen `exact_uniform_positions` generator required by the Pro decision. This can disagree on tie cases and makes `exact_uniform_identity` non-authoritative. Replace this local construction with the canonical helper, preserving the global `(16c+2r,16c+2r+1)` pairing.

## Evidence

- Static syntax check passed: `python -m py_compile opentad/models/utils/temporal_grid.py opentad/models/backbones/backbone_wrapper.py opentad/models/backbones/vit_adapter.py opentad/models/detectors/single_stage.py tests/test_pjst_d1_contract.py` (exit 0).
- Only two minimal tests were added (`tests/test_pjst_d1_contract.py:6-19`): a single all-uniform 8-slot case and invalid suffix. They do not cover the required 24x16/global K384 path, exact-uniform byte bypass, mixed batch, padding/partial pairs through PatchEmbed, checkpointing, finite gradients, state/optimizer identity, metadata instantiation, or pre-filter remap trace. The missing coverage is consequential given the interface blockers above.
- Local Torch runtime tests were not treated as scientific evidence; per repository rules, this Windows environment may fail loading `c10.dll`.

## Noted correct portions

- Remap is moved before threshold/top-k/NMS in `opentad/models/detectors/single_stage.py:165-201`, with OFF/ON shared path and a true-time state update at `:331-335`.
- PJST transform uses one cloned input and one `PatchEmbed` call (`opentad/models/backbones/vit_adapter.py:899-910`) and leaves support metadata out of that path.

## Handoff

next_owner: clean Builder
next_action: implement the single bounded integration correction set above in a fresh clean worktree; add focused contract tests for signature/metadata reachability, exact_uniform_positions identity, mixed-batch byte bypass, global K384 24x16 shape, padding, and temporal-checkpointing. Then return a new frozen snapshot for Critic recheck.
scientific_ambiguity: none identified; all blockers are deterministic implementation/interface defects preserving the frozen claim.
