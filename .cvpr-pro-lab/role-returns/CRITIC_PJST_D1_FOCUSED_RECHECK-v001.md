# PJST-D1 focused recheck v001

verdict: PJST_D1_FOCUSED_RECHECK_BLOCKED
snapshot: 843252052cb70460ad4fecf3f002a55566c6d6ff
diff: 877d893f61b754c76e402fd4be743b9707649845..843252052cb70460ad4fecf3f002a55566c6d6ff
role: independent Critic (read-only)

## Deterministic blocker

1. **PJST pair reshape is wrong and makes the production path fail.** The frozen contract is pair metadata `[B,24,8]`, while each packed clip needs `[B*24,8]` (`PRO_DUCA_PJST_DERIVATIVE_CAUSAL_FREEZE-v002.md:240-259`). `BackboneWrapper.forward` constructs `pjst["pair_scale"]` as `[B,192]` and applies `repeat_interleave(coords["actual"].shape[1], dim=0)` (`opentad/models/backbones/backbone_wrapper.py:133-135`), yielding `[B*24,192]`. `VisionTransformerAdapter.forward` rejects any PJST scale shape other than `(x.shape[0],8)` (`opentad/models/backbones/vit_adapter.py:893-900`). Thus a real PJST ON forward reaches a deterministic shape `ValueError`, not the required `[B*24,8]` bridge. This is a second equivalent implementation defect after the prior review's interface/reachability blockers; per assignment, stop the loop and do not propose a third repair.

2. **Temporal checkpointing metadata is not chunk-aligned.** The checkpoint helper slices `actual_positions`, `canonical_positions`, lengths and masks for each `chunk_dim==0` chunk (`opentad/models/backbones/backbone_wrapper.py:265-272`) but passes the full `pjst_pair_scale`/`pjst_pair_valid` closure tensors unchanged (`:246-262`). Even after correcting the pair shape, checkpoint chunks receive metadata for all `B*24` clips rather than the current chunk, violating the temporal-checkpointing execution contract and causing shape/semantic misbinding.

## Additional scope/protocol findings

- The only PJST config added is `configs/adatad/thumos/duca_pjst_d1_stage2.py`, where `model.backbone.custom.pjst_enabled=True` is consumed by `BackboneWrapper.__init__` (`opentad/models/backbones/backbone_wrapper.py:53-57`; config line 10). No matched OFF config is present in the diff, so the frozen two-arm causal estimand cannot be instantiated from this snapshot.
- Metadata reachability is now wired through `SingleStageDetector.forward_train/forward_test(... metas=metas)` (`opentad/models/detectors/single_stage.py:90-91,134-135`) and wrapper metadata extraction (`opentad/models/backbones/backbone_wrapper.py:101-114`), but this does not cure the shape/checkpoint defects above.
- Canonical generation is corrected: `build_pjst_pair_metadata` calls the unique `exact_uniform_positions` (`opentad/models/utils/temporal_grid.py:26-37`). The pair helper preserves global `(16c+2r,16c+2r+1)` pairing (`:24-25`).
- Mixed uniform bypass logic is present (`opentad/models/backbones/vit_adapter.py:901-911`) and the exact-uniform scale identity is tested, but these static checks do not reach a valid `[B*24,8]` runtime bridge.
- Padding/partial-pair handling is present in `build_pjst_pair_metadata` (`opentad/models/utils/temporal_grid.py:24-42`) and tests, but no end-to-end PatchEmbed execution test validates it.
- Production scope violation: commit `84325205` adds `.cvpr-pro-lab/role-returns/BUILDER_PJST_D1_FOCUSED_CORRECTION-v001.md` and `...PLAN-v001.md` (`git diff --name-status 877d893f..84325205`). Role receipts belong in the canonical coordination root, not the production implementation snapshot.
- Prior receipt's note that remap is before filtering/top-k/NMS is accepted as a code fact from `single_stage.py:165-201`; no new contrary finding was required for this recheck.

## Verification boundary

Static diff inspection and focused source-contract inspection only. No Evaluator, PRE_RUN, CUDA job, or experiment was started. Windows Torch `c10.dll` failure remains non-scientific, but the missing executable coverage for the actual `[B*24,8]` path is itself a blocker.

next_owner: Coordinator -> clean Builder
next_action: terminate this correction loop on the second equivalent implementation defect; route only the already-bounded deterministic correction under the accepted Pro decision, with no third Critic repair round.
scientific_ambiguity: none; blockers are deterministic implementation/interface defects preserving the frozen claim.
