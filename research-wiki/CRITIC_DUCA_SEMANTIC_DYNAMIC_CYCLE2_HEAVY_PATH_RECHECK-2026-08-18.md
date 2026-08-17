# DUCA semantic cycle2 heavy-path focused critic recheck (2026-08-18)

## Verdict

PASS — the focused heavy-path defect is closed at the frozen candidate. This verdict covers only dynamic heavy-path K handling; six-arm, split, and checkpoint issues remain independent follow-up work.

## Evidence

- `opentad/models/detectors/actionformer.py:386-424`, `_forward_backbone_dynamic_k`: computes per-sample valid lengths from the prefix mask, groups rows by stable sorted K buckets (`sorted(set(...))`), slices the temporal input with `narrow(time_dim, 0, bucket_k)` before calling `self.backbone(bucket_inputs, bucket_masks)`, scatters each bucket output back to original batch indices, then pads only the returned feature tensors to the maximum returned feature length and constructs a matching prefix output mask.
- `opentad/models/detectors/actionformer.py:145-146,228-230`: both `forward_train` and `forward_test` call `_forward_backbone_dynamic_k` before token compression whenever a backbone is present.
- `opentad/models/detectors/actionformer.py:411-415`: `executed_k` is written per sample from the actual `bucket_k` sent to the heavy backbone, with source tag `heavy_backbone_input_temporal_length`; it is not copied from target length or output mask length.
- `opentad/models/backbones/backbone_wrapper.py:86-116,119-124`: wrapper flattens only the supplied batch/segment dimensions and invokes the model backbone on those frames; mask application is after feature extraction. The dynamic caller therefore controls the temporal input presented to the model, rather than relying on hidden target-length padding.
- `opentad/models/detectors/actionformer.py:150,233` asserts feature/mask temporal equality immediately after heavy execution; `_forward_backbone_dynamic_k` itself pads features and creates a same-length prefix mask (`:416-424`).
- `opentad/models/dense_heads/anchor_free_head.py:301-306,324-331`: physical-grid conversion is invoked in train and test head paths before losses or `get_valid_proposals_scores` (threshold/top-k proposal filtering path). `anchor_free_head.py:163-199` maps selected-axis centers/strides back through `irregular_selected_positions` and dense valid length.
- `opentad/models/utils/post_processing/utils.py:109-117`: post-processing converts selected-axis segments to dense coordinates before seconds conversion; NMS occurs earlier in `single_stage.py:138-145`, while physical coordinates have already been restored in the head at test time.

## Focused test

`tests/test_duca_dynamic_heavy_recovery.py:21-35` uses a spy backbone with mixed masks K=2 and K=4, asserting actual calls `[2, 4]`, output shape `(2,2,4)`, matching output masks, and `executed_k == [2,4]`. `tests/test_duca_dynamic_heavy_recovery.py:37-57` separately checks irregular selected positions reach seconds conversion. Static semantic tests are in `tests/test_duca_semantic_indirect_static.py:6-16`.

The focused pytest invocation could not collect because the local Windows PyTorch DLL failed to initialize (`WinError 1114`, `c10.dll`); `py_compile` completed before collection. This is an environment limitation, not a code failure observed in the static review.

## Scientific drift

No heavy-path scientific drift found: K is derived from deploy-visible selector masks, actual heavy input is sliced per bucket, batch order is restored, and physical coordinates are restored downstream before proposal filtering/NMS/seconds conversion. The implementation changes execution batching/transport representation only.

## Next owner

Main DUCA owner: proceed with independent six-arm/split/checkpoint correction and validation. Do not treat this PASS as closing those items or as evidence for final-method claims.
