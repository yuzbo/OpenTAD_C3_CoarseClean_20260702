from pathlib import Path

import pytest
from mmengine.config import ConfigDict

torch = pytest.importorskip("torch")

from opentad.models.backbones.vit_adapter import VisionTransformerAdapter
from opentad.models.backbones.backbone_wrapper import BackboneWrapper
from opentad.datasets import transforms as _dataset_transforms  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]


def _tiny_videomae_adapter():
    torch.manual_seed(7)
    return VisionTransformerAdapter(
        img_size=4,
        patch_size=2,
        embed_dims=8,
        depth=2,
        num_heads=2,
        mlp_ratio=2,
        qkv_bias=True,
        num_frames=4,
        tubelet_size=2,
        use_mean_pooling=False,
        return_feat_map=True,
        with_cp=False,
        total_frames=8,
        adapter_index=[0, 1],
        drop_path_rate=0.0,
    ).eval()


def test_strict_padding_mask_makes_valid_tokens_invariant_to_padding_pixels():
    model = _tiny_videomae_adapter()
    frames = torch.randn(2, 3, 4, 4, 4)
    temporal_mask = torch.tensor(
        [[True, True, True, False], [False, False, False, False]]
    )
    counterfactual = frames.clone()
    counterfactual[0, :, 3] = 1000.0
    counterfactual[1] = -1000.0

    with torch.no_grad():
        reference = model(frames, temporal_mask=temporal_mask)
        changed = model(counterfactual, temporal_mask=temporal_mask)

    assert torch.equal(reference[0, :, :2], changed[0, :, :2])
    assert torch.count_nonzero(reference[1]) == 0
    assert torch.count_nonzero(changed[1]) == 0


def test_strict_padding_mask_blocks_gradients_from_valid_outputs_to_padding_pixels():
    model = _tiny_videomae_adapter()
    frames = torch.randn(2, 3, 4, 4, 4, requires_grad=True)
    temporal_mask = torch.tensor(
        [[True, True, True, False], [False, False, False, False]]
    )

    outputs = model(frames, temporal_mask=temporal_mask)
    outputs[0, :, :2].sum().backward()

    assert torch.count_nonzero(frames.grad[0, :, 3]) == 0
    assert torch.count_nonzero(frames.grad[1]) == 0


def test_all_valid_mask_preserves_the_original_videomae_path():
    model = _tiny_videomae_adapter()
    frames = torch.randn(2, 3, 4, 4, 4)
    temporal_mask = torch.ones(2, 4, dtype=torch.bool)

    with torch.no_grad():
        original = model(frames)
        strict = model(frames, temporal_mask=temporal_mask)

    assert torch.equal(original, strict)


def test_every_transformer_block_and_temporal_adapter_keeps_invalid_tokens_zero():
    model = _tiny_videomae_adapter()
    frames = torch.randn(2, 3, 4, 4, 4)
    temporal_mask = torch.tensor(
        [[True, True, True, False], [False, False, False, False]]
    )
    token_mask = temporal_mask.reshape(2, 2, 2).any(dim=-1)
    token_mask = token_mask[:, :, None].expand(-1, -1, 4).reshape(2, 8)
    captured = []
    handles = []
    for block in model.blocks:
        handles.append(block.register_forward_hook(lambda _module, _inputs, output: captured.append(output)))
        handles.append(
            block.adapter.register_forward_hook(
                lambda _module, _inputs, output: captured.append(output)
            )
        )

    with torch.no_grad():
        model(frames, temporal_mask=temporal_mask)
    for handle in handles:
        handle.remove()

    assert len(captured) == 4
    for activations in captured:
        assert torch.count_nonzero(activations[~token_mask]) == 0


def test_actionformer_passes_raw_masks_into_the_backbone_for_train_and_test():
    source = (ROOT / "opentad/models/detectors/actionformer.py").read_text(encoding="utf-8")

    assert source.count("self.backbone(inputs, masks)") >= 2


def test_backbone_wrapper_isolates_padding_before_chunked_videomae():
    wrapper = BackboneWrapper(
        ConfigDict(
            type="mmaction.Recognizer3D",
            backbone=dict(
                type="VisionTransformerAdapter",
                img_size=4,
                patch_size=2,
                embed_dims=8,
                depth=2,
                num_heads=2,
                mlp_ratio=2,
                qkv_bias=True,
                num_frames=4,
                tubelet_size=2,
                use_mean_pooling=False,
                return_feat_map=True,
                with_cp=False,
                total_frames=8,
                adapter_index=[0, 1],
                drop_path_rate=0.0,
            ),
            data_preprocessor=dict(
                type="mmaction.ActionDataPreprocessor",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                format_shape="NCTHW",
            ),
            custom=dict(
                pretrain=None,
                strict_temporal_padding_mask=True,
                pre_processing_pipeline=[
                    dict(
                        type="Rearrange",
                        keys=["frames"],
                        ops="b n c (t1 t) h w -> (b t1) n c t h w",
                        t1=2,
                    )
                ],
                post_processing_pipeline=[
                    dict(
                        type="Reduce",
                        keys=["feats"],
                        ops="b n c t h w -> b c t",
                        reduction="mean",
                    ),
                    dict(
                        type="Rearrange",
                        keys=["feats"],
                        ops="(b t1) c t -> b c (t1 t)",
                        t1=2,
                    ),
                ],
                norm_eval=False,
                freeze_backbone=False,
            ),
        )
    ).eval()
    frames = torch.randn(1, 1, 3, 8, 4, 4)
    masks = torch.tensor([[True, True, True, False, False, False, False, False]])
    counterfactual = frames.clone()
    counterfactual[:, :, :, 3:] = 255.0

    with torch.no_grad():
        reference = wrapper(frames, masks)
        changed = wrapper(counterfactual, masks)

    assert torch.equal(reference[:, :, :2], changed[:, :, :2])
    assert torch.count_nonzero(reference[:, :, 2:]) == 0
    assert torch.count_nonzero(changed[:, :, 2:]) == 0
    assert wrapper.latest_temporal_padding_mask_summary["strict_isolation_verified"] is True
    assert wrapper.latest_temporal_padding_mask_summary["output_valid_counts"] == [2]
