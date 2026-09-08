import pytest

try:
    import torch
except (ImportError, OSError):
    pytest.skip("Torch unavailable on this host", allow_module_level=True)

from opentad.models.backbones.vit_adapter import Adapter, VisionTransformerAdapter


def test_tia_crosses_clip_boundary_but_not_video_boundary():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = Adapter(embed_dims=4, mlp_ratio=1, temporal_size=16).to(device)
    with torch.no_grad():
        for layer in (adapter.down_proj, adapter.up_proj):
            layer.weight.copy_(torch.eye(4, device=device))
            layer.bias.zero_()
        adapter.dwconv.weight.fill_(1)
        adapter.dwconv.bias.zero_()
        adapter.conv.weight.copy_(torch.eye(4, device=device).unsqueeze(-1))
        adapter.conv.bias.zero_()
    # Two videos, each packed as two 8-tubelet attention clips, one spatial site.
    x = torch.zeros(4, 8, 4, device=device, requires_grad=True)
    with torch.no_grad():
        x[0, 7] = 1
        x[1, 7] = 2
    result = adapter(x, 1, 1)
    assert result[1, 0].abs().sum() > 0
    assert torch.count_nonzero(result[2:]) == 0
    separate = torch.cat([adapter(x[:2], 1, 1), adapter(x[2:], 1, 1)])
    torch.testing.assert_close(result, separate)
    grad = torch.autograd.grad(result[1, 0].sum(), x)[0]
    assert grad[0, 7].abs().sum() > 0
    assert torch.count_nonzero(grad[2:]) == 0


def test_backbone_uses_complete_post_selection_temporal_extent():
    for frames, tubelets in ((384, 192), (768, 384)):
        model = VisionTransformerAdapter(img_size=16, patch_size=16, embed_dims=8,
                                         depth=1, num_heads=2, num_frames=16,
                                         tubelet_size=2, total_frames=frames, adapter_index=[0])
        assert model.blocks[0].adapter.temporal_size == tubelets
