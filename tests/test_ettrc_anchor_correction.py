import copy

import pytest

try:
    import torch
except (ImportError, OSError):
    pytest.skip("Torch unavailable on this host", allow_module_level=True)

from opentad.models.backbones.et_trc_videomae import TaylorResidualBlock


def block():
    torch.manual_seed(27)
    result = TaylorResidualBlock(embed_dims=8, num_heads=2, mlp_ratio=2,
                                 stride_k=4, segment_size=8, jacobian_rank=4)
    with torch.no_grad():
        result.jacobian_approx.temporal.weight.fill_(0.7)
    return result.eval()


def test_neighbor_taps_do_not_change_exact_anchors_before_adapter():
    candidate = block()
    dense = copy.deepcopy(candidate)
    dense.enable_taylor = False
    x = torch.randn(2, 16, 8)
    actual = candidate(x, h=1, w=2).reshape(2, 8, 2, 8)
    expected = dense(x, h=1, w=2).reshape(2, 8, 2, 8)
    torch.testing.assert_close(actual[:, [0, 4, 7]], expected[:, [0, 4, 7]], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(actual[:, [1, 2, 3, 5, 6]], expected[:, [1, 2, 3, 5, 6]])


def test_nonanchor_proxy_gradient_and_video_isolation():
    candidate = block()
    x = torch.randn(2, 16, 8, requires_grad=True)
    out = candidate(x, h=1, w=2)
    out.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for parameter in candidate.jacobian_approx.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    changed = x.detach().clone()
    changed[1] += 10
    torch.testing.assert_close(candidate(changed, h=1, w=2)[0], out.detach()[0])


def test_adapter_sees_exact_anchors_and_can_mix_them_afterwards():
    candidate = block()
    baseline = copy.deepcopy(candidate)
    baseline.enable_taylor = False
    class Capture(torch.nn.Module):
        def forward(self, x, h, w):
            self.observed = x.detach().clone()
            return x + x.mean(dim=1, keepdim=True)
    candidate.adapter = Capture()
    x = torch.randn(1, 16, 8)
    output = candidate(x, h=1, w=2)
    anchor_tokens = [0, 1, 8, 9, 14, 15]
    torch.testing.assert_close(candidate.adapter.observed[:, anchor_tokens], baseline(x, h=1, w=2)[:, anchor_tokens])
    assert not torch.equal(output, candidate.adapter.observed)
