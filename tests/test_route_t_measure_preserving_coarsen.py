import torch
from torch import nn

from opentad.models.backbones.vit_adapter import MeasurePreservingCoarsenRoute, RouteTEdgeRouter


class _AdapterBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_adapter = True
        self.core_token_counts = []
        self.adapter_token_counts = []

    def forward(self, x, h, w, apply_adapter=True):
        del h, w
        self.core_token_counts.append(int(x.shape[1]))
        assert apply_adapter is False
        return x + 1.0

    def adapter(self, x, h, w):
        del h, w
        self.adapter_token_counts.append(int(x.shape[1]))
        return x


def test_legal_partitions_and_full_block_path():
    route = MeasurePreservingCoarsenRoute(enabled=True, arm="uniform", embed_dims=4, run_seed=3407)
    assert len(route.legal_pairs) == 15
    assert all(right - left > 1 for left, right in route.legal_pairs)
    blocks = nn.ModuleList([_AdapterBlock(), _AdapterBlock()])
    tokens = torch.arange(2 * 8 * 3 * 4, dtype=torch.float32).reshape(2, 24, 4)
    output = route(tokens, tokens, blocks, h=1, w=3, training=True)
    assert output.shape == tokens.shape
    assert all(block.core_token_counts == [18] for block in blocks)
    assert all(block.adapter_token_counts == [24] for block in blocks)
    assert route.last_summary["support_complete"] is True


def test_router_seed_is_arm_independent_and_storage_is_independent():
    first = RouteTEdgeRouter(embed_dims=4, run_seed=5801)
    second = RouteTEdgeRouter(embed_dims=4, run_seed=5801)
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.equal(left, right)
        assert left.data_ptr() != right.data_ptr()


def test_boundary_loss_is_finite_and_backward_reaches_router():
    route = MeasurePreservingCoarsenRoute(enabled=True, arm="risk", embed_dims=4, run_seed=8123)
    blocks = nn.ModuleList([_AdapterBlock()])
    tokens = torch.randn(2, 24, 4, requires_grad=True)
    route(tokens, tokens, blocks, h=1, w=3, training=True)
    losses = route.auxiliary_loss([torch.tensor([[3.0, 11.0]]), torch.tensor([[5.0, 13.0]])])
    loss = losses["route_t_boundary_risk_loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert route.router.linear1.weight.grad is not None

