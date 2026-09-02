import pytest

torch = pytest.importorskip("torch")
from torch import nn


def test_gamma_and_q_paths_receive_gradients():
    torch.manual_seed(11)
    gamma = nn.Parameter(torch.zeros(()))
    q0 = nn.Conv1d(4, 4, 1)
    q1 = nn.Conv1d(4, 4, 1)
    local, global_ = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    gate = torch.sigmoid(torch.randn(2, 1, 3))
    opt = torch.optim.SGD([gamma, *q0.parameters(), *q1.parameters()], lr=0.1)
    for step in range(3):
        opt.zero_grad()
        residual = gamma * gate * (q0(local) - q1(global_))
        (residual + global_).square().mean().backward()
        assert gamma.grad is not None and torch.isfinite(gamma.grad).all()
        if step == 0:
            assert float(gamma.grad.abs()) > 0.0
        opt.step()
    assert float(gamma.detach().abs()) > 0.0
    assert q0.weight.grad is not None and float(q0.weight.grad.abs().sum()) > 0.0
    assert q1.weight.grad is not None and float(q1.weight.grad.abs().sum()) > 0.0
