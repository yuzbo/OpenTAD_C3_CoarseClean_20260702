"""Observer tests use synthetic tensors only; they do not produce scientific metrics."""
import base64
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from geosparse_ext.analysis_capture import OperationMacCounter, SelectionObserver


def test_mac_counter_preserves_outputs_gradients_and_counts_executed_operations():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.scout = nn.Conv2d(3, 2, 1)
            self.encoder = nn.Linear(2, 4)

        def forward(self, x):
            y = self.encoder(self.scout(x).mean((-2, -1)))
            return y @ y.T

    model = Model()
    x = torch.randn(2, 3, 4, 4, requires_grad=True)
    reference = model(x)
    reference.sum().backward()
    expected_grad = x.grad.clone()
    x.grad = None
    counter = OperationMacCounter(model)
    with counter:
        measured = model(x)
    measured.sum().backward()
    torch.testing.assert_close(reference, measured, atol=0, rtol=0)
    torch.testing.assert_close(expected_grad, x.grad, atol=0, rtol=0)
    assert counter.report()["macs"] == 192 + 16 + 16
    assert counter.report()["components"]["scout_router"] == 192
    assert counter.report()["components"]["heavy_encoder"] == 16
    assert counter.report()["complete_for_conv_linear_matmul"]
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_unsupported_executed_operator_is_not_reported_as_complete():
    model = nn.ConvTranspose2d(1, 1, 2)
    counter = OperationMacCounter(model)
    with counter:
        model(torch.ones(1, 1, 3, 3))
    assert not counter.report()["complete_for_conv_linear_matmul"]
    assert counter.report()["unsupported"]


def test_tail_mask_and_mixed_scale_zero_fine_still_has_heavy_cost():
    observer = object.__new__(SelectionObserver)
    encoder = SimpleNamespace(source=SimpleNamespace(embed_dims=8, blocks=[None]), trace=[dict(layer=0, parent=0, qkv_tokens=8, mlp_tokens=8)])
    observer.model = SimpleNamespace(geosparse=SimpleNamespace(encoder=encoder))
    observer.evidence = None
    selected = np.zeros((1, 8 * 4), dtype=bool)
    selected[0, -4:] = True  # padded tubelet must never appear selected
    observer.payload = dict(selected_native=selected, source_frame_id=np.arange(16), valid_frames=np.array([True] * 13 + [False] * 3),
        video_id="unit-fixture", window_id="unit-fixture:0", requested_budget=0., regular_s=np.arange(16.),
        intervals_s=np.stack([np.arange(16.), np.arange(1, 17.)], -1), spatial_transform=np.eye(3))
    counter = dict(macs=100, components={"heavy_encoder": 100}, scope="unit test", complete_for_conv_linear_matmul=True, unsupported={})
    row, arrays = observer.record(dict(route="C", model=dict(backbone="videomae_b")), counter, 0)
    unpacked = np.unpackbits(np.frombuffer(base64.b64decode(row["selected_bits"]), dtype=np.uint8), bitorder="little")
    assert unpacked.sum() == row["selected_native_members"] == 0
    assert row["support_valid"][6] == [True, False]
    assert row["heavy_macs"] > 0 and row["zero_heavy_parent_fraction"] == 0
    assert row["valid_native_members"] == 7 * 4
