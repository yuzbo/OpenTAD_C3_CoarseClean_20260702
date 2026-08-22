import torch
from opentad.models.backbones.physical_time import build_canonical_time_residual_bias
from opentad.models.backbones.backbone_wrapper import BackboneWrapper

def test_bias_shape_formula_and_uniform_zero():
    u = torch.arange(4, dtype=torch.float32).view(1, 4)
    b = build_canonical_time_residual_bias(u, u, 2, 1)
    assert b.shape == (1, 1, 8, 8) and torch.equal(b, torch.zeros_like(b))
    t = torch.tensor([[0., 1., 3., 4.]])
    b = build_canonical_time_residual_bias(t, u, 2, 1)
    assert torch.isfinite(b).all() and b.dtype == t.dtype

def test_temporal_to_spatial_broadcast():
    t = torch.tensor([[0., 1.]])
    u = torch.tensor([[0., 2.]])
    b = build_canonical_time_residual_bias(t, u, 2, 1)
    assert b[0, 0, 0, 3] == b[0, 0, 1, 2]

def test_shared_bias_rejects_per_head_expansion():
    u = torch.arange(4, dtype=torch.float32).view(1, 4)
    try:
        build_canonical_time_residual_bias(u, u, 2, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("SingleClock must reject num_heads != 1")

def test_temporal_checkpointing_slices_position_temporal_axis(monkeypatch):
    class StubBackbone:
        def __call__(self, frames, physical_positions=None):
            return frames + physical_positions[:, None, :, None, None]

    wrapper = object.__new__(BackboneWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = type("M", (), {"backbone": StubBackbone()})()
    monkeypatch.setattr(
        "opentad.models.backbones.backbone_wrapper.cp.checkpoint",
        lambda fn, frames, positions, use_reentrant=False: fn(frames, positions),
    )
    frames = torch.zeros(2, 3, 4, 1, 1)
    positions = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    out = wrapper.temporal_checkpointing(frames, 2, 2, positions)
    assert out.shape == frames.shape
    assert torch.equal(out[:, 0, :, 0, 0], positions)

def test_temporal_checkpointing_batch_axis_keeps_position_batches(monkeypatch):
    class StubBackbone:
        def __call__(self, frames, physical_positions=None):
            return frames + physical_positions[:, None, :, None, None]
    wrapper = object.__new__(BackboneWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = type("M", (), {"backbone": StubBackbone()})()
    monkeypatch.setattr(
        "opentad.models.backbones.backbone_wrapper.cp.checkpoint",
        lambda fn, frames, positions, use_reentrant=False: fn(frames, positions),
    )
    frames = torch.zeros(4, 3, 2, 1, 1)
    positions = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    out = wrapper.temporal_checkpointing(frames, 2, 0, positions)
    assert torch.equal(out[:, 0, :, 0, 0], positions)

def test_selected_positions_flat_and_segmented_contract():
    flat = torch.arange(2 * 24 * 16, dtype=torch.float32).reshape(2, 384)
    segmented = BackboneWrapper.normalize_physical_positions(flat, 2, 24, 16)
    assert segmented.shape == (48, 16)
    assert torch.equal(segmented[1], flat[0, 16:32])
    already = flat.reshape(2, 24, 16)
    assert torch.equal(BackboneWrapper.normalize_physical_positions(already, 2, 24, 16), segmented)
    for bad in (torch.zeros(2, 16), torch.zeros(2, 24, 8), torch.zeros(48, 16)):
        try:
            BackboneWrapper.normalize_physical_positions(bad, 2, 24, 16)
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous or mismatched position shape must be rejected")
