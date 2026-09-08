import torch
import pytest
from mmengine.config import ConfigDict
from geosparse_ext.matrix import base
from geosparse_ext.detector import GeoSparseDetector


def detector(route, estimator="pg_acquisition", input_positions=16, query_length=8):
    config = base(route, query_length=query_length, scout_width=16, scout_resolution=32,
                  estimator=estimator, exploration="none", probe_every=1)
    return GeoSparseDetector(
        projection=dict(type="Conv1DTransformerProj", in_channels=256 if route == "B" else 16,
                        out_channels=16, arch=(1, 1, 1), conv_cfg=dict(kernel_size=3, proj_pdrop=0.),
                        norm_cfg=dict(type="LN"), attn_cfg=dict(n_head=2, n_mha_win_size=-1),
                        path_pdrop=0., use_abs_pe=False, max_seq_len=query_length),
        neck=dict(type="FPNIdentity", in_channels=16, out_channels=16, num_levels=2),
        rpn_head=ConfigDict(type="ActionFormerHead", num_classes=2, in_channels=16, feat_channels=16,
                      num_convs=1, prior_generator=dict(type="PointGenerator", strides=[1, 2],
                      regression_range=[(0, 4), (4, 10000)]), loss_normalizer=10,
                      loss=dict(cls_loss=dict(type="FocalLoss"), reg_loss=dict(type="DIOULoss"))),
        geosparse=config,
        source_config=dict(img_size=32, embed_dims=16, depth=2, num_heads=2, total_frames=input_positions, drop_path_rate=0.,
                           adapter_index=[] if route == "B" else [0, 1]))


def batch(empty=False):
    intervals = torch.stack((torch.arange(16.) / 25, (torch.arange(16.) + 1) / 25), -1).numpy()
    return dict(inputs=torch.rand(1, 1, 3, 16, 32, 32) * 255,
                masks=torch.ones(1, 16, dtype=torch.bool),
                metas=[dict(video_name="unit", duration=1., fps=25., snippet_stride=1, offset_frames=0,
                            geosparse=dict(source_frame_id=list(range(16)), intervals_s=intervals,
                                           regular_s=list(torch.arange(16.).numpy() / 25),
                                           spatial_transform=torch.eye(3).numpy()))],
                gt_segments=[torch.empty(0, 2) if empty else torch.tensor([[2., 10.]])],
                gt_labels=[torch.empty(0, dtype=torch.long) if empty else torch.tensor([0])])


@pytest.mark.parametrize("route", ["A", "B", "C"])
@pytest.mark.parametrize("empty", [False, True])
def test_real_detector_pg_acquisition_backward(route, empty):
    torch.manual_seed(5)
    model = detector(route).train()
    model.epoch = 6
    losses = model.forward_train(**batch(empty))
    assert "actor_loss" in losses and "acquisition_loss" in losses
    assert torch.isfinite(losses["cost"])
    losses["cost"].backward()
    assert model.geosparse.scout.gain[-1].weight.grad is not None
    assert model.latest_acquisition["kind"] == "acquisition"
    assert all(p.grad is None for n, p in model.geosparse.encoder.source.named_parameters() if not p.requires_grad)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.rpn_head.parameters())


def test_pg_does_not_run_during_random_warmup():
    model = detector("A").train()
    losses = model.forward_train(**batch())
    assert "actor_loss" not in losses and "acquisition_loss" not in losses
    losses["cost"].backward()
    assert model.geosparse.scout.gain[-1].weight.grad is None


@pytest.mark.parametrize("route", ["A", "B", "C"])
@torch.no_grad()
def test_device_benchmark_boundary_matches_normal_inference(route):
    from geosparse_ext.data import video_batch
    model = detector(route).eval()
    inputs = batch()
    expected = model.forward_test(**inputs)
    device_batch = video_batch(inputs["inputs"], inputs["masks"], inputs["metas"], "cpu")
    actual = model.forward_video_batch(device_batch, inputs["metas"])
    for a, b in zip(expected[0] + expected[1], actual[0] + actual[1]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_b_112_fidelity_uses_seven_patch_grid_and_acquisition_gradients():
    model = detector("B").train()
    model.geosparse.config.update(source_resolution=112, axis="T")
    model.epoch = 6
    losses = model.forward_train(**batch())
    assert model.latest_route_plan.selected_native.shape == (1, 1, 8 * 7 * 7)
    assert all(row["native_tokens"] == 8 * 7 * 7 for row in model.geosparse.encoder.trace)
    assert "acquisition_loss" in losses
    losses["cost"].backward()
    assert model.geosparse.scout.gain[-1].weight.grad is not None


def test_ema_and_counterfactual_preserve_official_scalar_loss_normalizer():
    from geosparse_ext.runtime import ModelEMA, restore_mutable_state
    from opentad.utils.ema import ModelEma
    model = detector("B").train()
    ema = ModelEMA(model)
    official_ema = ModelEma(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    losses = model.forward_train(**batch())
    losses["cost"].backward()
    optimizer.step()
    ema.update(model)
    official_ema.update(model)
    assert model.rpn_head.loss_normalizer.is_floating_point()
    for name, value in ema.shadow.items():
        torch.testing.assert_close(value, official_ema.module.state_dict()[name], atol=0, rtol=0)
    restored = detector("B")
    restore_mutable_state(restored, dict(format="geosparse_full_state_v2", state_dict_ema=ema.state_dict()), "state_dict_ema")
    for name, value in ema.shadow.items():
        torch.testing.assert_close(restored.state_dict()[name], value)


@pytest.mark.parametrize("route", ["A", "B", "C"])
def test_ordinary_resume_preserves_fractional_normalizer_and_next_gradients(route):
    import copy
    from geosparse_ext.runtime import restore_mutable_state, seed_all

    seed_all(29)
    model = detector(route).train()
    inputs = batch(empty=True)
    model.forward_train(**inputs)
    normalizer = model.rpn_head.loss_normalizer
    assert normalizer.is_floating_point() and float(normalizer) != int(normalizer)
    saved = dict(format="geosparse_full_state_v2", state_dict=copy.deepcopy(model.state_dict()))
    restored = detector(route).train()
    parameter_ids = [id(p) for p in restored.parameters()]
    assert not restored.rpn_head.loss_normalizer.is_floating_point()
    restore_mutable_state(restored, saved, "state_dict")
    assert [id(p) for p in restored.parameters()] == parameter_ids
    restored.epoch, restored.minibatch = model.epoch, model.minibatch
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    outputs = []
    for current in (model, restored):
        seed_all(31)
        current.zero_grad(set_to_none=True)
        losses = current.forward_train(**inputs)
        losses["cost"].backward()
        outputs.append(losses)
    for name in outputs[0]:
        torch.testing.assert_close(outputs[0][name], outputs[1][name], rtol=0, atol=0)
    for left, right in zip(model.parameters(), restored.parameters()):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


@pytest.mark.parametrize("route", ["A", "B", "C"])
def test_official_detector_optimizer_covers_extensions_exactly_once(route):
    from geosparse_ext.runtime import optimizer_for
    model = detector(route).train()
    cfg = ConfigDict(lr=1e-4, weight_decay=.05,
                     backbone=dict(custom=[dict(lr=1e-4, weight_decay=.05)]))
    optimizer = optimizer_for(model, cfg)
    loss = model.forward_train(**batch())["cost"]
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
