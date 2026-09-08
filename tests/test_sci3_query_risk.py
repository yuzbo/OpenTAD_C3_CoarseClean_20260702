"""Source-operator checks; synthetic inputs, no scientific performance scores."""
import copy

import pytest
import torch

from geosparse_research.query_risk import collect_query_risk, evaluate_query_pair
from test_geosparse_detector import detector
from test_sci3_interventions import measure, source_case


def query_pair(model, data, video, left, right):
    return evaluate_query_pair(model, video, data["metas"], data["gt_segments"],
                               data["gt_labels"], left, right)


@pytest.mark.parametrize("route", ["A", "B", "C"])
@pytest.mark.parametrize("empty", [False, True])
def test_actual_pair_retains_scalar_risk_and_signed_query_gains(route, empty):
    model, data, video, original, swapped = source_case(route, empty, odd_tail=True)
    expected = measure(model, data, video, original, swapped)
    measured = query_pair(model, data, video, original, swapped)
    reverse = query_pair(model, data, video, swapped, original)
    assert measured["signed_gain"] == expected["signed_gain"]
    for side in ("left", "right"):
        assert measured[side]["risk"] == expected[side]["risk"]
        assert measured[side]["trace"] == expected[side]["trace"]
        queries = measured[side]["query_risk"]
        assert queries["points"].shape == (24, 4)  # 16 + 8 FPN locations, not 24 input frames
        assert queries["level"].tolist() == [0] * 16 + [1] * 8
        assert not queries["valid"][0, 15]
        assert bool(queries["positive"].any()) != empty
        for key, values in queries["components"].items():
            assert not values[~queries["valid"]].any()
            assert float(values.double().sum()) == pytest.approx(measured[side]["risk"][key], abs=1e-6, rel=1e-5)
        assert not queries["components"]["reg_loss"][~queries["positive"]].any()
        assert queries["components"]["cls_loss"][queries["valid"] & ~queries["positive"]].sum() > 0
    for key, gain in measured["signed_query_gain"].items():
        assert torch.equal(gain, -reverse["signed_query_gain"][key])
        assert abs(measured["query_gain_reduction_residual"][key]) < 2e-6
    assert "losses" not in model.rpn_head.__dict__


def head_inputs():
    torch.manual_seed(91)
    head = detector("A").rpn_head.eval()
    cls = [torch.randn(2, 2, n) for n in (16, 8)]
    reg = [torch.rand(2, 2, n) * 4 for n in (16, 8)]
    mask = [torch.arange(n)[None] < torch.tensor([n - 1, n // 2])[:, None] for n in (16, 8)]
    points = head.prior_generator([torch.zeros(2, 16, n) for n in (16, 8)])
    segments = [torch.tensor([[1., 7.], [1., 7.], [6., 14.]]), torch.empty(0, 2)]
    labels = [torch.tensor([0, 1, 1]), torch.empty(0, dtype=torch.long)]
    return head, (cls, reg, mask, points, segments, labels)


@pytest.mark.parametrize("center,smoothing,weight", [("radius", 0., 1.), ("none", .1, 2.), ("radius", .1, 0.)])
def test_source_targets_multilabel_smoothing_and_regression_weight(center, smoothing, weight):
    head, inputs = head_inputs()
    head.center_sample, head.label_smoothing, head.loss_weight = center, smoothing, weight
    normalizer = head.loss_normalizer.clone()
    with torch.no_grad():
        losses = head.losses(*inputs)
    result = collect_query_risk(head, *inputs, source_losses=losses)
    assert (result["class_targets"].sum(-1) > 1).any()
    assert not result["positive"][1].any()
    assert torch.equal(head.loss_normalizer, normalizer)
    for key in ("cls_loss", "reg_loss"):
        assert float(result["components"][key].double().sum()) == pytest.approx(float(losses[key]), rel=1e-5, abs=1e-6)


def test_no_valid_queries_preserves_zero_source_risk():
    head, inputs = head_inputs()
    for mask in inputs[2]:
        mask.zero_()
    with torch.no_grad():
        source = head.losses(*inputs)
    result = collect_query_risk(head, *inputs, source_losses=source)
    assert result["normalizer"] == 1
    assert all(not value.any() for value in result["components"].values())


@pytest.mark.parametrize("fail", [False, True])
def test_query_pair_restores_original_method_and_complete_eval_state(monkeypatch, fail):
    model, data, video, original, swapped = source_case("B")
    model.train()
    model.geosparse.scout.eval()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    modes = [m.training for m in model.modules()]
    trace = copy.deepcopy(model.geosparse.encoder.trace)
    rng = torch.get_rng_state()
    method = model.rpn_head.losses
    calls = []

    def original_instance(*args, **kwargs):
        calls.append(None)
        if fail and len(calls) == 2:
            raise RuntimeError("injected query-loss failure")
        return method(*args, **kwargs)

    monkeypatch.setattr(model.rpn_head, "losses", original_instance)
    if fail:
        with pytest.raises(RuntimeError, match="query-loss failure"):
            query_pair(model, data, video, original, swapped)
    else:
        query_pair(model, data, video, original, swapped)
    assert model.rpn_head.losses is original_instance
    assert len(calls) == 2
    assert modes == [m.training for m in model.modules()]
    assert trace == model.geosparse.encoder.trace
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[key])


def test_training_normalization_is_not_relabelled_eval_risk():
    head, inputs = head_inputs()
    head.train()
    with pytest.raises(ValueError, match="FP32 source-head eval"):
        collect_query_risk(head, *inputs, source_losses={})
