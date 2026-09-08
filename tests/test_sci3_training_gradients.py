"""Real small VideoMAE/TIA/ActionFormer gradients, with synthetic video input."""
import copy
import random

import numpy as np
import pytest
import torch

from geosparse_ext.runtime import subset_batch
from geosparse_research.training_gradients import measure_training_gradients
from test_geosparse_detector import batch, detector


def setup(route, empty=False):
    torch.manual_seed(5)
    model = detector(route, query_length=16).train()
    model.epoch = 6
    data = batch(empty)
    data["masks"][0, -1] = False
    return model, data


@pytest.mark.parametrize("route", ["A", "B", "C"])
def test_components_reconstruct_real_total_and_match_plain_backward(route):
    model, data = setup(route)
    measured = measure_training_gradients(model, data, loss_scale=128.)
    assert measured["gradient_status"] == "FINITE"
    assert measured["reconstruction"]["relative_l2"] < 2e-5
    assert measured["acquisition_forward_calls"] == 1
    assert set(measured["gradients"]) == {"task", "actor", "critic", "acquisition", "total"}
    for name in ["actor", "critic", "acquisition"]:
        assert measured["gradients"][name]["groups"]["scout_shared"]["norm"] > 0
        assert "rpn_head" not in measured["gradients"][name]["groups"]
    # State and RNG restoration makes the ordinary next forward identical.
    model.forward_train(**data)["cost"].backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    assert measured["global_clip"]["preclip_norm"] == pytest.approx(float(norm), rel=2e-5)
    assert measured["global_clip"]["coefficient"] == pytest.approx(min(1., 1. / (float(norm) + 1e-6)), rel=2e-5)


@pytest.mark.parametrize("fail", [False, True])
def test_training_diagnostic_restores_state_grads_rng_and_mixed_modes(monkeypatch, fail):
    model, data = setup("B")
    model.geosparse.config["budget_mode"] = "dynamic"
    model.geosparse.scout.eval()
    model.geosparse.regular_tia.temporal_size = 123
    model.pending_cost = [torch.tensor([.4])]
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    params = {name: value.clone() for name, value in model.state_dict().items()}
    grads = {name: value.grad for name, value in model.named_parameters()}
    flags = [p.requires_grad for p in model.parameters()]
    modes = [m.training for m in model.modules()]
    pending = model.pending_cost
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    normalizer = model.rpn_head.loss_normalizer.clone()
    if fail:
        forward = model.forward_train

        def broken_forward(**kwargs):
            forward(**kwargs)
            random.random(), np.random.rand(), torch.rand(1)
            raise RuntimeError("injected after full training forward")

        monkeypatch.setattr(model, "forward_train", broken_forward)
        with pytest.raises(RuntimeError, match="injected"):
            measure_training_gradients(model, data)
    else:
        measure_training_gradients(model, data)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, params[name], atol=0, rtol=0)
    for name, value in model.named_parameters():
        assert value.grad is grads[name]
        if value.grad is not None:
            assert torch.all(value.grad == 1)
    assert flags == [p.requires_grad for p in model.parameters()]
    assert modes == [m.training for m in model.modules()]
    assert model.minibatch == 0 and model.latest_route_plan is None and model.latest_acquisition is None
    assert model.pending_cost is pending and len(pending) == 1
    assert model.geosparse.regular_tia.temporal_size == 123
    torch.testing.assert_close(model.rpn_head.loss_normalizer, normalizer, atol=0, rtol=0)
    assert random.getstate() == rng[0]
    current = np.random.get_state()
    assert current[0] == rng[1][0] and np.array_equal(current[1], rng[1][1]) and current[2:] == rng[1][2:]
    assert torch.equal(rng[2], torch.get_rng_state())


def test_microbatch_aggregation_matches_the_same_partition_with_empty_gt():
    model, first = setup("C")
    second = batch(empty=True)
    data = {key: torch.cat((first[key], second[key]), 0) if torch.is_tensor(first[key])
            else first[key] + second[key] for key in first}
    measured = measure_training_gradients(model, data, microbatch_size=1)
    assert measured["primary_forward_calls"] == 2
    assert [r["minibatch"] for r in measured["microbatches"]] == [0, 1]
    assert measured["microbatches"][0]["normalizer_after"] == measured["microbatches"][1]["normalizer_before"]
    for i in range(2):
        (model.forward_train(**subset_batch(data, i, i + 1))["cost"] / 2).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    assert measured["global_clip"]["preclip_norm"] == pytest.approx(float(norm), rel=2e-5)
    assert measured["reconstruction"]["relative_l2"] < 2e-5


def test_warmup_and_actionness_use_only_actual_present_losses():
    model, data = setup("A", empty=True)
    model.epoch = 0
    model.geosparse.config["actionness_aux"] = True
    measured = measure_training_gradients(model, data)
    assert set(measured["gradients"]) == {"task", "actionness", "total"}
    assert measured["acquisition_forward_calls"] == 0
    assert measured["reconstruction"]["relative_l2"] < 2e-5


def test_frozen_backbone_stays_frozen_and_cpu_amp_is_not_implied():
    model, data = setup("A")
    with pytest.raises(ValueError, match="CUDA"):
        measure_training_gradients(model, data, amp=True)
    measured = measure_training_gradients(model, data)
    assert "source_backbone" not in measured["gradients"]["total"]["groups"]
    assert measured["gradients"]["task"]["groups"]["source_tia"]["norm"] > 0
