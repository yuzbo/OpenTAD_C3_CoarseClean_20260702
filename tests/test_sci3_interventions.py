"""Synthetic inputs through actual VideoMAE/TIA/ActionFormer, not mock metrics."""
from dataclasses import replace
import copy
import random

import numpy as np
import pytest
import torch

from geosparse_ext.data import video_batch
from geosparse_ext.geometry import canonical_atoms, native_layout
from geosparse_ext.matrix import base
from geosparse_ext.routing import make_plan
from geosparse_research.interventions import CounterfactualRunner, equal_cost_swap, parent_token_counts
from test_geosparse_detector import batch, detector


def source_case(route, empty=False, odd_tail=False):
    torch.manual_seed(14)
    model = detector(route, query_length=16).eval()
    data = batch(empty)
    if odd_tail:
        data["masks"][0, -1] = False
    video = video_batch(data["inputs"], data["masks"], data["metas"], "cpu")
    with torch.no_grad():
        _, plan, _, _, _ = model.geosparse(video)
    valid = native_layout(video).valid
    selected = torch.where(plan.selected_atoms[0])[0]
    unselected = torch.where(~plan.selected_atoms[0])[0]
    swapped = equal_cost_swap(plan, valid, route, 0, int(selected[0]), int(unselected[0]))
    return model, data, video, plan, swapped


def measure(model, data, video, left, right):
    return CounterfactualRunner(model).evaluate_pair(
        video, data["metas"], data["gt_segments"], data["gt_labels"], left, right)


@pytest.mark.parametrize("route", ["A", "B", "C"])
@pytest.mark.parametrize("empty", [False, True])
def test_actual_source_pair_same_plan_and_reverse_swap(route, empty):
    model, data, video, original, swapped = source_case(route, empty, odd_tail=True)
    same = measure(model, data, video, original, original)
    assert all(value == 0 for value in same["signed_gain"].values())
    forward = measure(model, data, video, original, swapped)
    reverse = measure(model, data, video, swapped, original)
    assert forward["left"]["heavy_macs"] == forward["right"]["heavy_macs"] > 0
    assert forward["left"]["detector_mask"] == data["masks"].tolist()
    assert forward["left"]["selected_atoms"] != forward["right"]["selected_atoms"]
    for key, value in forward["signed_gain"].items():
        assert np.isfinite(value)
        assert value == -reverse["signed_gain"][key]
    assert not swapped.learned_sample.any() and not swapped.log_prob.any()
    assert all(order.numel() == 0 for order in swapped.sampling_order)


@pytest.mark.parametrize("fail", [False, True])
def test_pair_restores_complete_state_rng_modes_and_tia_scope(monkeypatch, fail):
    model, data, video, original, swapped = source_case("B")
    model.train()
    model.geosparse.scout.eval()  # restoration must retain mixed modes
    model.geosparse.regular_tia.temporal_size = 123
    parameters = {key: value.detach().clone() for key, value in model.named_parameters()}
    buffers = {key: value.clone() for key, value in model.named_buffers()}
    modes = [module.training for module in model.modules()]
    flags = [parameter.requires_grad for parameter in model.parameters()]
    trace = copy.deepcopy(model.geosparse.encoder.trace)
    execution_rng = model.geosparse.execution_rng
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    if fail:
        loss = model._task_losses
        calls = []

        def fail_second(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                model.rpn_head.loss_normalizer.add_(99)
                random.random(), np.random.rand(), torch.rand(1)
                raise RuntimeError("injected second-branch failure")
            return loss(*args, **kwargs)

        monkeypatch.setattr(model, "_task_losses", fail_second)
        with pytest.raises(RuntimeError, match="second-branch"):
            measure(model, data, video, original, swapped)
    else:
        measure(model, data, video, original, swapped)
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, parameters[name], atol=0, rtol=0)
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[name], atol=0, rtol=0)
    assert modes == [module.training for module in model.modules()]
    assert flags == [parameter.requires_grad for parameter in model.parameters()]
    assert trace == model.geosparse.encoder.trace
    assert model.geosparse.execution_rng is execution_rng
    assert model.geosparse.regular_tia.temporal_size == 123
    assert rng[0] == random.getstate()
    now = np.random.get_state()
    assert rng[1][0] == now[0] and np.array_equal(rng[1][1], now[1]) and rng[1][2:] == now[2:]
    assert torch.equal(rng[2], torch.get_rng_state())


@pytest.mark.parametrize("route", ["A", "B", "C"])
def test_swap_uses_actual_valid_members_and_keeps_original_plan(route):
    atoms = canonical_atoms(16, 2, 2)
    valid = torch.ones(1, 16, 2, 2, dtype=torch.bool)
    valid[:, 6] = False
    config = base(route, selector="uniform", quota="per_clip_equal_quota")
    plan = make_plan(torch.zeros(1, len(atoms)), atoms, valid, 8, (2, 2), config, training=False)
    before = plan.selected_native.clone()
    donor = int(torch.where(plan.selected_atoms[0, :8])[0][0])
    candidates = [int(x) for x in torch.where(~plan.selected_atoms[0, :8])[0] if int(x) != 6]
    swapped = equal_cost_swap(plan, valid, route, 0, donor, candidates[0])
    assert torch.equal(parent_token_counts(plan, valid, route), parent_token_counts(swapped, valid, route))
    assert torch.equal(plan.selected_native, before)
    with pytest.raises(ValueError, match="nonempty equal-size"):
        equal_cost_swap(plan, valid, route, 0, donor, 6)
    other = int(torch.where(~plan.selected_atoms[0, 8:])[0][0]) + 8
    with pytest.raises(ValueError, match="one Heavy parent"):
        equal_cost_swap(plan, valid, route, 0, donor, other)
    for selected in [torch.zeros_like(plan.selected_atoms), torch.ones_like(plan.selected_atoms)]:
        with pytest.raises(ValueError, match="selected donor"):
            equal_cost_swap(replace(plan, selected_atoms=selected), valid, route, 0, donor, candidates[0])


def test_source_full_control_cannot_pretend_to_accept_a_forced_plan():
    model = detector("A")
    model.geosparse.config.update(selector="none", budget=1.)
    with pytest.raises(ValueError, match="overrides forced plans"):
        CounterfactualRunner(model)
