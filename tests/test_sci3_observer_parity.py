import copy
import random

import numpy as np
import pytest
import torch

from geosparse_research.observer_parity import compare_observers, tensor_difference
from geosparse_research.fixed_budget_diagnostic import fixed_half_budget
from test_geosparse_detector import batch, detector


def test_comparison_retains_differences_and_nonfinite_evidence():
    row = tensor_difference(torch.tensor([1., 0., float("nan")]), torch.tensor([1.03125, .01, 0.]))
    assert not row["exact"] and row["different_elements"] == 3
    assert row["nonfinite_elements"] == 1 and row["max_abs"] == .03125
    assert row["max_rel_nonzero_reference"] == .03125
    assert not tensor_difference(torch.zeros(2), torch.zeros(3))["exact"]
    assert not tensor_difference(torch.ones(2).half(), torch.ones(2))["exact"]


def test_real_detector_separated_observers_and_repeats_are_exact_on_cpu():
    torch.manual_seed(23)
    model = detector("A", query_length=16).eval()
    model.geosparse.config["budget_mode"] = "dynamic"
    state = {key: value.clone() for key, value in model.state_dict().items()}
    config = copy.deepcopy(model.geosparse.config)
    with fixed_half_budget(model):
        result, raw = compare_observers(model, batch(), amp=False)
    assert result["status"] == "EXACT_IN_THIS_PROBE" and result["forward_calls"] == 9
    assert len(raw) == 9 and raw["plain_0"]["requested_budget"].item() == .5
    assert model.latest_route_plan is None and model.geosparse.config == config
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], atol=0, rtol=0)


def test_forward_exception_restores_state_and_removes_measurement_hooks():
    model = detector("A", query_length=16).eval()
    inputs = batch()
    original = model.forward_test
    buffers = {key: value.clone() for key, value in model.named_buffers()}
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    hooks = [(len(module._forward_hooks), len(module._forward_pre_hooks)) for module in model.modules()]
    calls = 0

    def failing(**inputs):
        nonlocal calls
        calls += 1
        if calls == 3:  # selection observer has been installed
            next(model.buffers()).add_(1)
            random.random(); np.random.rand(); torch.rand(3)
            raise RuntimeError("injected inference failure")
        return original(**inputs)

    model.forward_test = failing
    with pytest.raises(RuntimeError, match="injected inference failure"):
        compare_observers(model, inputs, amp=False)
    assert model.latest_route_plan is None and not model.training
    assert hooks == [(len(module._forward_hooks), len(module._forward_pre_hooks)) for module in model.modules()]
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[name], atol=0, rtol=0)
    assert random.getstate() == rng[0]
    assert np.random.get_state()[0] == rng[1][0]
    np.testing.assert_array_equal(np.random.get_state()[1], rng[1][1])
    torch.testing.assert_close(torch.get_rng_state(), rng[2], atol=0, rtol=0)
