"""Real tiny detector checks; no scientific accuracy is generated here."""
import copy

import pytest
import torch

from geosparse_ext.analysis_capture import OperationMacCounter, SelectionObserver
from geosparse_ext.matrix import base
from geosparse_research.fixed_budget_diagnostic import fixed_half_budget, add_eligible_heavy_reference, cost_summary
from test_geosparse_detector import batch, detector


@pytest.mark.parametrize("tail", [16, 15, 13])
@torch.no_grad()
def test_fixed_override_reduces_actual_heavy_and_same_pass_observer_preserves_output(tail):
    torch.manual_seed(23)
    model = detector("A", query_length=16).eval()
    model.geosparse.config["budget_mode"] = "dynamic"
    model.geosparse.scout.budget.weight.zero_()
    model.geosparse.scout.budget.bias.copy_(torch.tensor([0., 0., 0., 0., 100.]))
    inputs = batch()
    inputs["masks"][:, tail:] = False
    original = copy.deepcopy(model.geosparse.config)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    model.forward_test(**inputs)
    full = sum(row["qkv_tokens"] for row in model.geosparse.encoder.trace)
    assert model.latest_route_plan.requested_budget.item() == 1.
    with fixed_half_budget(model):
        assert model.geosparse.config["budget_mode"] == "fixed"
        observer = SelectionObserver(model)
        try:
            with OperationMacCounter(model) as counter:
                measured = model.forward_test(**inputs)
            row, arrays = observer.record(dict(route="A", model=base("A")), counter.report(), 0)
        finally:
            observer.close()
        add_eligible_heavy_reference(row)
        assert row["requested_budget"] == .5
        assert sum(layer["qkv_tokens"] for layer in row["layers"]) < full
        assert 0 < row["heavy_eligible_full_ratio"] < 1
        assert row["heavy_eligible_full_ratio"] != .5
        assert arrays["valid_frames"].sum() == tail
        unobserved = model.forward_test(**inputs)
        for left, right in zip(measured[0] + measured[1], unobserved[0] + unobserved[1]):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        assert cost_summary([row])["windows"] == 1
    assert model.geosparse.config == original
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], atol=0, rtol=0)


def test_override_exception_restores_config():
    model = detector("A").eval()
    model.geosparse.config["budget_mode"] = "dynamic"
    original = copy.deepcopy(model.geosparse.config)
    with pytest.raises(RuntimeError, match="injected"):
        with fixed_half_budget(model):
            raise RuntimeError("injected")
    assert model.geosparse.config == original


@pytest.mark.parametrize("side", [10, 14])
def test_actual_grid_and_odd_frame_eligibility_denominators(side):
    d = 768
    full_parent = 8 * side * side
    mac = lambda k: 12 * k * d * d + 2 * k * k * d
    row = dict(route="A", native_shape=[16, side, side], valid_tubelets=[True] * 9 + [False] * 7,
        heavy_reference=dict(parent_shape=[8, side, side], width=d, depth=12),
        heavy_reference_macs=24 * mac(full_parent),
        heavy_macs=12 * (mac(full_parent) + mac(side * side)))
    row["heavy_mac_ratio"] = row["heavy_macs"] / row["heavy_reference_macs"]
    add_eligible_heavy_reference(row)
    assert row["eligible_parent_tokens"] == [full_parent, side * side]
    assert row["heavy_eligible_full_ratio"] == 1.
    assert 0 < row["heavy_padded_full_ratio"] < 1.
