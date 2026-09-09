import copy
import torch
import pytest

from geosparse_research.ordinary_repeat_diagnostic import (
    algorithm_mode, backend_settings, compare_forwards, measure_repeats, ordinary_trial,
)
from test_geosparse_detector import batch, detector


def test_strict_settings_restore_after_exception():
    original = backend_settings()
    with pytest.raises(RuntimeError, match="injected"):
        with algorithm_mode("strict") as settings:
            assert settings["deterministic_algorithms"] and not settings["deterministic_warn_only"]
            assert settings["cudnn_deterministic"] and not settings["cudnn_benchmark"]
            raise RuntimeError("injected")
    assert backend_settings() == original


def test_plan_identity_is_not_just_token_count():
    left = dict(plan={"selected_native": torch.tensor([True, False]),
                      "log_prob": torch.tensor(.1)}, losses={"cost": torch.tensor(2.)}, acquisition=None)
    right = copy.deepcopy(left)
    right["plan"]["selected_native"] = torch.tensor([False, True])
    assert not compare_forwards(left, right)["executed_choices_exact"]
    right = copy.deepcopy(left)
    right["plan"]["log_prob"].add_(1.)
    result = compare_forwards(left, right)
    assert result["executed_choices_exact"] and result["differing_plan_fields"] == ["log_prob"]
    right["losses"]["cost"].add_(1.)
    assert not compare_forwards(left, right)["all_losses_exact"]


def test_source_and_strict_real_b_repeat_restore_state(tmp_path):
    torch.manual_seed(5)
    model = detector("B", query_length=16)
    model.epoch = 6
    inputs = batch(empty=False)
    original = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    settings = backend_settings()
    for mode in ("source", "strict"):
        result = measure_repeats(model, inputs, tmp_path / mode,
                                microbatch=1, amp=False, loss_scale=128., mode=mode)
        for row in result["comparisons"].values():
            assert row["gradients"]["vector_close"]
            assert all(f["executed_choices_exact"] and f["all_losses_exact"]
                       and f["acquisition_exact"] for f in row["forwards"])
        assert not result["loss_component_attribution_result"]
    assert torch.equal(torch.get_rng_state(), rng) and backend_settings() == settings
    assert model.minibatch == 0 and model.latest_route_plan is None and not model.pending_cost
    assert all(p.grad is None for p in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_backward_failure_retains_forward_and_restores_state(tmp_path, monkeypatch):
    torch.manual_seed(5)
    model = detector("B", query_length=16)
    model.epoch = 6
    inputs = batch(empty=True)
    original = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    def fail(*args, **kwargs):
        raise RuntimeError("injected unsupported backward")
    monkeypatch.setattr(torch.Tensor, "backward", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="unsupported backward"):
        ordinary_trial(model, inputs, output, microbatch=1, amp=False, loss_scale=1.)
    forward = torch.load(output / "forward-000.pt", map_location="cpu")
    assert forward["plan"]["selected_native"].dtype == torch.bool and "cost" in forward["losses"]
    assert not (output / "gradients.pt").exists()
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.minibatch == 0 and model.latest_route_plan is None and not model.pending_cost
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
