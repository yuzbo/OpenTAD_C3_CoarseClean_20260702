import copy
import json
import pytest
import torch

from geosparse_research.training_gradient_parity import compare_vectors, measure_parity
from test_geosparse_detector import batch, detector


def test_equal_norm_different_direction_is_not_equivalent():
    row = compare_vectors({"w": torch.tensor([1., 0.])}, {"w": torch.tensor([0., 1.])})
    assert row["scalar_norm_close"] and not row["vector_close"]
    assert row["difference_l2"] == pytest.approx(2 ** .5)


def test_disconnected_and_nonfinite_gradients_do_not_pass():
    assert not compare_vectors({}, {"w": torch.zeros(1)})["vector_close"]
    row = compare_vectors({"w": torch.tensor([float("nan")])}, {"w": torch.ones(1)})
    assert not row["vector_close"] and row["relative_l2"] is None


def test_real_source_vectors_and_artifacts_preserve_state(tmp_path):
    torch.manual_seed(5)
    model = detector("B", query_length=16)
    model.epoch = 6
    inputs = batch(empty=True)
    original = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    output = tmp_path / "parity"
    result = measure_parity(model, inputs, output, microbatch=1, amp=False, loss_scale=128., clip_norm=1.)
    assert result["attribution_equivalence_passed"]
    for name in ("ordinary_before", "ordinary_repeat", "ordinary_after", "decomposed_total"):
        tensors = torch.load(output / (name + ".pt"), map_location="cpu")
        assert tensors and all(t.device.type == "cpu" for t in tensors.values())
    assert json.loads((output / "parity.json").read_text())["comparisons"]["decomposed_total"]["vector_close"]
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.minibatch == 0 and all(parameter.grad is None for parameter in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_mismatch_evidence_is_saved_without_relaxing_the_gate(tmp_path, monkeypatch):
    import geosparse_research.training_gradients as gradients
    original_measure = gradients.measure_training_gradients

    def perturbed(*args, total_gradient_sink, **kwargs):
        def capture(total):
            changed = {name: value.clone() for name, value in total.items()}
            first = next(iter(changed))
            changed[first].add_(1.)
            total_gradient_sink(changed)
        return original_measure(*args, total_gradient_sink=capture, **kwargs)

    monkeypatch.setattr(gradients, "measure_training_gradients", perturbed)
    torch.manual_seed(5)
    model = detector("B", query_length=16)
    model.epoch = 6
    output = tmp_path / "mismatch"
    result = measure_parity(model, batch(empty=True), output, microbatch=1, amp=False, loss_scale=1., clip_norm=1.)
    assert not result["attribution_equivalence_passed"]
    assert (output / "decomposed_total.pt").is_file()
    assert (output / "decomposition.unverified.json").is_file()
    assert json.loads((output / "parity.json").read_text())["attribution_equivalence_passed"] is False
