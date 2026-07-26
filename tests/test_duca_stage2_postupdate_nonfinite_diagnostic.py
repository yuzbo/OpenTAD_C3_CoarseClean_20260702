import torch
import pytest

from tools.bata.diagnose_duca_stage2_postupdate_nonfinite import (
    _capture_contribution_distribution_loss,
    _capture_selected_detector_contribution,
    _optimizer_step_ran,
    _summarize_named_tensors,
    _validate_prefix_target_indices,
)


def test_optimizer_step_receipt_matches_grad_scaler_scale_rule():
    assert _optimizer_step_ran(8192.0, 8192.0) is True
    assert _optimizer_step_ran(8192.0, 16384.0) is True
    assert _optimizer_step_ran(8192.0, 4096.0) is False


def test_named_tensor_summary_reports_the_nonfinite_tensor_name():
    summary = _summarize_named_tensors(
        [
            ("finite", torch.tensor([1.0, 2.0])),
            ("bad", torch.tensor([float("nan"), float("inf")])),
        ]
    )

    assert summary["finite"] is False
    assert summary["nan_count"] == 1
    assert summary["posinf_count"] == 1
    assert summary["nonfinite_names"] == ["bad"]


def test_prefix_target_indices_require_the_next_batch_after_the_update_prefix():
    assert _validate_prefix_target_indices(2, None) == (2, 2)
    assert _validate_prefix_target_indices(2, 2) == (2, 2)
    with pytest.raises(ValueError, match="positive"):
        _validate_prefix_target_indices(0, None)
    with pytest.raises(ValueError, match="immediately after"):
        _validate_prefix_target_indices(2, 1)


def test_contribution_capture_preserves_finite_first_order_contribution_math():
    class Selector:
        @staticmethod
        def _selected_detector_contribution(selected_inputs, objective):
            gradient = torch.autograd.grad(objective, selected_inputs, retain_graph=True)[0]
            return (selected_inputs.detach() * gradient.detach()).abs().mean(dim=(1,))

    selector = Selector()
    audit, restore = _capture_selected_detector_contribution(selector)
    selected = torch.tensor([[[2.0, 3.0]]], requires_grad=True)
    contribution = selector._selected_detector_contribution(selected, selected.square().sum())
    restore()

    assert torch.allclose(contribution, torch.tensor([[8.0, 18.0]]))
    assert audit[0]["objective"]["finite"] is True
    assert audit[0]["selected_inputs"]["finite"] is True
    assert audit[0]["gradient"]["finite"] is True
    assert audit[0]["contribution"]["finite"] is True


def test_distribution_capture_preserves_finite_contribution_cross_entropy():
    class Selector:
        @staticmethod
        def _contribution_distribution_loss(
            logits, target, valid_mask, teacher_mask, *, temperature
        ):
            valid = valid_mask.to(dtype=torch.bool)
            target = target.to(dtype=logits.dtype).clamp_min(0.0).masked_fill(~valid, 0.0)
            mass = target.sum(dim=1)
            active = teacher_mask.to(dtype=torch.bool) & (mass > 1.0e-8)
            normalized = target / mass[:, None].clamp_min(torch.finfo(target.dtype).eps)
            log_probs = torch.log_softmax(logits.masked_fill(~valid, -torch.finfo(logits.dtype).max) / temperature, dim=1)
            return -(normalized * log_probs).sum(dim=1)[active].mean(), active

    selector = Selector()
    audit, restore = _capture_contribution_distribution_loss(selector)
    loss, active = selector._contribution_distribution_loss(
        torch.zeros((1, 2)),
        torch.ones((1, 2)),
        torch.ones((1, 2), dtype=torch.bool),
        torch.ones((1,), dtype=torch.bool),
        temperature=1.0,
    )
    restore()

    assert torch.allclose(loss, torch.log(torch.tensor(2.0)))
    assert active.tolist() == [True]
    assert audit[0]["normalized_target"]["finite"] is True
    assert audit[0]["log_probs"]["finite"] is True
    assert audit[0]["loss"]["finite"] is True
