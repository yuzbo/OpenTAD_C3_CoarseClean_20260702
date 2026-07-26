import torch
import pytest

from tools.bata.diagnose_duca_stage2_postupdate_nonfinite import (
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
