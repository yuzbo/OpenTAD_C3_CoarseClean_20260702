import torch

from tools.bata.diagnose_duca_stage2_postupdate_nonfinite import (
    _optimizer_step_ran,
    _summarize_named_tensors,
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
