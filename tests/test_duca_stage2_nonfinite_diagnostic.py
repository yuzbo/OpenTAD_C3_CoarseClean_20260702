import torch

from tools.bata.diagnose_duca_stage2_nonfinite_loss import (
    _finite_tensor_summary,
    _normalize_state_dict,
)


def test_finite_tensor_summary_reports_nan_and_infinities_without_json_nan():
    summary = _finite_tensor_summary(torch.tensor([1.0, float("nan"), float("inf"), float("-inf")]))

    assert summary["finite"] is False
    assert summary["finite_count"] == 1
    assert summary["nan_count"] == 1
    assert summary["posinf_count"] == 1
    assert summary["neginf_count"] == 1
    assert summary["finite_min"] == 1.0
    assert summary["finite_max"] == 1.0


def test_normalize_state_dict_drops_only_leading_ddp_prefix():
    state = {"module.weight": torch.tensor(1.0), "model.module.bias": torch.tensor(2.0)}

    normalized = _normalize_state_dict(state)

    assert set(normalized) == {"weight", "model.module.bias"}
