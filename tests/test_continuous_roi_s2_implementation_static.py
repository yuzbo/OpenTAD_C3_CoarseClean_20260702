import copy

import pytest

from tools.bata.validate_continuous_roi_s2_implementation import (
    _strip_u128_wrapper_fields,
    validate_implementation,
)


def test_continuous_roi_s2_static_implementation_passes_fail_closed():
    audit = validate_implementation()
    assert audit["status"] == "PASS"
    assert audit["u128_selector_parameters"] == 0
    assert audit["u128_new_parameters"] == 609449
    assert audit["official_test_materialized"] is False
    assert audit["training_authorized"] is False
    assert audit["full_model_cuda_gate_required"] is True


def test_wrapper_field_stripping_does_not_mutate_input():
    value = {
        "backbone": {
            "custom": {
                "wrapper_type": "continuous_roi_common_support_u128",
                "continuous_roi_knots": 12,
                "native_crop_chunk_num": 48,
                "pretrain": "checkpoint.pth",
            }
        }
    }
    original = copy.deepcopy(value)
    stripped = _strip_u128_wrapper_fields(value)
    assert value == original
    assert stripped == {"backbone": {"custom": {"pretrain": "checkpoint.pth"}}}


def test_static_validator_fails_when_official_test_is_materialized(monkeypatch):
    from tools.bata import validate_continuous_roi_s2_implementation as module

    original = module.Config.fromfile

    def mutated(path):
        cfg = original(path)
        if "continuous_roi_s2_" in str(path):
            cfg.dataset.test = copy.deepcopy(cfg.dataset.val)
        return cfg

    monkeypatch.setattr(module.Config, "fromfile", mutated)
    with pytest.raises(ValueError, match="official test"):
        module.validate_implementation()
