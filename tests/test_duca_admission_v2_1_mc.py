from __future__ import annotations

import pytest

import tools.bata.duca_admission_v2_1_mc as mc_module
from tools.bata.duca_admission_v2_1_mc import (
    batch_delete_jackknife,
    certify_mc_error,
    run_prefix_extensible_mc,
)
from tools.bata.duca_admission_v2_1_metrics import METRIC_IDS


def make_replicates(start, stop):
    return [
        {
            metric_id: float(((index + 1) * (metric_index + 3)) % 17 - 8)
            for metric_index, metric_id in enumerate(METRIC_IDS)
        }
        for index in range(start, stop)
    ]


def test_batch_jackknife_really_deletes_one_complete_batch():
    result = batch_delete_jackknife(
        delta_hat={metric_id: 0.0 for metric_id in METRIC_IDS},
        replicates=make_replicates(0, 40),
        exact_zero={metric_id: False for metric_id in METRIC_IDS},
        batch_size=10,
        alpha=0.1,
    )
    assert result["batch_count"] == 4
    assert result["deleted_per_replicate"] == 10
    assert result["remaining_per_replicate"] == 30
    assert all(row["replicate_count"] == 30 for row in result["deletion_results"])


def test_prefix_extension_appends_once_without_regenerating_prefix(monkeypatch):
    calls = []

    def generator(start, stop):
        calls.append((start, stop))
        return make_replicates(start, stop)

    certificate_calls = []

    def controlled_certificate(**_kwargs):
        certificate_calls.append(1)
        passed = len(certificate_calls) == 2
        return {
            "status": "PASSED" if passed else "FAILED_CLOSED",
            "failure_code": None if passed else "MC_UNSTABLE",
            "passed": passed,
            "checks": [],
        }

    monkeypatch.setattr(mc_module, "certify_mc_error", controlled_certificate)
    result = run_prefix_extensible_mc(
        generator=generator,
        delta_hat={metric_id: 0.0 for metric_id in METRIC_IDS},
        exact_zero={metric_id: False for metric_id in METRIC_IDS},
        initial_replicates=40,
        maximum_replicates=80,
        batch_size=10,
        alpha=0.1,
    )
    assert calls == [(0, 40), (40, 80)]
    assert result["extended"] is True
    assert result["replicate_count"] == 80
    assert result["certificate"]["passed"] is True
    assert result["authorization_scope"] == "NONE"


def test_mc_certificate_rejects_nonfinite_half_width():
    full = {
        "mc_required": True,
        "active_metrics": ["M00"],
        "q_plus": 1.0,
        "q_minus": 1.0,
        "scales": {"M00": 1.0},
        "lower": {"M00": -1.0},
        "upper": {"M00": 1.0},
    }
    jackknife = {
        "half_widths_99": {
            "q_plus": float("nan"),
            "q_minus": 0.1,
            "lower": {"M00": 0.1},
            "upper": {"M00": 0.1},
        }
    }
    with pytest.raises(ValueError, match="nonfinite"):
        certify_mc_error(
            full_result=full,
            delta_hat={"M00": 0.0},
            jackknife=jackknife,
        )
