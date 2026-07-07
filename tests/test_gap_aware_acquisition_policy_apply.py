from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata import apply_gap_aware_acquisition_policy as apply_gas_vt


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _provenance() -> dict:
    return {
        "p_action_source": "lowres_action_probe",
        "probe_model": "mobilenetv3_64px",
        "no_gt_generation": True,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "prediction_uses_gt": False,
    }


def test_gas_vt_apply_requires_checkpoint_by_default(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "samples.gas_vt.jsonl"
    input_jsonl.write_text(
        json.dumps({"sample_id": "video_test_0001|0", "dense_len": 4, "valid_len": 4, "frame_signals": {"p_action": [0.1, 0.9, 0.2, 0.8]}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checkpoint_path is required"):
        apply_gas_vt.run_policy_application(input_jsonl, output_jsonl)


def test_gas_vt_apply_rejects_bootstrap_when_checkpoint_is_present(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "samples.gas_vt.jsonl"
    checkpoint = tmp_path / "policy.pth"
    checkpoint.write_bytes(b"fake checkpoint")
    input_jsonl.write_text(
        json.dumps({"sample_id": "video_test_0001|0", "dense_len": 4, "valid_len": 4, "frame_signals": {"p_action": [0.1, 0.9, 0.2, 0.8]}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        apply_gas_vt.run_policy_application(
            input_jsonl,
            output_jsonl,
            checkpoint_path=checkpoint,
            allow_bootstrap_for_tests=True,
        )


def test_gas_vt_apply_emits_deploy_safe_metadata_and_strips_invisible_payload(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "samples.gas_vt.jsonl"
    summary_json = tmp_path / "summary.json"
    input_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "video_test_0001|0",
                "dense_len": 8,
                "valid_len": 8,
                "frame_signals": {"p_action": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]},
                "paction_positive_provenance": _provenance(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = apply_gas_vt.run_policy_application(
        input_jsonl,
        output_jsonl,
        summary_json=summary_json,
        fixed_budgets=(3, 5),
        dynamic_budget_buckets=[2, 4, 6],
        strict_deploy_source=True,
        strip_deploy_invisible_payload=True,
        allow_bootstrap_for_tests=True,
        source_jsonl_for_hash=input_jsonl,
    )
    rows = _read_jsonl(output_jsonl)

    assert summary["decision"] == "C3_GAS_VT_POLICY_APPLICATION_READY"
    assert summary["policy_family"] == "GAS-VT"
    assert summary["source_jsonl_sha256"]
    assert rows[0]["deploy_invisible_payload_stripped"] is True
    assert "frame_signals" not in rows[0]
    assert "p_action" not in rows[0]
    strategies = rows[0]["strategy_selected_positions"]
    assert sorted(key for key in strategies if key.startswith("gas_vt_")) == [
        "gas_vt_dynamic",
        "gas_vt_fixed_384",
        "gas_vt_fixed_768",
    ]
    meta = rows[0]["gas_vt_policy"]
    assert meta["source"] == "bootstrap_gas_vt_surrogate_policy"
    assert meta["policy_family"] == "GAS-VT"
    assert meta["decode_mode"] == "hard_gap_aware_topk"
    assert meta["uses_uniform_fill"] is False
    assert meta["uses_uniform_scaffold"] is False
    assert meta["source_jsonl_sha256"] == summary["source_jsonl_sha256"]


def test_gas_vt_checkpoint_apply_scores_each_strategy_with_its_target_budget(tmp_path: Path, monkeypatch) -> None:
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "samples.gas_vt.jsonl"
    checkpoint = tmp_path / "policy.pth"
    checkpoint.write_bytes(b"fake checkpoint")
    input_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "video_test_0001|0",
                "dense_len": 8,
                "valid_len": 8,
                "frame_signals": {"p_action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
                "paction_positive_provenance": _provenance(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    seen_budgets: list[int | None] = []

    monkeypatch.setattr(
        apply_gas_vt,
        "load_policy_checkpoint",
        lambda *args, **kwargs: (object(), {"dynamic_budget_buckets": [2, 4, 6]}),
    )

    def fake_checkpoint_scores(_model, p_action, *, valid, target_budget, device):
        seen_budgets.append(target_budget)
        frame_values = [float(target_budget or 0) + float(idx) / 100.0 for idx, _ in enumerate(p_action)]
        budget_scores = [0.0, 0.0, 1.0]
        return frame_values, budget_scores

    monkeypatch.setattr(apply_gas_vt, "checkpoint_policy_scores", fake_checkpoint_scores)

    apply_gas_vt.run_policy_application(
        input_jsonl,
        output_jsonl,
        fixed_budgets=(3, 5),
        checkpoint_path=checkpoint,
        device="cpu",
    )
    row = _read_jsonl(output_jsonl)[0]

    assert seen_budgets == [6, 3, 5, 6]
    assert row["gas_vt_policy"]["apply_time_target_budgets"] == {
        "gas_vt_fixed_384": 3,
        "gas_vt_fixed_768": 5,
        "gas_vt_dynamic": 6,
    }
    assert row["gas_vt_policy"]["budget_conditioning_rule"] == "checkpoint_two_pass_strategy_specific_target_budget"
    assert row["gas_vt_policy"]["budget_conditioned_frame_values"] is True


def test_gas_vt_apply_rejects_gt_payload_in_strict_source(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "samples.gas_vt.jsonl"
    row = {
        "sample_id": "video_test_0001|0",
        "dense_len": 4,
        "valid_len": 4,
        "frame_signals": {"p_action": [0.1, 0.9, 0.2, 0.8]},
        "paction_positive_provenance": _provenance(),
        "action_target": [0, 1, 0, 1],
        "uses_gt_for_diagnostics": True,
    }
    input_jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden strict deploy p_action source"):
        apply_gas_vt.run_policy_application(
            input_jsonl,
            output_jsonl,
            strict_deploy_source=True,
            allow_bootstrap_for_tests=True,
        )
