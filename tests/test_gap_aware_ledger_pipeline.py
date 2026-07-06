from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata import run_gap_aware_ledger_pipeline as gas_pipeline
from tools.bata import validate_paction_learned_policy_ledger as validate_ledger


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sample(sample_id: str, p_action: list[float] | None = None) -> dict:
    if p_action is None:
        p_action = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]
    return {
        "sample_id": sample_id,
        "split": "training",
        "dense_len": 8,
        "valid_len": 8,
        "frame_signals": {"p_action": p_action},
        "paction_positive_provenance": {
            "p_action_source": "lowres_action_probe",
            "probe_model": "mobilenetv3_64px",
            "no_gt_generation": True,
            "uses_teacher": False,
            "uses_oracle": False,
            "uses_cache": False,
            "uses_prediction_cache": False,
            "uses_raw_prediction": False,
            "prediction_uses_gt": False,
        },
        "action_target": [0, 1, 0, 1, 0, 1, 0, 0],
        "gt_boundaries": [1, 3, 5],
    }


def test_gap_aware_pipeline_generates_three_named_gas_vt_ledgers_with_separated_sources(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.jsonl"
    checkpoint = tmp_path / "gas_vt_policy.pth"
    checkpoint.write_bytes(b"fake gas vt checkpoint")
    _write_jsonl(
        source,
        [
            _sample("video_test_0001|0", p_action=[0.05, 0.08, 0.03, 0.07, 0.02, 0.06, 0.04, 0.05]),
            _sample("video_test_0002|0", p_action=[0.95, 0.90, 0.92, 0.88, 0.94, 0.91, 0.89, 0.93]),
        ],
    )

    monkeypatch.setattr(gas_pipeline.apply_policy, "load_policy_checkpoint", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bootstrap should be used")))

    summary = gas_pipeline.run_pipeline(
        input_jsonl=source,
        checkpoint_path=checkpoint,
        out_dir=tmp_path / "out",
        fixed_budgets=(2, 4),
        dynamic_target_len=4,
        dynamic_budget_buckets=[2, 4],
        device="cpu",
        allow_bootstrap_for_tests=True,
    )

    assert summary["decision"] == "C3_GAS_VT_LEDGER_PIPELINE_READY"
    assert sorted(summary["ledgers"]) == ["gas_vt_dynamic", "gas_vt_fixed_384", "gas_vt_fixed_768"]
    assert Path(summary["canonical_input_jsonl"]).name == "source.canonical_unique.jsonl"
    assert Path(summary["selection_sample_jsonl"]).name == "source.selection_deploy.jsonl"
    assert Path(summary["metric_sample_jsonl"]).name == "source.canonical_unique.jsonl"
    for name, item in summary["ledgers"].items():
        assert Path(item["ledger_jsonl"]).is_file(), name
        assert item["validation_summary"]["required_policy_source"] == "learned_paction_gas_vt_policy_checkpoint"
        assert item["validation_summary"]["boundary_bracket_support@r1"] is not None
        assert "action_interior_bin_coverage" in item["validation_summary"]
        assert "p_action_rank_spearman" in item["validation_summary"]
        assert "max_hole_by_video_top10" in item["validation_summary"]
    dynamic_summary = summary["ledgers"]["gas_vt_dynamic"]["validation_summary"]
    assert dynamic_summary["require_nonconstant_selected_count"] is True
    assert dynamic_summary["min_selected_count"] != dynamic_summary["max_selected_count"]


def test_validator_reports_gas_vt_extra_metrics_and_writes_csv(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    csv_path = tmp_path / "holes.csv"
    checkpoint = tmp_path / "policy.pth"
    checkpoint.write_bytes(b"checkpoint")
    sha = validate_ledger._sha256_file(checkpoint)
    sample = _sample("video_test_0001|0", p_action=[0.7, 0.9, 0.8, 0.85, 0.75, 0.82, 0.72, 0.1])
    sample["gas_vt_policy"] = {
        "source": "learned_paction_gas_vt_policy_checkpoint",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha,
        "policy_checkpoint_sha256": sha,
        "policy_family": "GAS-VT",
        "uses_uniform_fill": False,
        "uses_uniform_scaffold": False,
        "p_action_provenance": sample["paction_positive_provenance"],
    }
    sample["strategy_selected_positions"] = {"gas_vt_fixed_384": [0, 1, 2, 3, 4, 5, 6]}
    _write_jsonl(samples, [sample])
    _write_jsonl(
        ledger,
        [
            {
                "schema_version": "pc_ot_mras_frontend_value_transport_ledger_v0",
                "sample_id": "video_test_0001|0",
                "selected_positions_unit": "local_dense_index",
                "selected_positions": [0, 1, 2, 3, 4, 5, 6],
                "target_len": 7,
                "selected_count": 7,
                "valid_len": 8,
                "dense_len": 8,
                "deploy_selection_ledger": True,
                "diagnostic_only": False,
                "policy_source": "learned_paction_gas_vt_policy_checkpoint",
                "policy_checkpoint_path": str(checkpoint),
                "policy_checkpoint_sha256": sha,
                "uses_gt": False,
                "uses_teacher": False,
                "uses_oracle": False,
                "uses_cache": False,
                "uses_prediction_cache": False,
                "uses_raw_prediction": False,
                "uses_checkpoint": False,
                "prediction_uses_gt": False,
                "training_only": False,
                "diagnostics": {
                    "uniform_visible_fill_count": 0,
                    "source_strategy": "gas_vt_fixed_384",
                    "policy_source": "learned_paction_gas_vt_policy_checkpoint",
                    "policy_checkpoint_path": str(checkpoint),
                    "policy_checkpoint_sha256": sha,
                    "p_action_provenance": sample["paction_positive_provenance"],
                },
            }
        ],
    )

    summary = validate_ledger.validate_ledger(
        sample_jsonl=samples,
        ledger_jsonl=ledger,
        strategy="gas_vt_fixed_384",
        expected_target_len=7,
        require_selected_count=7,
        require_deployable=True,
        require_policy_source="learned_paction_gas_vt_policy_checkpoint",
        require_checkpoint_path=checkpoint,
        require_checkpoint_sha256=sha,
        require_paction_provenance=True,
        boundary_radii=[1, 2, 4, 8],
        max_hole_top10_csv=csv_path,
    )

    assert summary["boundary_support@r1"] == 1.0
    assert summary["boundary_bracket_support@r1"] == 1.0
    assert summary["action_interior_bin_coverage"] > 0.0
    assert summary["p_action_rank_spearman"] > 0.0
    assert summary["dynamic_budget_entropy"] == 0.0
    assert summary["dynamic_budget_iqr"] == 0.0
    assert summary["meanK_matched_uniform_similarity"] == summary["mean_uniform_similarity"]
    assert summary["max_hole_by_video_top10"][0]["video_name"] == "video_test_0001"
    assert csv_path.read_text(encoding="utf-8").splitlines()[0] == "video_name,sample_id,max_unselected_hole,selected_count,valid_len"
