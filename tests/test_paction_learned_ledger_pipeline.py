from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.bata import run_paction_learned_policy_ledger_pipeline as ledger_pipeline
from tools.bata import train_paction_acquisition_policy as train_policy
from tools.bata import validate_paction_learned_policy_ledger as validate_ledger
from tools.bata import paction_acquisition_policy as policy
from tools.bata import paction_source_samples


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sample_row(sample_id: str, p_action: list[float], action_target: list[float], boundaries: list[int]) -> dict:
    return {
        "sample_id": sample_id,
        "split": "training",
        "dense_len": len(p_action),
        "valid_len": len(p_action),
        "frame_signals": {"p_action": p_action},
        "p_action": p_action,
        "probe_model": "mobilenetv3_64px",
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
        "action_target": action_target,
        "gt_boundaries": boundaries,
        "strategy_selected_positions": {"delta_p_action": [1, 4]},
    }


def _max_unselected_hole(selected: list[int], valid_len: int) -> int:
    selected_set = set(selected)
    current = 0
    max_hole = 0
    for idx in range(valid_len):
        if idx in selected_set:
            max_hole = max(max_hole, current)
            current = 0
        else:
            current += 1
    return max(max_hole, current)


def test_learned_score_decoder_enforces_max_hole_without_uniform_scaffold() -> None:
    frame_values = [0.99, 0.98, 0.97, 0.96, 0.95, 0.01, 0.02, 0.03, 0.04, 0.94, 0.93, 0.92]

    plain_topk = policy.constrained_topk(frame_values, budget=6)
    constrained = policy.constrained_topk(frame_values, budget=6, max_unselected_hole=3)

    assert plain_topk == [0, 1, 2, 3, 4, 9]
    assert _max_unselected_hole(plain_topk, valid_len=len(frame_values)) == 4
    assert _max_unselected_hole(constrained, valid_len=len(frame_values)) <= 3
    assert 8 in constrained
    assert constrained != [0, 2, 4, 6, 8, 10]


def test_policy_row_records_learned_score_constrained_gap_decoder() -> None:
    row = {
        "sample_id": "video_test_0001|0",
        "dense_len": 12,
        "valid_len": 12,
        "strategy_selected_positions": {},
        "frame_signals": {"p_action": [0.1] * 12},
    }

    enriched = policy.add_policy_decision_to_sample_row(
        row,
        frame_values=[0.99, 0.98, 0.97, 0.96, 0.95, 0.01, 0.02, 0.03, 0.04, 0.94, 0.93, 0.92],
        fixed_budget=6,
        dynamic_budget_scores=[1.0],
        dynamic_budget_buckets=[6],
        max_unselected_hole=3,
    )

    selected = enriched["strategy_selected_positions"]["learned_paction_gap_loss_value"]
    assert _max_unselected_hole(selected, valid_len=12) <= 3
    assert enriched["paction_policy"]["max_unselected_hole"] == 3
    assert enriched["paction_policy"]["gap_control"] == "learned_score_constrained_gap_no_uniform_fill"
    assert enriched["paction_policy"]["uses_uniform_scaffold"] is False
    assert enriched["paction_policy"]["uses_uniform_fill"] is False


@pytest.mark.skipif(
    os.name == "nt",
    reason="torch training smoke tests run on the Linux/remote training environment",
)
def test_learned_policy_pipeline_generates_fixed_and_dynamic_strict_ledgers(tmp_path: Path) -> None:
    sample_jsonl = tmp_path / "samples.jsonl"
    checkpoint = tmp_path / "policy.pth"
    rows = [
        _sample_row("video_test_0001|0", [0.05, 0.70, 0.92, 0.20, 0.80, 0.10], [0, 1, 1, 0, 1, 0], [1, 4]),
        _sample_row("video_test_0002|0", [0.10, 0.25, 0.88, 0.91, 0.35, 0.05], [0, 0, 1, 1, 0, 0], [2, 3]),
    ]
    _write_jsonl(sample_jsonl, rows)
    train_policy.run_training(
        sample_jsonl,
        out_dir=tmp_path / "train_out",
        checkpoint_path=checkpoint,
        epochs=1,
        batch_size=2,
        hidden_dim=8,
        num_layers=1,
        fixed_budget=3,
        dynamic_budget_buckets=[2, 3, 4],
        device="cpu",
        seed=0,
    )

    summary = ledger_pipeline.run_pipeline(
        input_jsonl=sample_jsonl,
        checkpoint_path=checkpoint,
        out_dir=tmp_path / "ledgers",
        fixed_budgets=[2, 4],
        dynamic_target_len=4,
        dynamic_budget_buckets=[2, 3, 4],
        device="cpu",
        deploy_selection_ledger=True,
    )

    assert summary["decision"] == "C3_PACTION_LEARNED_POLICY_LEDGER_PIPELINE_READY"
    assert sorted(summary["ledgers"]) == ["learned_dynamic", "learned_fixed_2", "learned_fixed_4"]
    canonical_jsonl = tmp_path / "ledgers" / "source.canonical_unique.jsonl"
    selection_jsonl = tmp_path / "ledgers" / "source.selection_deploy.jsonl"
    assert summary["source_canonicalization"]["input_jsonl"] == str(sample_jsonl)
    assert summary["source_canonicalization"]["output_jsonl"] == str(canonical_jsonl)
    assert summary["selection_sample_jsonl"] == str(selection_jsonl)
    assert summary["selection_source_report"]["output_jsonl"] == str(selection_jsonl)
    assert summary["metric_sample_jsonl"] == str(canonical_jsonl)
    selection_rows = _read_jsonl(selection_jsonl)
    assert all("action_target" not in row for row in selection_rows)
    assert all("gt_boundaries" not in row for row in selection_rows)
    assert all("frame_signals" in row for row in selection_rows)
    for name, item in summary["ledgers"].items():
        ledger_path = Path(item["ledger_jsonl"])
        sample_path = Path(item["sample_jsonl"])
        assert ledger_path.is_file(), name
        assert sample_path.is_file(), name
        sample_rows = _read_jsonl(sample_path)
        assert all("action_target" not in row for row in sample_rows)
        assert all("gt_boundaries" not in row for row in sample_rows)
        assert item["validation_summary"]["metric_sample_jsonl"] == str(canonical_jsonl)
        ledger_rows = _read_jsonl(ledger_path)
        assert ledger_rows
        assert all(row["deploy_selection_ledger"] is True for row in ledger_rows)
        assert all(row["diagnostics"]["uniform_visible_fill_count"] == 0 for row in ledger_rows)
        assert item["validation_summary"]["uses_uniform_fill"] is False
        assert "max_gap" in item["validation_summary"]
        assert "p95_gap" in item["validation_summary"]
        assert "max_unselected_hole" in item["validation_summary"]
        assert "p95_unselected_hole" in item["validation_summary"]
        assert "max_uniform_similarity" in item["validation_summary"]
        assert "boundary_support_r1" in item["validation_summary"]
        assert "action_positive_coverage" in item["validation_summary"]


def test_learned_policy_ledger_validator_rejects_uniform_fill_and_forbidden_flags(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(
        samples,
        [_sample_row("video_test_0001|0", [0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1], [1, 3])],
    )
    _write_jsonl(
        ledger,
        [
            {
                "schema_version": "pc_ot_mras_frontend_value_transport_ledger_v0",
                "sample_id": "video_test_0001|0",
                "selected_positions_unit": "local_dense_index",
                "selected_positions": [1, 3],
                "target_len": 2,
                "selected_count": 2,
                "valid_len": 4,
                "dense_len": 4,
                "deploy_selection_ledger": True,
                "diagnostic_only": False,
                "uses_gt": False,
                "uses_teacher": False,
                "uses_oracle": False,
                "uses_cache": False,
                "uses_raw_prediction": False,
                "uses_checkpoint": False,
                "diagnostics": {"uniform_visible_fill_count": 1, "source_strategy": "learned_paction_gap_loss_value"},
            }
        ],
    )

    with pytest.raises(ValueError, match="uniform_visible_fill_count"):
        validate_ledger.validate_ledger(
            sample_jsonl=samples,
            ledger_jsonl=ledger,
            strategy="learned_paction_gap_loss_value",
            expected_target_len=2,
            require_selected_count=2,
            require_deployable=True,
        )


def test_learned_policy_ledger_validator_rejects_bootstrap_policy_source(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    checkpoint = tmp_path / "policy.pth"
    checkpoint.write_bytes(b"fake checkpoint bytes")
    _write_jsonl(
        samples,
        [
            {
                **_sample_row("video_test_0001|0", [0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1], [1, 3]),
                "paction_policy": {
                    "source": "bootstrap_paction_gap_loss_surrogate_policy",
                    "checkpoint_path": str(checkpoint),
                    "uses_uniform_fill": False,
                    "uses_uniform_scaffold": False,
                },
            }
        ],
    )
    _write_jsonl(
        ledger,
        [
            {
                "schema_version": "pc_ot_mras_frontend_value_transport_ledger_v0",
                "sample_id": "video_test_0001|0",
                "selected_positions_unit": "local_dense_index",
                "selected_positions": [1, 3],
                "target_len": 2,
                "selected_count": 2,
                "valid_len": 4,
                "dense_len": 4,
                "deploy_selection_ledger": True,
                "diagnostic_only": False,
                "policy_source": "learned_paction_gap_loss_policy_checkpoint",
                "policy_checkpoint_path": str(checkpoint),
                "policy_checkpoint_sha256": "2f05d4b689d2705f6f780e9304eb440953164c179f7f42c7eac98227cd3fc60c",
                "uses_gt": False,
                "uses_teacher": False,
                "uses_oracle": False,
                "uses_cache": False,
                "uses_raw_prediction": False,
                "uses_checkpoint": False,
                "diagnostics": {
                    "uniform_visible_fill_count": 0,
                    "source_strategy": "learned_paction_gap_loss_value",
                    "policy_source": "learned_paction_gap_loss_policy_checkpoint",
                    "policy_checkpoint_path": str(checkpoint),
                    "policy_checkpoint_sha256": "2f05d4b689d2705f6f780e9304eb440953164c179f7f42c7eac98227cd3fc60c",
                },
            }
        ],
    )

    with pytest.raises(ValueError, match="paction_policy.source"):
        validate_ledger.validate_ledger(
            sample_jsonl=samples,
            ledger_jsonl=ledger,
            strategy="learned_paction_gap_loss_value",
            expected_target_len=2,
            require_selected_count=2,
            require_deployable=True,
            require_policy_source="learned_paction_gap_loss_policy_checkpoint",
            require_checkpoint_path=checkpoint,
        )


def test_learned_policy_ledger_validator_reports_multiple_boundary_radii(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(
        samples,
        [_sample_row("video_test_0001|0", [0.1, 0.9, 0.2, 0.8, 0.1, 0.1], [0, 1, 0, 1, 0, 0], [1, 5])],
    )
    _write_jsonl(
        ledger,
        [
            {
                "schema_version": "pc_ot_mras_frontend_value_transport_ledger_v0",
                "sample_id": "video_test_0001|0",
                "selected_positions_unit": "local_dense_index",
                "selected_positions": [1, 3],
                "target_len": 2,
                "selected_count": 2,
                "valid_len": 6,
                "dense_len": 6,
                "deploy_selection_ledger": True,
                "diagnostic_only": False,
                "uses_gt": False,
                "uses_teacher": False,
                "uses_oracle": False,
                "uses_cache": False,
                "uses_raw_prediction": False,
                "uses_checkpoint": False,
                "diagnostics": {"uniform_visible_fill_count": 0, "source_strategy": "learned_paction_gap_loss_value"},
            }
        ],
    )

    summary = validate_ledger.validate_ledger(
        sample_jsonl=samples,
        ledger_jsonl=ledger,
        strategy="learned_paction_gap_loss_value",
        expected_target_len=2,
        require_selected_count=2,
        require_deployable=True,
        boundary_radii=[1, 2, 4, 8],
    )

    assert summary["boundary_support_r1"] == 0.5
    assert summary["boundary_support_r2"] == 1.0
    assert summary["boundary_support_r4"] == 1.0
    assert summary["boundary_support_r8"] == 1.0
    assert summary["max_unselected_hole"] == 2
    assert summary["p95_unselected_hole"] == 2.0


def test_learned_policy_ledger_validator_rejects_uniform_like_scaffold_pattern(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(
        samples,
        [_sample_row("video_test_0001|0", [0.1, 0.9, 0.2, 0.8, 0.1, 0.1], [0, 1, 0, 1, 0, 0], [1, 5])],
    )
    _write_jsonl(
        ledger,
        [
            {
                "schema_version": "pc_ot_mras_frontend_value_transport_ledger_v0",
                "sample_id": "video_test_0001|0",
                "selected_positions_unit": "local_dense_index",
                "selected_positions": [0, 3],
                "target_len": 2,
                "selected_count": 2,
                "valid_len": 6,
                "dense_len": 6,
                "deploy_selection_ledger": True,
                "diagnostic_only": False,
                "uses_gt": False,
                "uses_teacher": False,
                "uses_oracle": False,
                "uses_cache": False,
                "uses_raw_prediction": False,
                "uses_checkpoint": False,
                "diagnostics": {"uniform_visible_fill_count": 0, "source_strategy": "learned_paction_gap_loss_value"},
            }
        ],
    )

    with pytest.raises(ValueError, match="uniform similarity"):
        validate_ledger.validate_ledger(
            sample_jsonl=samples,
            ledger_jsonl=ledger,
            strategy="learned_paction_gap_loss_value",
            expected_target_len=2,
            require_selected_count=2,
            require_deployable=True,
            max_uniform_similarity=0.99,
        )


def test_learned_policy_ledger_validator_requires_paction_positive_provenance(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    row = {
        **_sample_row("video_test_0001|0", [0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1], [1, 3]),
        "paction_policy": {
            "source": "learned_paction_gap_loss_policy_checkpoint",
            "checkpoint_path": str(tmp_path / "policy.pth"),
            "checkpoint_sha256": "a" * 64,
            "uses_uniform_fill": False,
            "uses_uniform_scaffold": False,
        },
    }
    _write_jsonl(samples, [row])
    _write_jsonl(
        ledger,
        [
            {
                "schema_version": "pc_ot_mras_frontend_value_transport_ledger_v0",
                "sample_id": "video_test_0001|0",
                "selected_positions_unit": "local_dense_index",
                "selected_positions": [1, 3],
                "target_len": 2,
                "selected_count": 2,
                "valid_len": 4,
                "dense_len": 4,
                "deploy_selection_ledger": True,
                "diagnostic_only": False,
                "uses_gt": False,
                "uses_teacher": False,
                "uses_oracle": False,
                "uses_cache": False,
                "uses_raw_prediction": False,
                "uses_checkpoint": False,
                "diagnostics": {"uniform_visible_fill_count": 0, "source_strategy": "learned_paction_gap_loss_value"},
            }
        ],
    )

    with pytest.raises(ValueError, match="p_action positive provenance"):
        validate_ledger.validate_ledger(
            sample_jsonl=samples,
            ledger_jsonl=ledger,
            strategy="learned_paction_gap_loss_value",
            expected_target_len=2,
            require_selected_count=2,
            require_deployable=True,
            require_paction_provenance=True,
        )

    row["paction_policy"]["p_action_provenance"] = {
        "p_action_source": "lowres_action_probe",
        "probe_model": "mobilenetv3",
        "no_gt_generation": True,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "prediction_uses_gt": False,
    }
    _write_jsonl(samples, [row])
    summary = validate_ledger.validate_ledger(
        sample_jsonl=samples,
        ledger_jsonl=ledger,
        strategy="learned_paction_gap_loss_value",
        expected_target_len=2,
        require_selected_count=2,
        require_deployable=True,
        require_paction_provenance=True,
    )
    assert summary["paction_provenance_verified"] is True


def test_source_sample_canonicalization_drops_identical_duplicates_and_rejects_conflicts(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "source.jsonl"
    canonical_jsonl = tmp_path / "source.canonical_unique.jsonl"
    report_json = tmp_path / "dedup_report.json"
    row = _sample_row("video_test_0001|0", [0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1], [1, 3])
    _write_jsonl(input_jsonl, [row, dict(row)])

    report = paction_source_samples.canonicalize_unique_sample_jsonl(
        input_jsonl,
        canonical_jsonl,
        report_json=report_json,
        split="test",
    )

    assert report["input_rows"] == 2
    assert report["output_rows"] == 1
    assert report["duplicate_count"] == 1
    assert report["conflicting_duplicate_count"] == 0
    assert _read_jsonl(canonical_jsonl) == [row]
    assert json.loads(report_json.read_text(encoding="utf-8"))["duplicates"][0]["sample_id"] == "video_test_0001|0"

    conflicting_jsonl = tmp_path / "source.conflicting.jsonl"
    conflict = dict(row)
    conflict["valid_len"] = 3
    _write_jsonl(conflicting_jsonl, [row, conflict])

    with pytest.raises(ValueError, match="conflicting duplicate sample_id"):
        paction_source_samples.canonicalize_unique_sample_jsonl(
            conflicting_jsonl,
            tmp_path / "should_not_exist.jsonl",
            split="test",
        )


def test_source_sample_canonicalization_treats_provenance_as_canonical(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "source.jsonl"
    row = _sample_row("video_test_0001|0", [0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1], [1, 3])
    conflict = dict(row)
    conflict["paction_positive_provenance"] = dict(row["paction_positive_provenance"], probe_model="other_model")
    _write_jsonl(input_jsonl, [row, conflict])

    with pytest.raises(ValueError, match="conflicting duplicate sample_id"):
        paction_source_samples.canonicalize_unique_sample_jsonl(
            input_jsonl,
            tmp_path / "should_not_exist.jsonl",
            split="test",
        )


@pytest.mark.parametrize("flag", ["uses_gt", "uses_gt_for_diagnostics", "diagnostic_only", "training_only"])
def test_paction_positive_provenance_rejects_generation_leak_flags(flag: str) -> None:
    row = _sample_row("video_test_0001|0", [0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1], [1, 3])
    provenance = dict(row["paction_positive_provenance"])
    provenance[flag] = True

    with pytest.raises(ValueError, match=flag):
        paction_source_samples.validate_paction_positive_provenance(provenance, source_name="unit-test")


def test_paction_policy_trainer_defaults_to_training_split_guard() -> None:
    args = train_policy.build_arg_parser().parse_args(
        [
            "--train-jsonl",
            "train.samples.jsonl",
            "--out-dir",
            "out",
        ]
    )

    assert args.expected_split == "training"


@pytest.mark.parametrize("flag", ["uses_gt", "training_only", "uses_teacher", "uses_cache"])
def test_selection_deploy_source_rejects_true_generation_or_cache_flags_before_strip(tmp_path: Path, flag: str) -> None:
    input_jsonl = tmp_path / "source.jsonl"
    selection_jsonl = tmp_path / "source.selection_deploy.jsonl"
    row = _sample_row("video_test_0001|0", [0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1], [1, 3])
    row[flag] = True
    _write_jsonl(input_jsonl, [row])

    with pytest.raises(ValueError, match=flag):
        paction_source_samples.write_deploy_selection_source_jsonl(input_jsonl, selection_jsonl)


def test_selection_deploy_source_strips_diagnostic_flags_and_can_infer_legacy_provenance(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "source.jsonl"
    selection_jsonl = tmp_path / "source.selection_deploy.jsonl"
    row = _sample_row("video_test_0001|0", [0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1], [1, 3])
    row.pop("paction_positive_provenance")
    row["probe_model"] = "mobilenetv3_64px"
    row["uses_gt_for_diagnostics"] = True
    row["diagnostic_only"] = True
    _write_jsonl(input_jsonl, [row])

    with pytest.raises(ValueError, match="p_action positive provenance is required"):
        paction_source_samples.write_deploy_selection_source_jsonl(input_jsonl, selection_jsonl)

    report = paction_source_samples.write_deploy_selection_source_jsonl(
        input_jsonl,
        selection_jsonl,
        allow_inferred_paction_positive_provenance=True,
    )
    rows = _read_jsonl(selection_jsonl)

    assert report["inferred_paction_positive_provenance_count"] == 1
    assert report["stripped_true_diagnostic_flag_counts"] == {
        "diagnostic_only": 1,
        "uses_gt_for_diagnostics": 1,
    }
    assert "uses_gt_for_diagnostics" not in rows[0]
    assert "diagnostic_only" not in rows[0]
    assert rows[0]["paction_positive_provenance"]["inferred_from_source_row"] is True
    assert rows[0]["paction_positive_provenance"]["probe_model"] == "mobilenetv3_64px"
    assert rows[0]["paction_positive_provenance"]["no_gt_generation"] is True


def test_selection_deploy_source_strips_metric_payload_without_laundering_flags(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "source.jsonl"
    selection_jsonl = tmp_path / "source.selection_deploy.jsonl"
    report_json = tmp_path / "selection.report.json"
    row = _sample_row("video_test_0001|0", [0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1], [1, 3])
    row["deploy_selection_ledger"] = False
    _write_jsonl(input_jsonl, [row])

    report = paction_source_samples.write_deploy_selection_source_jsonl(
        input_jsonl,
        selection_jsonl,
        report_json=report_json,
        split="test",
    )
    rows = _read_jsonl(selection_jsonl)

    assert report["input_rows"] == 1
    assert report["output_rows"] == 1
    assert rows[0]["frame_signals"] == {"p_action": [0.1, 0.9, 0.2, 0.8]}
    assert "action_target" not in rows[0]
    assert "gt_boundaries" not in rows[0]
    assert "uses_gt_for_diagnostics" not in rows[0]
    assert "diagnostic_only" not in rows[0]
    assert "deploy_selection_ledger" not in rows[0]
    assert rows[0]["paction_positive_provenance"]["probe_model"] == "mobilenetv3_64px"

    policy_jsonl = tmp_path / "source.policy.jsonl"
    summary = ledger_pipeline.apply_policy.run_policy_application(
        selection_jsonl,
        policy_jsonl,
        fixed_budget=2,
        device="cpu",
        strict_deploy_source=True,
        strip_deploy_invisible_payload=True,
        allow_bootstrap_for_tests=True,
    )
    assert summary["strict_deploy_source"] is True

    bad = dict(row, uses_gt=True)
    _write_jsonl(input_jsonl, [bad])
    with pytest.raises(ValueError, match="forbidden strict deploy p_action source"):
        paction_source_samples.write_deploy_selection_source_jsonl(input_jsonl, selection_jsonl)


def test_deploy_conversion_requires_checkpoint_policy_metadata(tmp_path: Path) -> None:
    sample_jsonl = tmp_path / "samples.policy.jsonl"
    ledger_jsonl = tmp_path / "ledger.jsonl"
    row = {
        **_sample_row("video_test_0001|0", [0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1], [1, 3]),
        "strategy_selected_positions": {"learned_paction_gap_loss_value": [1, 3]},
    }
    _write_jsonl(sample_jsonl, [row])

    with pytest.raises(ValueError, match="paction_policy metadata is required"):
        ledger_pipeline.convert_ledger.run_conversion(
            sample_jsonl,
            ledger_jsonl,
            strategy="learned_paction_gap_loss_value",
            target_len=2,
            require_selected_count=2,
            deploy_selection_ledger=True,
        )


def test_policy_training_expected_split_rejects_validation_rows() -> None:
    row = _sample_row("video_test_0001|0", [0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1], [1, 3])
    row["split"] = "validation"

    with pytest.raises(ValueError, match="expected split"):
        train_policy._prepared_rows([row], dynamic_budget_buckets=[2], expected_split="training")


def test_pipeline_canonicalizes_source_before_policy_application(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_jsonl = tmp_path / "source.jsonl"
    checkpoint = tmp_path / "policy.pth"
    checkpoint.write_bytes(b"checkpoint")
    row = _sample_row("video_test_0001|0", [0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1], [1, 3])
    _write_jsonl(source_jsonl, [row, dict(row)])
    applied_inputs: list[Path] = []
    metric_inputs: list[Path] = []

    def fake_apply(input_jsonl, output_jsonl, **kwargs):
        applied_inputs.append(Path(input_jsonl))
        source_rows = _read_jsonl(Path(input_jsonl))
        assert len(source_rows) == 1
        assert "action_target" not in source_rows[0]
        assert "gt_boundaries" not in source_rows[0]
        assert kwargs["strict_deploy_source"] is True
        out_rows = []
        for item in source_rows:
            enriched = dict(item)
            enriched["strategy_selected_positions"] = {
                "learned_paction_gap_loss_value": [1, 3],
                "learned_paction_gap_loss_dynamic_budget": [1, 3],
            }
            enriched["paction_policy"] = {
                "source": "learned_paction_gap_loss_policy_checkpoint",
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": "a" * 64,
                "uses_uniform_fill": False,
                "uses_uniform_scaffold": False,
                "p_action_provenance": item["paction_positive_provenance"],
            }
            out_rows.append(enriched)
        _write_jsonl(Path(output_jsonl), out_rows)
        return {"decision": "fake"}

    def fake_convert(sample_jsonl, ledger_jsonl, **kwargs):
        rows = _read_jsonl(Path(sample_jsonl))
        _write_jsonl(
            Path(ledger_jsonl),
            [
                {
                    "sample_id": item["sample_id"],
                    "selected_positions": [1, 3],
                    "selected_count": 2,
                    "target_len": kwargs["target_len"],
                    "valid_len": item["valid_len"],
                    "dense_len": item["dense_len"],
                    "deploy_selection_ledger": True,
                    "diagnostic_only": False,
                    "diagnostics": {"uniform_visible_fill_count": 0},
                }
                for item in rows
            ],
        )
        return {"output_rows": len(rows)}

    def fake_validate(*, metric_sample_jsonl, **kwargs):
        metric_inputs.append(Path(metric_sample_jsonl))
        assert Path(metric_sample_jsonl).name == "source.canonical_unique.jsonl"
        return {
            "min_selected_count": 2,
            "max_selected_count": 2,
            "boundary_support_r1": 1.0,
            "uses_uniform_fill": False,
            "uses_uniform_scaffold": False,
        }

    monkeypatch.setattr(ledger_pipeline.apply_policy, "_sha256_file", lambda path: "a" * 64)
    monkeypatch.setattr(ledger_pipeline.apply_policy, "run_policy_application", fake_apply)
    monkeypatch.setattr(ledger_pipeline.convert_ledger, "run_conversion", fake_convert)
    monkeypatch.setattr(ledger_pipeline.validate_ledger, "validate_ledger", fake_validate)

    summary = ledger_pipeline.run_pipeline(
        input_jsonl=source_jsonl,
        checkpoint_path=checkpoint,
        out_dir=tmp_path / "out",
        fixed_budgets=[2],
        dynamic_target_len=2,
        dynamic_budget_buckets=[2],
        device="cpu",
    )

    assert applied_inputs
    assert applied_inputs[0].name == "source.selection_deploy.jsonl"
    assert set(metric_inputs) == {tmp_path / "out" / "source.canonical_unique.jsonl"}
    assert summary["selection_sample_jsonl"] == str(tmp_path / "out" / "source.selection_deploy.jsonl")
    assert summary["source_canonicalization"]["duplicate_count"] == 1
