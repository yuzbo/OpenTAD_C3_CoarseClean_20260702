from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata import validate_sparse_tad_claim_budget as gate


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_claim_budget_gate_accepts_fixed384_summary(tmp_path: Path) -> None:
    summary = tmp_path / "fixed384.validation.json"
    _write_json(
        summary,
        {
            "strategy": "detector_aware_fixed_384",
            "target_len": 384,
            "max_selected_count": 384,
            "selected_count_histogram": {"384": 12},
            "uses_teacher": False,
            "uses_gt": False,
        },
    )

    report = gate.build_claim_budget_report(ledger_summaries=[summary])

    assert report["decision"] == "C3_SPARSE_TAD_CLAIM_BUDGET_GATE_PASS"
    assert report["budget_contract"]["fixed_768_and_dynamic_768_are_diagnostic_only"] is True


def test_claim_budget_gate_rejects_768_or_dynamic_over_budget(tmp_path: Path) -> None:
    summary = tmp_path / "dynamic.validation.json"
    _write_json(
        summary,
        {
            "strategy": "detector_aware_dynamic",
            "target_len": 768,
            "max_selected_count": 512,
            "selected_count_histogram": {"384": 3, "512": 2},
        },
    )

    with pytest.raises(AssertionError, match="exceeds claim budget"):
        gate.build_claim_budget_report(ledger_summaries=[summary])


def test_claim_budget_gate_rejects_forbidden_claim_flags(tmp_path: Path) -> None:
    summary = tmp_path / "bad.validation.json"
    _write_json(
        summary,
        {
            "strategy": "detector_aware_fixed_384",
            "target_len": 384,
            "max_selected_count": 384,
            "uses_teacher": True,
        },
    )

    with pytest.raises(AssertionError, match="uses_teacher"):
        gate.build_claim_budget_report(ledger_summaries=[summary])


def test_claim_budget_gate_rejects_missing_selected_count_evidence(tmp_path: Path) -> None:
    summary = tmp_path / "target-only.validation.json"
    _write_json(
        summary,
        {
            "strategy": "detector_aware_fixed_384",
            "target_len": 384,
        },
    )

    with pytest.raises(AssertionError, match="selected-count evidence"):
        gate.build_claim_budget_report(ledger_summaries=[summary])


def test_claim_budget_gate_checks_spec_and_claim_manifest(tmp_path: Path) -> None:
    spec = tmp_path / "fixed384.spec.json"
    manifest = tmp_path / "fixed384.claim_manifest.json"
    _write_json(
        spec,
        {
            "variant": "detector_aware_fixed_384",
            "target": 384,
            "required_selected_count": 384,
        },
    )
    _write_json(
        manifest,
        {
            "claim_name": "paper_main_fixed_384",
            "variant_name": "detector_aware_fixed_384",
            "paper_main_claim": True,
            "max_selected_count": 384,
            "selected_count_histogram": {"384": 2},
        },
    )

    report = gate.build_claim_budget_report(specs=[spec], claim_manifests=[manifest])

    assert report["decision"] == gate.READY
    assert report["evidence_reports"][0]["source_type"] == "spec"
    assert report["evidence_reports"][1]["source_type"] == "claim_manifest"


def test_claim_budget_gate_rejects_dynamic_bucket_over_budget(tmp_path: Path) -> None:
    manifest = tmp_path / "dynamic.claim_manifest.json"
    _write_json(
        manifest,
        {
            "variant_name": "detector_aware_dynamic",
            "dynamic_budget_buckets": [128, 256, 384, 512],
            "dynamic_target": 512,
            "selected_count_histogram": {"256": 2, "512": 1},
        },
    )

    with pytest.raises(AssertionError, match="exceeds claim budget"):
        gate.build_claim_budget_report(claim_manifests=[manifest])


def test_claim_budget_gate_rejects_dynamic_variant_for_paper_main_even_with_384_ceiling(tmp_path: Path) -> None:
    manifest = tmp_path / "dynamic_384.claim_manifest.json"
    _write_json(
        manifest,
        {
            "variant_name": "detector_aware_dynamic",
            "dynamic_budget_buckets": [128, 256, 384],
            "max_selected_count": 384,
            "selected_count_histogram": {"128": 1, "256": 1, "384": 1},
        },
    )

    with pytest.raises(AssertionError, match="diagnostic"):
        gate.build_claim_budget_report(claim_manifests=[manifest])


def test_claim_budget_gate_marks_diagnostic_without_paper_main_pass(tmp_path: Path) -> None:
    manifest = tmp_path / "fixed768_diagnostic.claim_manifest.json"
    _write_json(
        manifest,
        {
            "variant_name": "detector_aware_fixed_768",
            "claim_mode": "diagnostic",
            "diagnostic_only": True,
            "target_len": 768,
            "max_selected_count": 768,
            "selected_count_histogram": {"768": 1},
        },
    )

    report = gate.build_claim_budget_report(claim_manifests=[manifest], claim_mode="diagnostic")

    assert report["decision"] == gate.DIAGNOSTIC_ONLY
    assert report["paper_main_claim_allowed"] is False
    assert report["diagnostic_only"] is True


def test_claim_budget_gate_rejects_diagnostic_for_paper_main(tmp_path: Path) -> None:
    manifest = tmp_path / "diagnostic.claim_manifest.json"
    _write_json(
        manifest,
        {
            "variant_name": "detector_aware_fixed_384",
            "diagnostic_only": True,
            "target_len": 384,
            "max_selected_count": 384,
            "selected_count_histogram": {"384": 1},
        },
    )

    with pytest.raises(AssertionError, match="diagnostic"):
        gate.build_claim_budget_report(claim_manifests=[manifest])


def test_claim_budget_gate_rejects_paction_lattice_for_paper_main_even_at_384(tmp_path: Path) -> None:
    manifest = tmp_path / "paction_lattice.claim_manifest.json"
    _write_json(
        manifest,
        {
            "variant_name": "paction_lattice_replace_score_only_move50",
            "target_len": 384,
            "max_selected_count": 384,
            "selected_count_histogram": {"384": 1},
        },
    )

    with pytest.raises(AssertionError, match="diagnostic"):
        gate.build_claim_budget_report(claim_manifests=[manifest])


def test_claim_budget_gate_cli_failure_emits_json(tmp_path: Path, capsys) -> None:
    manifest = tmp_path / "bad.claim_manifest.json"
    _write_json(
        manifest,
        {
            "variant_name": "detector_aware_fixed_768",
            "target_len": 768,
            "max_selected_count": 768,
            "selected_count_histogram": {"768": 1},
        },
    )

    exit_code = gate.main(["--claim-manifest", str(manifest)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert payload["decision"] == gate.FAIL
    assert "exceeds claim budget" in payload["error"]


def test_claim_budget_gate_accepts_active_fixed384_config(monkeypatch) -> None:
    monkeypatch.setenv("C3_DETECTOR_AWARE_LEDGER_VARIANT", "detector_aware_fixed_384")
    monkeypatch.setenv("C3_DETECTOR_AWARE_TRAIN_LEDGER_PATH", "/tmp/train.jsonl")
    monkeypatch.setenv("C3_DETECTOR_AWARE_VAL_LEDGER_PATH", "/tmp/val.jsonl")
    monkeypatch.setenv("C3_DETECTOR_AWARE_TEST_LEDGER_PATH", "/tmp/test.jsonl")

    report = gate.build_claim_budget_report(
        configs=["configs/adatad/thumos/c3_detector_aware_ledger_adatad_full_train.py"]
    )

    checked = report["config_reports"][0]["checked_budget_fields"]
    assert checked["window_size"] == 384


def test_claim_budget_gate_rejects_active_fixed768_config(monkeypatch) -> None:
    monkeypatch.setenv("C3_DETECTOR_AWARE_LEDGER_VARIANT", "detector_aware_fixed_768")
    monkeypatch.setenv("C3_DETECTOR_AWARE_TRAIN_LEDGER_PATH", "/tmp/train.jsonl")
    monkeypatch.setenv("C3_DETECTOR_AWARE_VAL_LEDGER_PATH", "/tmp/val.jsonl")
    monkeypatch.setenv("C3_DETECTOR_AWARE_TEST_LEDGER_PATH", "/tmp/test.jsonl")

    with pytest.raises(AssertionError, match="config exceeds claim budget"):
        gate.build_claim_budget_report(
            configs=["configs/adatad/thumos/c3_detector_aware_ledger_adatad_full_train.py"]
        )
