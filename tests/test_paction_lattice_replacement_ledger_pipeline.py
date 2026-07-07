from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from tools.bata import apply_paction_acquisition_policy as apply_policy
from tools.bata import paction_lattice_replacement_policy as lattice
from tools.bata import run_paction_lattice_replacement_ledger_pipeline as pipeline
from tools.bata import validate_paction_lattice_replacement_ledger as lattice_validator


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _patch_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(apply_policy, "_sha256_file", _sha256_file)
    monkeypatch.setattr(
        apply_policy,
        "load_policy_checkpoint",
        lambda checkpoint_path, *, device="cuda": ("dummy-model", {"dynamic_budget_buckets": [2, 4]}),
    )

    def fake_scores(model, p_action, *, valid=None, device="cuda"):
        del model, valid, device
        return [float(value) for value in p_action], [0.0, 1.0]

    monkeypatch.setattr(apply_policy, "checkpoint_policy_scores", fake_scores)


def _source_row(sample_id: str, p_action: list[float]) -> dict:
    dense_len = len(p_action)
    return {
        "sample_id": sample_id,
        "video_name": sample_id.split("|", 1)[0],
        "window_start_frame": int(sample_id.split("|", 1)[1]),
        "dense_len": dense_len,
        "valid_len": dense_len,
        "frame_signals": {"p_action": p_action},
        "action_target": [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0][:dense_len],
        "gt_boundaries": [1.0, 3.0, 6.0],
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
    }


def _source_row_without_explicit_provenance(sample_id: str, p_action: list[float]) -> dict:
    row = _source_row(sample_id, p_action)
    row.pop("paction_positive_provenance")
    row["matrix_model_id"] = "official_asformer_lowres"
    row["official_action_seg_backend"] = "official_asformer"
    row["tcn_variant"] = "official_asformer"
    row["uses_gt_for_diagnostics"] = True
    return row


def test_lattice_replacement_pipeline_generates_deployable_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_checkpoint(monkeypatch)
    input_jsonl = tmp_path / "source.jsonl"
    checkpoint = tmp_path / "policy.pth"
    checkpoint.write_text("dummy checkpoint", encoding="utf-8")
    input_jsonl.write_text(
        "\n".join(
            [
                json.dumps(_source_row("video_test_0001|0", [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])),
                json.dumps(_source_row("video_test_0002|0", [0.8, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5, 0.4])),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    variant = lattice.NO_PROTECT_STRATEGY
    summary = pipeline.run_pipeline(
        input_jsonl=input_jsonl,
        checkpoint_path=checkpoint,
        out_dir=tmp_path / "out",
        variants=[variant],
        fixed_budget=4,
        device="cpu",
        deploy_selection_ledger=True,
        local_radius=2,
        summary_json=tmp_path / "out" / "pipeline.summary.json",
    )

    assert summary["decision"] == pipeline.READY
    assert summary["score_only"] is True
    assert summary["uses_manual_slots"] is False
    assert summary["uses_uniform_scaffold"] is True
    assert summary["scaffold_type"] == "uniform_lattice_local_replacement"
    assert summary["uses_uniform_fill"] is False
    scaffold_summary = summary["lattice_scaffold_summary_by_variant"][variant]
    assert scaffold_summary["uses_uniform_scaffold"] is True
    assert scaffold_summary["scaffold_type"] == "uniform_lattice_local_replacement"
    assert scaffold_summary["protected_uniform_count"]["min"] == 0
    assert scaffold_summary["replaceable_uniform_count"]["min"] == 4
    assert scaffold_summary["base_uniform_jaccard"]["min"] >= 0.0
    assert scaffold_summary["base_uniform_jaccard"]["max"] <= 1.0
    written_apply_summary = json.loads(Path(summary["apply_summary_json"]).read_text(encoding="utf-8"))
    assert written_apply_summary["uses_uniform_scaffold"] is True
    assert written_apply_summary["scaffold_type"] == "uniform_lattice_local_replacement"
    sample_rows = _read_jsonl(Path(summary["sample_jsonl"]))
    ledger_rows = _read_jsonl(Path(summary["ledgers"][variant]["ledger_jsonl"]))
    assert len(sample_rows) == 2
    assert len(ledger_rows) == 2
    for sample_row, ledger_row in zip(sample_rows, ledger_rows):
        assert "frame_signals" not in sample_row
        assert "p_action" not in sample_row
        assert "gt_boundaries" not in sample_row
        assert "action_target" not in sample_row
        assert ledger_row["deploy_selection_ledger"] is True
        assert ledger_row["uses_gt"] is False
        assert ledger_row["uses_teacher"] is False
        assert ledger_row["uses_oracle"] is False
        assert ledger_row["uses_prediction_cache"] is False
        assert ledger_row["policy_source"] == "learned_paction_gap_loss_policy_checkpoint"
        assert ledger_row["selected_positions"] == sample_row["strategy_selected_positions"][variant]
        assert ledger_row["selected_count"] == 4
    validation = summary["ledgers"][variant]["validation_summary"]
    assert validation["decision"] == "C3_PACTION_LATTICE_REPLACEMENT_LEDGER_VALIDATION_PASS"
    assert validation["lattice_metadata"]["row_count"] == 2


def test_formal_lattice_pipeline_requires_explicit_deploy_provenance_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_checkpoint(monkeypatch)
    input_jsonl = tmp_path / "source.jsonl"
    checkpoint = tmp_path / "policy.pth"
    checkpoint.write_text("dummy checkpoint", encoding="utf-8")
    input_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    _source_row_without_explicit_provenance(
                        "video_test_0001|0", [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]
                    )
                ),
                json.dumps(
                    _source_row_without_explicit_provenance(
                        "video_test_0002|0", [0.8, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5, 0.4]
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="p_action positive provenance is required"):
        pipeline.run_pipeline(
            input_jsonl=input_jsonl,
            checkpoint_path=checkpoint,
            out_dir=tmp_path / "out",
            variants=[lattice.MOVE50_STRATEGY],
            fixed_budget=4,
            device="cpu",
            deploy_selection_ledger=True,
            local_radius=2,
        )


def test_lattice_pipeline_infers_deploy_provenance_only_with_explicit_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_checkpoint(monkeypatch)
    input_jsonl = tmp_path / "source.jsonl"
    checkpoint = tmp_path / "policy.pth"
    checkpoint.write_text("dummy checkpoint", encoding="utf-8")
    input_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    _source_row_without_explicit_provenance(
                        "video_test_0001|0", [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]
                    )
                ),
                json.dumps(
                    _source_row_without_explicit_provenance(
                        "video_test_0002|0", [0.8, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5, 0.4]
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = pipeline.run_pipeline(
        input_jsonl=input_jsonl,
        checkpoint_path=checkpoint,
        out_dir=tmp_path / "out",
        variants=[lattice.MOVE50_STRATEGY],
        fixed_budget=4,
        device="cpu",
        deploy_selection_ledger=True,
        allow_inferred_paction_positive_provenance=True,
        local_radius=2,
    )

    report = summary["selection_source_report"]
    assert summary["formal_provenance_mode"] == "inferred_paction_positive_provenance_explicit_opt_in"
    assert report["allow_inferred_paction_positive_provenance"] is True
    assert report["inferred_paction_positive_provenance_count"] == 2
    rows = _read_jsonl(Path(summary["selection_sample_jsonl"]))
    assert rows[0]["paction_positive_provenance"]["inferred_from_source_row"] is True
    assert rows[0]["paction_positive_provenance"]["no_gt_generation"] is True
    assert rows[0]["paction_positive_provenance"]["uses_gt_for_diagnostics"] is False
    assert "action_target" not in rows[0]
    assert "gt_boundaries" not in rows[0]


def test_lattice_validator_cli_accepts_positive_deploy_compatibility_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict = {}

    def fake_validate_lattice_ledger(**kwargs):
        called.update(kwargs)
        return {"decision": lattice_validator.READY}

    monkeypatch.setattr(lattice_validator, "validate_lattice_ledger", fake_validate_lattice_ledger)

    rc = lattice_validator.main(
        [
            "--sample-jsonl",
            "samples.jsonl",
            "--metric-sample-jsonl",
            "metrics.jsonl",
            "--ledger-jsonl",
            "ledger.jsonl",
            "--strategy",
            lattice.MOVE50_STRATEGY,
            "--allow-short-valid-ratio-count",
            "--require-deployable",
        ]
    )

    assert rc == 0
    assert called["allow_short_valid_ratio_count"] is True
    assert called["require_deployable"] is True


def test_lattice_replacement_pipeline_defaults_to_diagnostic_uniform_scaffold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_checkpoint(monkeypatch)
    input_jsonl = tmp_path / "source.jsonl"
    checkpoint = tmp_path / "policy.pth"
    checkpoint.write_text("dummy checkpoint", encoding="utf-8")
    input_jsonl.write_text(
        json.dumps(_source_row("video_test_0001|0", [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])) + "\n",
        encoding="utf-8",
    )

    summary = pipeline.run_pipeline(
        input_jsonl=input_jsonl,
        checkpoint_path=checkpoint,
        out_dir=tmp_path / "out",
        variants=[lattice.NO_PROTECT_STRATEGY],
        fixed_budget=4,
        device="cpu",
        local_radius=2,
    )

    assert summary["diagnostic_only"] is True
    assert summary["deploy_selection_ledger"] is False
    assert summary["paper_main_claim_allowed"] is False
    assert summary["formal_provenance_mode"] == "diagnostic_ledger"
    assert summary["uses_uniform_scaffold"] is True
    assert summary["diagnostic_scope"] == "paction_lattice_replacement_policy_diagnostic_not_main_method"

    ledger_path = Path(summary["ledgers"][lattice.NO_PROTECT_STRATEGY]["ledger_jsonl"])
    ledger_rows = _read_jsonl(ledger_path)
    assert ledger_rows[0]["diagnostic_only"] is True
    assert ledger_rows[0]["deploy_selection_ledger"] is False
    assert ledger_rows[0]["uses_uniform_scaffold"] is True
    assert ledger_rows[0]["paper_main_claim_allowed"] is False


def test_lattice_full_train_launcher_requires_explicit_inferred_provenance_opt_in() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts" / "run_c3_paction_lattice_replacement_adatad_full_train_gpu1.sh").read_text(
        encoding="utf-8"
    )

    assert 'PACTION_LATTICE_ALLOW_INFERRED_PROVENANCE="${PACTION_LATTICE_ALLOW_INFERRED_PROVENANCE:-0}"' in text
    assert 'PACTION_LATTICE_ALLOW_INFERRED_PROVENANCE}" == "1"' in text
    assert "args+=(--allow-inferred-paction-positive-provenance)" in text
