from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from tools.bata import apply_paction_acquisition_policy as apply_policy
from tools.bata import paction_lattice_replacement_policy as lattice
from tools.bata import run_paction_lattice_replacement_ledger_pipeline as pipeline


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

    summary = pipeline.run_pipeline(
        input_jsonl=input_jsonl,
        checkpoint_path=checkpoint,
        out_dir=tmp_path / "out",
        variants=[lattice.MOVE50_STRATEGY],
        fixed_budget=4,
        device="cpu",
        deploy_selection_ledger=True,
        local_radius=2,
        summary_json=tmp_path / "out" / "pipeline.summary.json",
    )

    assert summary["decision"] == pipeline.READY
    assert summary["score_only"] is True
    assert summary["uses_manual_slots"] is False
    assert summary["uses_uniform_fill"] is False
    sample_rows = _read_jsonl(Path(summary["sample_jsonl"]))
    ledger_rows = _read_jsonl(Path(summary["ledgers"][lattice.MOVE50_STRATEGY]["ledger_jsonl"]))
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
        assert ledger_row["selected_positions"] == sample_row["strategy_selected_positions"][lattice.MOVE50_STRATEGY]
        assert ledger_row["selected_count"] == 4
    validation = summary["ledgers"][lattice.MOVE50_STRATEGY]["validation_summary"]
    assert validation["decision"] == "C3_PACTION_LATTICE_REPLACEMENT_LEDGER_VALIDATION_PASS"
    assert validation["lattice_metadata"]["row_count"] == 2


def test_lattice_pipeline_infers_deploy_provenance_from_canonical_source_export(
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
        local_radius=2,
    )

    report = summary["selection_source_report"]
    assert report["allow_inferred_paction_positive_provenance"] is True
    assert report["inferred_paction_positive_provenance_count"] == 2
    rows = _read_jsonl(Path(summary["selection_sample_jsonl"]))
    assert rows[0]["paction_positive_provenance"]["inferred_from_source_row"] is True
    assert rows[0]["paction_positive_provenance"]["no_gt_generation"] is True
    assert rows[0]["paction_positive_provenance"]["uses_gt_for_diagnostics"] is False
    assert "action_target" not in rows[0]
    assert "gt_boundaries" not in rows[0]
