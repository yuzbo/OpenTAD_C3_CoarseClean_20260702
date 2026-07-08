from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from tools.bata import apply_paction_acquisition_policy as apply_policy
from tools.bata import apply_paction_lattice_replacement_policy as apply_lattice
from tools.bata import paction_lattice_replacement_policy as lattice


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _patch_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(apply_policy, "_sha256_file", _sha256_file)
    monkeypatch.setattr(
        apply_policy,
        "load_policy_checkpoint",
        lambda checkpoint_path, *, device="cuda": ("dummy-model", {"dynamic_budget_buckets": [128, 256, 384]}),
    )

    def fake_scores(model, p_action, *, valid=None, device="cuda"):
        del model, valid, device
        frame_values = [float(value) for value in p_action]
        return frame_values, [0.0, 0.5, 1.0]

    monkeypatch.setattr(apply_policy, "checkpoint_policy_scores", fake_scores)


def test_lattice_application_requires_checkpoint(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "samples.lattice.jsonl"
    input_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "video_test_0001|0",
                "dense_len": 8,
                "valid_len": 8,
                "frame_signals": {"p_action": [0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TypeError):
        apply_lattice.run_lattice_replacement_application(input_jsonl, output_jsonl)  # type: ignore[call-arg]


def test_lattice_application_emits_score_only_metadata_and_strips_deploy_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_checkpoint(monkeypatch)
    checkpoint = tmp_path / "policy.pth"
    checkpoint.write_text("dummy checkpoint", encoding="utf-8")
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "samples.lattice.jsonl"
    summary_json = tmp_path / "summary.json"
    input_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "video_test_0001|0",
                "dense_len": 512,
                "valid_len": 512,
                "frame_signals": {"p_action": [float(idx % 17) / 16.0 for idx in range(512)]},
                "paction_positive_provenance": _provenance(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = apply_lattice.run_lattice_replacement_application(
        input_jsonl,
        output_jsonl,
        checkpoint_path=checkpoint,
        summary_json=summary_json,
        variants=[lattice.MOVE50_STRATEGY],
        fixed_budget=384,
        device="cpu",
        local_radius=2,
        strip_deploy_invisible_payload=True,
        strict_deploy_source=True,
    )
    rows = _read_jsonl(output_jsonl)

    assert summary["decision"] == apply_lattice.READY
    assert summary["score_only"] is True
    assert summary["uses_manual_slots"] is False
    assert summary["uses_uniform_fill"] is False
    assert summary_json.is_file()
    row = rows[0]
    assert "frame_signals" not in row
    assert "p_action" not in row
    assert row["deploy_invisible_payload_stripped"] is True
    selected = row["strategy_selected_positions"][lattice.MOVE50_STRATEGY]
    assert selected == sorted(selected)
    assert len(selected) == 384
    metadata = row["paction_policy"]
    assert metadata["source"] == "learned_paction_gap_loss_policy_checkpoint"
    assert metadata["selection_decoder"] == "score_only_lattice_replacement_v1"
    assert metadata["score_only"] is True
    assert metadata["diagnostic_only"] is True
    assert metadata["paper_main_claim_allowed"] is False
    assert metadata["diagnostic_scope"] == "paction_lattice_replacement_policy_diagnostic_not_main_method"
    assert metadata["uses_manual_boundary_slots"] is False
    assert metadata["uses_manual_transition_slots"] is False
    assert metadata["uses_manual_uncertainty_slots"] is False
    assert metadata["uses_uniform_scaffold"] is True
    assert metadata["scaffold_type"] == "uniform_lattice_local_replacement"
    assert metadata["uses_uniform_fill"] is False
    assert metadata["p_action_provenance"]["probe_model"] == "mobilenetv3_64px"
    diagnostics = metadata["lattice_replacement_diagnostics_by_strategy"][lattice.MOVE50_STRATEGY]
    assert diagnostics["selected_count"] == 384
    assert diagnostics["protected_uniform_count"] == 192
    for forbidden in ("gt", "teacher", "oracle", "boundary", "transition", "uncertainty", "context", "role"):
        assert not any(forbidden in key.lower() for key in diagnostics)


def test_lattice_application_uses_short_valid_ratio_budget_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_checkpoint(monkeypatch)
    checkpoint = tmp_path / "policy.pth"
    checkpoint.write_text("dummy checkpoint", encoding="utf-8")
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "samples.lattice.jsonl"
    input_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "video_test_0001|0",
                "dense_len": 768,
                "valid_len": 251,
                "frame_signals": {"p_action": [float(idx % 17) / 16.0 for idx in range(768)]},
                "paction_positive_provenance": _provenance(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    apply_lattice.run_lattice_replacement_application(
        input_jsonl,
        output_jsonl,
        checkpoint_path=checkpoint,
        variants=[lattice.MOVE25_STRATEGY, lattice.MOVE50_STRATEGY, lattice.MOVE75_STRATEGY],
        fixed_budget=384,
        device="cpu",
        local_radius=2,
    )
    row = _read_jsonl(output_jsonl)[0]

    assert len(row["strategy_selected_positions"][lattice.MOVE50_STRATEGY]) == 126
    assert len(row["strategy_selected_positions"][lattice.MOVE75_STRATEGY]) == 126
    move25 = row["paction_policy"]["lattice_replacement_diagnostics_by_strategy"][lattice.MOVE25_STRATEGY]
    move50 = row["paction_policy"]["lattice_replacement_diagnostics_by_strategy"][lattice.MOVE50_STRATEGY]
    move75 = row["paction_policy"]["lattice_replacement_diagnostics_by_strategy"][lattice.MOVE75_STRATEGY]
    assert row["paction_policy"]["effective_lattice_budget"] == 126
    assert move25["selected_count"] == 126
    assert move25["protected_uniform_count"] == 94
    assert move25["replaceable_uniform_count"] == 32
    assert move50["selected_count"] == 126
    assert move50["protected_uniform_count"] == 63
    assert move50["replaceable_uniform_count"] == 63
    assert move75["selected_count"] == 126
    assert move75["protected_uniform_count"] == 32
    assert move75["replaceable_uniform_count"] == 94


def test_lattice_radius_variant_records_adaptive_context_radius_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_checkpoint(monkeypatch)
    checkpoint = tmp_path / "policy.pth"
    checkpoint.write_text("dummy checkpoint", encoding="utf-8")
    input_jsonl = tmp_path / "samples.jsonl"
    output_jsonl = tmp_path / "samples.lattice.jsonl"
    input_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "video_test_0001|0",
                "dense_len": 12,
                "valid_len": 12,
                "frame_signals": {"p_action": [0.02, 0.05, 0.10, 0.90, 0.88, 0.55, 0.48, 0.80, 0.20, 0.10, 0.05, 0.01]},
                "paction_positive_provenance": _provenance(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    apply_lattice.run_lattice_replacement_application(
        input_jsonl,
        output_jsonl,
        checkpoint_path=checkpoint,
        variants=[lattice.RADIUS_MOVE25_STRATEGY],
        fixed_budget=6,
        device="cpu",
        local_radius=2,
    )
    row = _read_jsonl(output_jsonl)[0]
    selected = row["strategy_selected_positions"][lattice.RADIUS_MOVE25_STRATEGY]
    metadata = row["paction_policy"]
    radii = metadata["context_radius_by_strategy"][lattice.RADIUS_MOVE25_STRATEGY]

    assert metadata["selection_decoder"] == "score_only_lattice_replacement_with_adaptive_radius_v1"
    assert metadata["learned_context_radius_used"] is True
    assert metadata["context_radius_range"] == [0.0, 16.0]
    assert metadata["context_radius_unit"] == "local_dense_snippet_index"
    assert len(radii) == 12
    assert all(0.0 <= float(item) <= 16.0 for item in radii)
    assert any(float(radii[position]) > 0.0 for position in selected)
    assert metadata["lattice_radius_diagnostics_by_strategy"][lattice.RADIUS_MOVE25_STRATEGY]["selected_radius"]["max"] <= 16.0
