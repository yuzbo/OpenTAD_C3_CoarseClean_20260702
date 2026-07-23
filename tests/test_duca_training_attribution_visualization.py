from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata import export_duca_training_attribution as exporter
from tools.bata import plot_duca_training_attribution as plotter


def _record(epoch: int) -> dict:
    return {
        "schema_version": exporter.RECORD_SCHEMA_VERSION,
        "sample_id": "video_a|0",
        "video_id": "video_a",
        "valid_len": 8,
        "selected_positions": [0, 2, 5, 7],
        "p_action": [0.1] * 8,
        "abs_delta_p_action": [0.2] * 8,
        "transition_policy_scores": [0.3] * 8,
        "sampling_rate_logits": [0.0] * 8,
        "sampling_rates": [0.5] * 8,
        "sampling_density": [0.125] * 8,
        "sampling_rate_logit_gradient_abs": [0.01] * 8,
        "detector_cls_selected_input_x_gradient": [0.1] * 4,
        "detector_reg_selected_input_x_gradient": [0.2] * 4,
        "detector_cls_input_x_gradient_dense_interpolated": [0.1] * 8,
        "detector_reg_input_x_gradient_dense_interpolated": [0.2] * 8,
        "detector_contribution_prediction_distribution": [[0.125, 0.125]] * 8,
        "gt_segments": [[2.0, 6.0]],
        "gt_boundary_validity": [[True, True]],
        "gt_role": "visualization_overlay_and_train_loss_only_not_inference_decision",
        "source": {"checkpoint_epoch": epoch, "train_only_attribution": True},
    }


def test_training_attribution_cli_exposes_fixed_batch_and_checkpoint_state() -> None:
    source = Path(exporter.__file__).read_text(encoding="utf-8")
    assert '"--batch-index"' in source
    assert '"--checkpoint-state"' in source
    assert '"optimizer_step_executed": False' in source
    assert '"gt_used_for_inference_decision": False' in source


def test_plotter_groups_one_video_window_across_epochs(tmp_path: Path) -> None:
    records = []
    for epoch in (0, 10):
        path = tmp_path / f"epoch_{epoch}.jsonl"
        path.write_text(json.dumps(_record(epoch)) + "\n", encoding="utf-8")
        records.append(path)
    grouped = {}
    for path in records:
        for record in plotter._read_jsonl(path):
            grouped.setdefault(record["sample_id"], []).append(record)
    assert sorted(plotter._epoch(record) for record in grouped["video_a|0"]) == [0, 10]
    assert plotter._normalize([1.0, 3.0]) == [0.25, 0.75]
    assert plotter._channel([[0.1, 0.2], [0.3, 0.4]], 1) == [0.2, 0.4]


def test_training_attribution_plot_contract_labels_gt_as_noninference() -> None:
    source = Path(plotter.__file__).read_text(encoding="utf-8")
    assert "GT is overlay only, not inference input" in source
    assert "sampling_rate_logit_gradient_abs" in source
    assert "detector_cls_input_x_gradient_dense_interpolated" in source


def test_plotter_renders_png_and_pdf_from_synthetic_train_only_records(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    records = []
    for epoch in (0, 10):
        path = tmp_path / f"epoch_{epoch}.jsonl"
        path.write_text(json.dumps(_record(epoch)) + "\n", encoding="utf-8")
        records.append(path)

    summary = plotter.plot_training_attribution(
        records_jsonl=records,
        output_prefix=tmp_path / "attribution",
        sample_id="video_a|0",
    )

    assert summary["epochs"] == [0, 10]
    assert summary["gt_overlay_not_inference_input"] is True
    assert len(summary["overlay_outputs"]) == 2
    for path in summary["outputs"]:
        assert Path(path).is_file()
