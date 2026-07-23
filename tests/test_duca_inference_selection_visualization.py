from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata import plot_duca_inference_selection as plotter


def _record(epoch: int, sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "valid_len": 8,
        "selected_positions": [0, 2, 5, 7],
        "transition_policy_scores": [0.1, 0.2, 0.8, 0.3, 0.2, 0.7, 0.3, 0.1],
        "sampling_rates": [0.4, 0.5, 0.9, 0.4, 0.2, 0.8, 0.5, 0.3],
        "gt_segments": [[2.0, 6.0]],
        "gt_boundary_validity": [[True, True]],
        "source": {"checkpoint_epoch": epoch},
    }


def test_inference_selection_plot_renders_all_fixed_validation_samples(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    records = []
    for epoch in (9, 19, 29):
        path = tmp_path / f"epoch_{epoch}.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(_record(epoch, sample))
                for sample in ("validation_a|0", "validation_b|16")
            )
            + "\n",
            encoding="utf-8",
        )
        records.append(path)

    summary = plotter.plot_inference_selection(
        records_jsonl=records,
        output_prefix=tmp_path / "validation_selection",
        all_fixed_samples=True,
    )

    assert summary["sample_ids"] == ["validation_a|0", "validation_b|16"]
    assert summary["epochs"]["validation_a|0"] == [10, 20, 30]
    assert summary["selector_only_inference"] is True
    assert summary["gt_overlay_not_selector_input"] is True
    for output in summary["outputs"]:
        assert Path(output).is_file()
