from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.bata import export_frozen_kinetics_actionness as exporter


def test_lightweight_provider_set_excludes_heavy_prebackbone_models() -> None:
    assert "efficient_x3d_xs" in exporter.LIGHTWEIGHT_PROVIDERS
    assert "x3d_xs" in exporter.LIGHTWEIGHT_PROVIDERS
    assert "videomae_s" not in exporter.LIGHTWEIGHT_PROVIDERS
    assert "slowfast_r50" not in exporter.LIGHTWEIGHT_PROVIDERS
    assert "videomae_s" in exporter.HEAVY_OR_UPPER_BOUND_PROVIDERS
    with pytest.raises(ValueError, match="too heavy|not a lightweight"):
        exporter.export_actionness(
            annotation_json=Path("unused.json"),
            video_roots=[],
            output_jsonl=Path("unused.jsonl"),
            provider="videomae_s",
        )


def test_frozen_kinetics_provenance_is_no_thumos_and_cost_visible() -> None:
    provenance = exporter._source_provenance(
        provider="efficient_x3d_xs",
        pretrained=True,
        score_mode="entropy_mix",
        clip_frames=4,
        frame_interval=12,
        crop_size=160,
    )

    assert provenance["source_name"] == "frozen_kinetics_efficient_x3d_xs_actionness"
    assert provenance["training_dataset"] == "Kinetics"
    assert provenance["thumos_trained"] is False
    assert provenance["uses_labels"] is False
    assert provenance["uses_teacher"] is False
    assert provenance["uses_gt"] is False
    assert provenance["calibration_split"] == "none"
    assert provenance["efficiency_claim_role"] == "lightweight_train_free_prebackbone_candidate"


def test_sample_times_use_original_time_bin_centers() -> None:
    times = exporter._sample_times(8.0, 4)

    assert times == pytest.approx([1.0, 3.0, 5.0, 7.0])


def test_video_index_maps_stems(tmp_path: Path) -> None:
    video = tmp_path / "video_test_0000004.mp4"
    video.write_bytes(b"not-a-real-video")

    index = exporter._build_video_index([tmp_path])

    assert index["video_test_0000004"] == video


@pytest.mark.skipif(os.name == "nt", reason="local Windows torch DLL initialization is not reliable")
def test_export_actionness_streams_rows_and_progress(tmp_path: Path, monkeypatch, capsys) -> None:
    torch = pytest.importorskip("torch")
    video_path = tmp_path / "video_test_0000004.mp4"
    video_path.write_bytes(b"fake-video")

    class FakeModel:
        def __call__(self, inputs):
            return torch.ones(inputs.shape[0], 4)

    monkeypatch.setattr(exporter, "_read_json", lambda _path: {"database": {}})
    monkeypatch.setattr(
        exporter,
        "_video_database",
        lambda _annotation, subset=None: [("video_test_0000004", {"duration": 2.0})],
    )
    monkeypatch.setattr(exporter, "_build_video_index", lambda _roots: {"video_test_0000004": video_path})
    monkeypatch.setattr(exporter, "_load_model", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(
        exporter,
        "_decode_clip_decord",
        lambda *args, **kwargs: torch.zeros(3, 1, 2, 2),
    )

    output_jsonl = tmp_path / "actionness.jsonl"
    summary_json = tmp_path / "summary.json"
    summary = exporter.export_actionness(
        annotation_json=tmp_path / "ann.json",
        video_roots=[tmp_path],
        output_jsonl=output_jsonl,
        summary_json=summary_json,
        provider="x3d_xs",
        dense_window_size=2,
        batch_size=1,
        device="cpu",
    )

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    captured = capsys.readouterr()

    assert summary["row_count"] == 2
    assert len(rows) == 2
    assert rows[0]["video_id"] == "video_test_0000004"
    assert "[FROZEN_KINETICS_ACTIONNESS] video=1/1" in captured.err
    assert summary_json.exists()
