from __future__ import annotations

import json
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
