from __future__ import annotations

import json

import pytest

from opentad.evaluations.mAP import load_blocked_videos


def test_blocked_videos_accepts_json_array_and_line_list(tmp_path):
    json_path = tmp_path / "blocked.json"
    json_path.write_text(
        json.dumps(["video_validation_0001", "video_validation_0002"]),
        encoding="utf-8",
    )
    text_path = tmp_path / "blocked.txt"
    text_path.write_text(
        "video_validation_0001\nvideo_validation_0002\n",
        encoding="utf-8",
    )

    expected = ["video_validation_0001", "video_validation_0002"]
    assert load_blocked_videos(json_path) == expected
    assert load_blocked_videos(text_path) == expected


@pytest.mark.parametrize(
    "content, message",
    [
        ("", "empty"),
        ("{}", "JSON must be an array"),
        ('["video_1", "video_1"]', "duplicate"),
        ('["video_1", 2]', "nonempty strings"),
        ("[\n", "Expecting"),
    ],
)
def test_blocked_videos_fails_closed_on_invalid_artifacts(
    tmp_path,
    content,
    message,
):
    path = tmp_path / "blocked.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises((ValueError, json.JSONDecodeError), match=message):
        load_blocked_videos(path)
