import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "tools" / "bata" / "evaluate_actionformer_raw_predictions.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_actionformer_raw_predictions",
    MODULE_PATH,
)
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def test_prediction_video_coverage_requires_exact_official_test_set(monkeypatch):
    monkeypatch.setattr(evaluator.protocol, "OFFICIAL_EVALUATED_VIDEO_COUNT", 2)
    videos = [("train", "validation"), ("test-a", "test"), ("test-b", "test")]
    raw = {"video-id": ["test-a", "test-b", "test-b"]}
    evaluated, predicted = evaluator._validate_prediction_video_coverage(raw, videos)
    assert evaluated == ["test-a", "test-b"]
    assert predicted == ["test-a", "test-b"]

    with pytest.raises(evaluator.protocol.ProtocolError, match="omit"):
        evaluator._validate_prediction_video_coverage(
            {"video-id": ["test-a"]},
            videos,
        )
    with pytest.raises(evaluator.protocol.ProtocolError, match="non-test"):
        evaluator._validate_prediction_video_coverage(
            {"video-id": ["test-a", "test-b", "train"]},
            videos,
        )


def test_evaluator_is_pinned_and_cannot_issue_a_main_table_verdict():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "verify_official_source" in source
    assert "recompute_official_metrics" in source
    assert "complete_official_test_coverage" in source
    assert '"paper_main_table_eligible": False' in source
    assert '"matched_record_suite_required": True' in source
