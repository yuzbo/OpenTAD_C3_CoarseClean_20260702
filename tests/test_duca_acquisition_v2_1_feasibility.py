from __future__ import annotations

import hashlib
import json

import pytest

from tools.bata.audit_duca_acquisition_v2_1_feasibility import (
    audit_feasibility,
    audit_legacy_feasibility,
    enumerate_natural_window_valid_lengths,
)
from tools.bata.duca_evidence_io import verify_content_sha256
from tools.bata.duca_admission_v2_1_roles import validate_role_manifest


def test_current_sliding_enumerator_separates_full_and_short_videos():
    full_lengths = enumerate_natural_window_valid_lengths(frame_count=4 * 1000)
    short_lengths = enumerate_natural_window_valid_lengths(frame_count=4 * 250)

    assert full_lengths
    assert set(full_lengths) == {768}
    assert short_lengths == (250,)
    assert not any(value < 768 for value in full_lengths)
    assert 768 not in short_lengths


def test_terminal_window_is_back_shifted_to_full_at_769_snippets():
    assert enumerate_natural_window_valid_lengths(frame_count=4 * 769) == (768, 768)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"frame_count": 0},
        {"frame_count": 4, "snippet_stride": 0},
        {"frame_count": 4, "window_overlap_ratio": 1.0},
        {"frame_count": 4, "fps": 1.0, "duration": None},
    ),
)
def test_enumerator_rejects_invalid_runtime_geometry(kwargs):
    with pytest.raises(ValueError):
        enumerate_natural_window_valid_lengths(**kwargs)


def _write_feasibility_fixture(
    tmp_path, *, relative_annotation_path=False, historical_70_26=False
):
    snippet_counts = (
        [900] * 70 + [200] * 7 + [400] * 10 + [600] * 9
        if historical_70_26
        else [900] * 70 + [200] * 7 + [400] * 13 + [600] * 10
    )
    video_ids = [f"video_{index:03d}" for index in range(len(snippet_counts))]
    annotation = {
        "database": {
            video_id: {
                "subset": "training",
                "frame": snippet_count * 4,
                "duration": float(snippet_count),
                "annotations": [],
            }
            for video_id, snippet_count in zip(video_ids, snippet_counts)
        }
    }
    annotation_path = tmp_path / "annotation.json"
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    annotation_sha = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    split = {
        "schema": "duca_rime_video_split_manifest_v1",
        "assignment_sha256": "b" * 64,
        "annotation_sha256": annotation_sha,
        "annotation_path": (
            annotation_path.name
            if relative_annotation_path
            else str(annotation_path)
        ),
        "train_source_subset": "training",
        "train_roles": {
            "detector_selector_train": {
                "video_count": len(video_ids),
                "videos": video_ids,
            }
        },
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    split_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
    return split_path, split_sha, annotation_path


def test_corrected_v2_1_inventory_builds_disjoint_32_video_roles(tmp_path):
    split_path, split_sha, _annotation_path = _write_feasibility_fixture(tmp_path)
    receipt = audit_feasibility(
        split_manifest_path=split_path,
        expected_split_manifest_sha256=split_sha,
    )

    verify_content_sha256(receipt)
    validate_role_manifest(receipt)
    assert receipt["status"] == "PASSED"
    assert len(receipt["reserves"]) == 4
    assert len(receipt["triplets"]) == 32
    assert all(len(receipt["roles"][role]) == 32 for role in receipt["roles"])
    assert receipt["authorization_scope"] == "NONE"
    assert receipt["phase1_v2_authorized"] is False
    assert receipt["holdout_open_authorized"] is False
    assert receipt["paper_claim_allowed"] is False
    assert receipt["official_final_sealed"] is True


def test_historical_70_26_per_video_full_short_proposal_remains_failed_closed(tmp_path):
    split_path, split_sha, _annotation_path = _write_feasibility_fixture(
        tmp_path, historical_70_26=True
    )
    receipt = audit_legacy_feasibility(
        split_manifest_path=split_path,
        expected_split_manifest_sha256=split_sha,
    )
    verify_content_sha256(receipt)
    assert receipt["status"] == "failed"
    assert (
        "natural_full_and_short_per_video_infeasible_under_current_sliding_enumerator"
        in receipt["reason_codes"]
    )
    assert receipt["phase1_v2_authorized"] is False
    assert receipt["admission_effect"] is False


def test_feasibility_audit_resolves_annotation_relative_to_manifest(tmp_path):
    split_path, split_sha, _annotation_path = _write_feasibility_fixture(
        tmp_path,
        relative_annotation_path=True,
    )
    receipt = audit_feasibility(
        split_manifest_path=split_path,
        expected_split_manifest_sha256=split_sha,
    )
    assert receipt["status"] == "PASSED"
    verify_content_sha256(receipt)


def test_feasibility_audit_rejects_annotation_drift(tmp_path):
    split_path, split_sha, annotation_path = _write_feasibility_fixture(tmp_path)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["database"]["video_000"]["frame"] += 4
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="annotation SHA-256 drift"):
        audit_feasibility(
            split_manifest_path=split_path,
            expected_split_manifest_sha256=split_sha,
        )


def test_feasibility_audit_rejects_duplicate_source_video_ids(tmp_path):
    split_path, _split_sha, _annotation_path = _write_feasibility_fixture(tmp_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    videos = split["train_roles"]["detector_selector_train"]["videos"]
    videos[-1] = videos[0]
    split_path.write_text(json.dumps(split), encoding="utf-8")
    split_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="unique"):
        audit_feasibility(
            split_manifest_path=split_path,
            expected_split_manifest_sha256=split_sha,
        )


def test_feasibility_audit_rejects_invalid_utf8_before_json_parsing(tmp_path):
    split_path = tmp_path / "split.json"
    split_path.write_bytes(b'{"schema":"broken","video":"\xff"}')
    split_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
    with pytest.raises(UnicodeDecodeError):
        audit_feasibility(
            split_manifest_path=split_path,
            expected_split_manifest_sha256=split_sha,
        )


@pytest.mark.parametrize(
    ("mutator", "match"),
    (
        (
            lambda split, annotation: split["train_roles"]["detector_selector_train"][
                "videos"
            ].__setitem__(0, 7),
            "video IDs must be strings",
        ),
        (
            lambda split, annotation: split.__setitem__("train_source_subset", "validation"),
            "exact training source subset",
        ),
        (
            lambda split, annotation: annotation["database"]["video_000"].__setitem__(
                "frame", 1.5
            ),
            "frame count is invalid",
        ),
    ),
)
def test_corrected_audit_rejects_silent_type_or_subset_coercion(tmp_path, mutator, match):
    split_path, _split_sha, annotation_path = _write_feasibility_fixture(tmp_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    mutator(split, annotation)
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    split["annotation_sha256"] = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    split_path.write_text(json.dumps(split), encoding="utf-8")
    split_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match=match):
        audit_feasibility(
            split_manifest_path=split_path,
            expected_split_manifest_sha256=split_sha,
        )
