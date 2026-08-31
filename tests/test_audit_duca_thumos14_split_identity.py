from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path

import pytest

from tools.bata.audit_duca_thumos14_split_identity import (
    DEFAULT_STAGE1,
    DEFAULT_STAGE2,
    ROOT,
    config_inheritance_chain,
    extract_mapping_ids,
    load_resolved_config,
    parse_annotation_identity,
    run_audit,
)


def _video_record(subset: str, label: str) -> dict:
    return {
        "subset": subset,
        "duration": 10.0,
        "frame": 250,
        "annotations": [{"label": label, "segment": [1.25, 7.75]}],
    }


def _write_annotation(path: Path, training_ids: list[str], held_out_ids: list[str], held_label: str) -> None:
    database = {item: _video_record("training", "TrainAction") for item in training_ids}
    database.update({item: _video_record("validation", held_label) for item in held_out_ids})
    path.write_text(json.dumps({"version": "THUMOS14", "database": database}), encoding="utf-8")


def _write_actionformer_annotation(path: Path, ids: list[str]) -> None:
    database = {item: _video_record("test", "AF_SECRET") for item in ids}
    path.write_text(json.dumps({"database": database}), encoding="utf-8")


def test_held_out_labels_and_segments_are_not_decoded(tmp_path: Path) -> None:
    annotation = tmp_path / "annotation.json"
    _write_annotation(
        annotation,
        ["video_validation_0000001"],
        ["video_test_0000001"],
        "HELDOUT_SECRET_LABEL",
    )

    identity = parse_annotation_identity(annotation)

    assert identity.held_out_annotation_values_decoded is False
    assert "HELDOUT_SECRET_LABEL" not in identity.decoded_strings
    assert "segment" not in identity.decoded_strings
    assert identity.records[0].valid_training_annotation_count == 1
    assert identity.records[1].valid_training_annotation_count is None


def test_prediction_id_extraction_skips_prediction_payload(tmp_path: Path) -> None:
    prediction = tmp_path / "predictions.json"
    prediction.write_text(
        json.dumps(
            {
                "results": {
                    "video_test_0000001": [
                        {"label": "PREDICTION_SECRET", "segment": [2.0, 3.0], "score": 0.9}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    assert extract_mapping_ids(prediction) == {"video_test_0000001"}


def test_config_chain_resolves_without_loading_a_checkpoint() -> None:
    stage1_chain = config_inheritance_chain(DEFAULT_STAGE1)
    stage2_chain = config_inheritance_chain(DEFAULT_STAGE2)
    before = {name: os.environ.get(name) for name in (
        "DUCA_STAGE1_CHECKPOINT",
        "DUCA_STAGE1_CHECKPOINT_SHA256",
        "DUCA_STAGE1_CHECKPOINT_EPOCH",
    )}

    stage1 = load_resolved_config(DEFAULT_STAGE1)
    stage2 = load_resolved_config(DEFAULT_STAGE2)

    assert stage1_chain[0] == DEFAULT_STAGE1.resolve()
    assert stage2_chain[0] == DEFAULT_STAGE2.resolve()
    assert len(stage1_chain) >= 6
    assert len(stage2_chain) >= 6
    assert stage1["dataset"]["train"]["subset_name"] == "training"
    assert stage2["dataset"]["test"]["subset_name"] == "validation"
    assert stage2["evaluation"]["subset"] == "validation"
    assert stage1["workflow"]["primary_checkpoint_epoch"] == 29
    assert stage2["duca_stage1_checkpoint"] == "IDENTITY_AUDIT_PLACEHOLDER_DO_NOT_LOAD"
    assert stage2["workflow"]["model_initialization"]["expected_checkpoint_epoch"] == 29
    assert before == {name: os.environ.get(name) for name in before}


def _audit_fixture(tmp_path: Path, with_explanation: bool) -> Namespace:
    training_ids = [f"video_validation_{idx:07d}" for idx in range(1, 201)]
    held_out_ids = [f"video_test_{idx:07d}" for idx in range(1, 212)]
    extra_actionformer_id = "video_test_0000270"
    actionformer_ids = held_out_ids + [extra_actionformer_id]

    annotation = tmp_path / "thumos_14_anno.json"
    _write_annotation(annotation, training_ids, held_out_ids, "HELDOUT_DO_NOT_DECODE")
    class_map = tmp_path / "category_idx.txt"
    class_map.write_text("".join(f"class_{idx}\n" for idx in range(20)), encoding="utf-8")
    media_root = tmp_path / "video"
    media_root.mkdir()
    for video_id in training_ids + held_out_ids:
        (media_root / f"{video_id}.mp4").write_bytes(b"fixture")

    historical = tmp_path / "historical.json"
    historical.write_text(json.dumps({"results": {item: [] for item in held_out_ids}}), encoding="utf-8")
    actionformer = tmp_path / "actionformer.json"
    _write_actionformer_annotation(actionformer, actionformer_ids)
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    for video_id in actionformer_ids:
        (feature_root / f"{video_id}.npy").write_bytes(b"fixture")
    exclusion = tmp_path / "README.txt"
    exclusion.write_text(
        f"OpenTAD canonical media excludes {extra_actionformer_id}\n" if with_explanation else "no exclusions\n",
        encoding="utf-8",
    )
    return Namespace(
        repo_root=ROOT,
        stage1_config=DEFAULT_STAGE1,
        stage2_config=DEFAULT_STAGE2,
        annotation=annotation,
        class_map=class_map,
        media_root=media_root,
        media_suffix=".mp4",
        historical_211=historical,
        historical_mapping_key="results",
        actionformer_annotation=actionformer,
        actionformer_subset="test",
        actionformer_feature_root=feature_root,
        actionformer_feature_suffix=[".npy"],
        exclusion_source=[exclusion],
        output_dir=tmp_path / "output",
        ffprobe="unused",
        ffprobe_timeout=1,
        skip_media_decode=False,
    )


def _successful_probe(path: Path, executable: str, timeout: int) -> dict:
    assert path.is_file()
    return {"width": 320, "height": 240, "codec_name": "fixture"}


def test_full_identity_audit_passes_only_with_source_backed_211_212_difference(tmp_path: Path) -> None:
    report = run_audit(_audit_fixture(tmp_path, with_explanation=True), probe=_successful_probe)

    assert report["verdict"] == "PASS"
    assert report["counts"]["T_annotation"] == 200
    assert report["counts"]["H_annotation"] == 211
    assert report["counts"]["AF_annotation"] == 212
    assert report["identity"]["T_annotation"] == report["identity"]["T_loader"]
    assert report["identity"]["H_annotation"] == report["identity"]["H_evaluator"]
    assert report["isolation"]["held_out_label_or_segment_access"] is False
    assert (tmp_path / "output" / "split_identity_report.json").is_file()


def test_unexplained_211_212_difference_blocks(tmp_path: Path) -> None:
    report = run_audit(_audit_fixture(tmp_path, with_explanation=False), probe=_successful_probe)

    assert report["verdict"] == "BLOCK"
    assert "ActionFormer/OpenTAD ID difference lacks source-backed explanation" in report["blockers"]


def test_duplicate_database_ids_are_rejected(tmp_path: Path) -> None:
    annotation = tmp_path / "duplicate.json"
    annotation.write_text(
        '{"database":{"video_test_0000001":{"subset":"validation","annotations":[]},'
        '"video_test_0000001":{"subset":"validation","annotations":[]}}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate video ID"):
        parse_annotation_identity(annotation)
