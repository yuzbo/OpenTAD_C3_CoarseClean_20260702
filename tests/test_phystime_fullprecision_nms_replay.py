import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from mmengine.config import Config, ConfigDict

import opentad.cores.test_engine as test_engine
import opentad.models.utils.post_processing.nms.nms as nms_module
import tools.bata.replay_phystime_p0_fullprecision_nms as replay


def _post_cfg(**overrides):
    values = {
        "sliding_window": True,
        "nms": {
            "use_soft_nms": False,
            "iou_threshold": 0.5,
            "max_seg_num": 100,
            "multiclass": True,
        },
        "filter_invalid_proposals": True,
        "proposal_min_duration": 1.0e-6,
        "round_before_cross_window_nms": False,
        "round_after_cross_window_nms": False,
        "segment_round_digits": 2,
        "score_round_digits": 4,
    }
    values.update(overrides)
    return ConfigDict(values)


def _threshold_nms(segs, scores, labels, **kwargs):
    assert labels.dtype == torch.long
    order = torch.argsort(scores, descending=True)
    kept = []
    for index in order.tolist():
        candidate = segs[index]
        suppress = False
        for kept_index in kept:
            reference = segs[kept_index]
            intersection = (
                torch.minimum(candidate[1], reference[1])
                - torch.maximum(candidate[0], reference[0])
            ).clamp(min=0)
            union = (
                candidate[1]
                - candidate[0]
                + reference[1]
                - reference[0]
                - intersection
            )
            if float(intersection / union) > kwargs["iou_threshold"]:
                suppress = True
                break
        if not suppress:
            kept.append(index)
    kept = torch.tensor(kept, dtype=torch.long)
    return segs[kept], scores[kept], labels[kept]


def _stable_threshold_nms(segs, scores, labels, **kwargs):
    assert labels.dtype == torch.long
    order = sorted(
        range(len(scores)),
        key=lambda index: -float(scores[index]),
    )
    kept = []
    for index in order:
        candidate = segs[index]
        suppress = False
        for kept_index in kept:
            reference = segs[kept_index]
            intersection = (
                torch.minimum(candidate[1], reference[1])
                - torch.maximum(candidate[0], reference[0])
            ).clamp(min=0)
            union = (
                candidate[1]
                - candidate[0]
                + reference[1]
                - reference[0]
                - intersection
            )
            if float(intersection / union) > kwargs["iou_threshold"]:
                suppress = True
                break
        if not suppress:
            kept.append(index)
    kept = torch.tensor(kept, dtype=torch.long)
    return segs[kept], scores[kept], labels[kept]


def test_single_stage_keeps_legacy_default_but_allows_p0_full_precision():
    source = Path("opentad/models/detectors/single_stage.py").read_text(
        encoding="utf-8"
    )
    assert "segment_output = [float(seg.item()) for seg in segment]" in source
    assert "score_output = float(score.item())" in source
    assert '"round_before_cross_window_nms",' in source
    assert "if round_before_cross_window_nms:" in source
    assert "round(value, segment_round_digits)" in source
    assert "round(score_output, score_round_digits)" in source

    root = Path(__file__).resolve().parents[1]
    ordinary = Config.fromfile(
        root
        / "configs"
        / "adatad"
        / "thumos"
        / "phystime_g1a_physical_metric_native_j192.py",
        lazy_import=False,
    )
    p0 = Config.fromfile(
        root
        / "configs"
        / "adatad"
        / "thumos"
        / "phystime_g1a_physical_metric_native_j192_p0_replay.py",
        lazy_import=False,
    )
    assert ordinary.post_processing.get(
        "round_before_cross_window_nms",
        True,
    ) is True
    assert p0.post_processing.round_before_cross_window_nms is False


def test_full_precision_and_legacy_rounding_cross_different_nms_branch(
    monkeypatch,
):
    monkeypatch.setattr(test_engine, "batched_nms", _threshold_nms)
    detections = {
        "video": [
            {"segment": [0.0, 1.0], "label": "action", "score": 0.9},
            {"segment": [0.334, 1.334], "label": "action", "score": 0.8},
        ]
    }

    full, full_audit = test_engine.apply_sliding_window_nms(
        detections,
        _post_cfg(),
        return_audit=True,
    )
    legacy, legacy_audit = test_engine.apply_sliding_window_nms(
        detections,
        _post_cfg(
            round_before_cross_window_nms=True,
            round_after_cross_window_nms=True,
        ),
        return_audit=True,
    )

    assert len(full["video"]) == 2
    assert len(legacy["video"]) == 1
    assert full_audit["aggregate"]["pre_nms_rounding_changed_segment_values"] == 0
    assert (
        legacy_audit["aggregate"]["pre_nms_rounding_changed_segment_values"]
        == 2
    )


def test_invalid_proposals_are_filtered_and_fully_audited(monkeypatch):
    observed = {}

    def capture_nms(segs, scores, labels, **kwargs):
        observed["segments"] = segs
        observed["scores"] = scores
        observed["labels"] = labels
        return segs, scores, labels

    monkeypatch.setattr(test_engine, "batched_nms", capture_nms)
    detections = {
        "video": [
            {"segment": [0.0, 1.0], "label": "action", "score": 0.9},
            {"segment": [0.0, 1.0], "label": "action", "score": float("nan")},
            {"segment": [2.0, 1.0], "label": "action", "score": 0.7},
            {"segment": [1.0, 1.0], "label": "action", "score": 0.6},
            {
                "segment": [float("inf"), 2.0],
                "label": "action",
                "score": 0.5,
            },
            {"segment": [0.0], "label": "action", "score": 0.4},
            {"segment": [0.0, 1.0], "label": None, "score": 0.3},
        ]
    }
    merged, audit = test_engine.apply_sliding_window_nms(
        detections,
        _post_cfg(filter_invalid_proposals=True),
        return_audit=True,
    )

    aggregate = audit["aggregate"]
    assert len(merged["video"]) == 1
    assert observed["segments"].shape == (1, 2)
    assert observed["scores"].shape == (1,)
    assert observed["labels"].dtype == torch.long
    assert aggregate["input_detections"] == 7
    assert aggregate["valid_detections"] == 1
    assert aggregate["invalid_detections"] == 6
    assert aggregate["filtered_detections"] == 6
    assert aggregate["raw_invalid_detections"] == 6
    assert aggregate["effective_invalid_detections"] == 0
    assert aggregate["rounding_induced_invalid_detections"] == 0
    assert aggregate["invalid_reason_counts"] == {
        "malformed_detection": 0,
        "malformed_segment": 1,
        "non_finite_segment": 1,
        "non_finite_score": 1,
        "non_positive_duration": 2,
        "invalid_label": 1,
    }


def test_legacy_rounding_induced_zero_duration_is_filtered_before_nms(
    monkeypatch,
):
    observed = {}

    def capture_nms(segs, scores, labels, **kwargs):
        observed["count"] = len(segs)
        return segs, scores, labels

    monkeypatch.setattr(test_engine, "batched_nms", capture_nms)
    detections = {
        "video": [
            {
                "segment": [0.001, 0.004],
                "label": "action",
                "score": 0.9,
            }
        ]
    }
    merged, audit = test_engine.apply_sliding_window_nms(
        detections,
        _post_cfg(
            filter_invalid_proposals=True,
            round_before_cross_window_nms=True,
            round_after_cross_window_nms=True,
        ),
        return_audit=True,
    )
    assert merged["video"] == []
    assert "count" not in observed
    assert audit["aggregate"]["raw_invalid_detections"] == 0
    assert audit["aggregate"]["effective_invalid_detections"] == 1
    assert audit["aggregate"]["effective_filtered_detections"] == 1
    assert audit["aggregate"]["rounding_induced_invalid_detections"] == 1


def test_legacy_unfiltered_rounding_induced_invalid_fails_before_nms(
    monkeypatch,
):
    called = False

    def forbidden_nms(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("NMS must not receive a rounded zero-duration segment")

    monkeypatch.setattr(test_engine, "batched_nms", forbidden_nms)
    with pytest.raises(test_engine.InvalidProposalError) as caught:
        test_engine.apply_sliding_window_nms(
            {
                "video": [
                    {
                        "segment": [0.001, 0.004],
                        "label": "action",
                        "score": 0.9,
                    }
                ]
            },
            _post_cfg(
                filter_invalid_proposals=False,
                round_before_cross_window_nms=True,
                round_after_cross_window_nms=True,
            ),
            return_audit=True,
        )
    assert called is False
    assert (
        caught.value.audit["aggregate"][
            "rounding_induced_invalid_detections"
        ]
        == 1
    )


def test_four_digit_score_rounding_can_change_the_selected_proposal(
    monkeypatch,
):
    monkeypatch.setattr(test_engine, "batched_nms", _stable_threshold_nms)
    detections = {
        "video": [
            {
                "segment": [0.0, 1.0],
                "label": "action",
                "score": 0.50004,
            },
            {
                "segment": [0.01, 1.01],
                "label": "action",
                "score": 0.50005,
            },
        ]
    }
    full = test_engine.apply_sliding_window_nms(
        detections,
        _post_cfg(),
    )
    legacy = test_engine.apply_sliding_window_nms(
        detections,
        _post_cfg(
            round_before_cross_window_nms=True,
            round_after_cross_window_nms=True,
        ),
    )
    assert full["video"][0]["segment"] == pytest.approx([0.01, 1.01])
    assert legacy["video"][0]["segment"] == pytest.approx([0.0, 1.0])


def test_unfiltered_invalid_input_fails_closed_before_nms(monkeypatch):
    called = False

    def forbidden_nms(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("NMS must not receive invalid proposals")

    monkeypatch.setattr(test_engine, "batched_nms", forbidden_nms)
    detections = {
        "video": [
            {"segment": [1.0, 1.0], "label": "action", "score": 0.5},
        ]
    }
    with pytest.raises(test_engine.InvalidProposalError) as caught:
        test_engine.apply_sliding_window_nms(
            detections,
            _post_cfg(filter_invalid_proposals=False),
            return_audit=True,
        )
    assert called is False
    assert caught.value.audit["aggregate"]["invalid_detections"] == 1
    assert caught.value.audit["aggregate"]["filtered_detections"] == 0


def test_batched_nms_normalizes_class_indices_to_long(monkeypatch):
    captured = {}

    def fake_apply(segs, scores, cls_idxs, *args):
        captured["dtype"] = cls_idxs.dtype
        return segs, scores, cls_idxs

    monkeypatch.setattr(nms_module.NMSop, "apply", fake_apply)
    _, _, labels = nms_module.batched_nms(
        torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        torch.tensor([0.5], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        use_soft_nms=False,
        multiclass=False,
        max_seg_num=10,
    )
    assert captured["dtype"] == torch.long
    assert labels.dtype == torch.long


def test_real_soft_nms_and_voting_path_keeps_finite_outputs():
    detections = {
        "video": [
            {"segment": [0.0, 1.0], "label": "action", "score": 0.9},
            {"segment": [0.1, 1.1], "label": "action", "score": 0.8},
            {"segment": [2.0, 3.0], "label": "action", "score": 0.7},
        ]
    }
    merged, audit = test_engine.apply_sliding_window_nms(
        detections,
        _post_cfg(
            nms={
                "use_soft_nms": True,
                "sigma": 0.7,
                "max_seg_num": 100,
                "multiclass": False,
                "voting_thresh": 0.7,
            }
        ),
        return_audit=True,
    )
    assert merged["video"]
    assert audit["aggregate"]["post_nms_invalid_detections"] == 0
    assert all(
        torch.isfinite(torch.tensor(item["segment"] + [item["score"]])).all()
        for item in merged["video"]
    )


def test_real_soft_nms_uses_the_exact_production_multiclass_contract():
    config = Config.fromfile(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "adatad"
        / "thumos"
        / "phystime_g1a_physical_metric_native_j192_p0_replay.py",
        lazy_import=False,
    )
    production_nms = dict(config.post_processing.nms)
    assert production_nms["multiclass"] is True
    detections = {
        "video": [
            {"segment": [0.0, 1.0], "label": "class_a", "score": 0.9},
            {"segment": [0.05, 1.05], "label": "class_a", "score": 0.8},
            {"segment": [0.0, 1.0], "label": "class_b", "score": 0.85},
        ]
    }
    merged, audit = test_engine.apply_sliding_window_nms(
        detections,
        _post_cfg(nms=production_nms),
        return_audit=True,
    )
    assert audit["policy"]["filter_invalid_proposals"] is True
    assert audit["aggregate"]["post_nms_invalid_detections"] == 0
    assert {"class_a", "class_b"}.issubset(
        {item["label"] for item in merged["video"]}
    )


def test_invalid_post_nms_output_fails_before_evaluator(monkeypatch):
    def invalid_nms(segs, scores, labels, **kwargs):
        return (
            torch.tensor([[float("nan"), 1.0]], dtype=torch.float32),
            scores[:1],
            labels[:1],
        )

    monkeypatch.setattr(test_engine, "batched_nms", invalid_nms)
    with pytest.raises(RuntimeError, match="NMS produced 1 invalid"):
        test_engine.apply_sliding_window_nms(
            {
                "video": [
                    {
                        "segment": [0.0, 1.0],
                        "label": "action",
                        "score": 0.5,
                    }
                ]
            },
            _post_cfg(),
        )


def test_pre_cross_window_artifact_roundtrip_is_full_precision(tmp_path):
    artifact_path = tmp_path / "pre_cross_window_detections.json.gz"
    payload = {
        "schema_version": "opentad_pre_cross_window_detections_v1",
        "artifact_kind": "pre_cross_window_nms_full_precision_detections",
        "results": {
            "video": [
                {
                    "segment": [0.123456789, 1.987654321],
                    "label": "action",
                    "score": 0.87654321,
                }
            ]
        },
    }
    test_engine._atomic_write_json_gzip(artifact_path, payload)
    loaded = replay.load_pre_cross_window_artifact(artifact_path)
    assert loaded == payload
    with gzip.open(artifact_path, "rt", encoding="utf-8") as handle:
        assert json.load(handle) == payload


def test_replay_runs_all_four_modes_from_the_same_frozen_input(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(test_engine, "batched_nms", _threshold_nms)
    monkeypatch.setattr(
        replay,
        "apply_sliding_window_nms",
        test_engine.apply_sliding_window_nms,
    )
    monkeypatch.setattr(
        replay,
        "evaluate_predictions",
        lambda cfg, payload: {
            "average_mAP": len(payload["results"]["video"]) / 10.0,
            "mAP@0.7": len(payload["results"]["video"]) / 20.0,
        },
    )
    cfg = SimpleNamespace(post_processing=_post_cfg(), evaluation={})
    artifact = {
        "results": {
            "video": [
                {"segment": [0.0, 1.0], "label": "action", "score": 0.9},
                {
                    "segment": [0.334, 1.334],
                    "label": "action",
                    "score": 0.8,
                },
            ]
        }
    }
    reports = replay.run_replay(
        cfg=cfg,
        pre_cross_window_payload=artifact,
        output_dir=tmp_path / "replay",
        evaluation_epoch=59,
    )
    assert set(reports) == set(replay.MODE_SPECS)
    assert all(report["status"] == "completed" for report in reports.values())
    assert reports["legacy_unfiltered"]["prediction_count"] == 1
    assert reports["legacy_filtered"]["prediction_count"] == 1
    assert reports["fullprecision_unfiltered"]["prediction_count"] == 2
    assert reports["fullprecision_filtered"]["prediction_count"] == 2


def test_gathered_multi_rank_results_match_single_merged_input(monkeypatch):
    monkeypatch.setattr(test_engine, "batched_nms", _threshold_nms)
    rank_results = [
        {
            "video": [
                {"segment": [0.0, 1.0], "label": "action", "score": 0.9}
            ]
        },
        {
            "video": [
                {
                    "segment": [0.334, 1.334],
                    "label": "action",
                    "score": 0.8,
                }
            ]
        },
    ]

    def fake_all_gather_object(outputs, local_result):
        assert local_result == rank_results[0]
        outputs[:] = rank_results

    monkeypatch.setattr(
        test_engine.dist,
        "all_gather_object",
        fake_all_gather_object,
    )
    gathered, audit, pre_cross = test_engine.gather_ddp_results(
        2,
        rank_results[0],
        _post_cfg(),
        return_audit=True,
        return_pre_cross_window=True,
    )
    expected = test_engine.apply_sliding_window_nms(
        {
            "video": rank_results[0]["video"] + rank_results[1]["video"],
        },
        _post_cfg(),
    )
    assert gathered == expected
    assert pre_cross == {
        "video": rank_results[0]["video"] + rank_results[1]["video"],
    }
    assert audit["aggregate"]["input_detections"] == 2


def test_legacy_ema_replay_must_reproduce_source_predictions(tmp_path):
    source_prediction_path = tmp_path / "source_result.json"
    replay_prediction_path = (
        tmp_path
        / "replay"
        / "modes"
        / "legacy_unfiltered"
        / "result_detection.json"
    )
    replay_prediction_path.parent.mkdir(parents=True)
    prediction = {
        "results": {
            "video": [
                {
                    "segment": [0.12, 1.23],
                    "label": "action",
                    "score": 0.8765,
                }
            ]
        },
        "evaluation_epoch": 59,
    }
    source_prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
    replay_prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
    source_completion = {
        "artifacts": {
            "predictions": {"path": str(source_prediction_path)}
        },
        "metrics": {"average_mAP": 0.5, "mAP@0.7": 0.25},
    }
    mode_report = {
        "status": "completed",
        "metrics": {
            "average_mAP": 0.5,
            "mAP@0.7": 0.25,
            "evaluation_epoch": 59,
        },
    }
    comparison = replay.compare_source_legacy_ema(
        source_completion=source_completion,
        mode_report=mode_report,
        replay_output_dir=tmp_path / "replay",
        weights_source="ema",
    )
    assert comparison["match"] is True
    assert comparison["predictions_match"] is True
    assert comparison["metrics_match"] is True
