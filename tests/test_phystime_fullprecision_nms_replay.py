import gzip
import json
from pathlib import Path

import pytest
import torch
from mmengine.config import Config, ConfigDict

import opentad.cores.test_engine as test_engine
import opentad.models.utils.post_processing.nms.nms as nms_module


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


def test_p0_config_disables_pre_cross_window_rounding():
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


def test_full_precision_and_legacy_rounding_take_different_nms_branches(
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
    assert full_audit["aggregate"][
        "pre_nms_rounding_changed_segment_values"
    ] == 0
    assert legacy_audit["aggregate"][
        "pre_nms_rounding_changed_segment_values"
    ] == 2


def test_invalid_proposals_are_filtered_and_audited(monkeypatch):
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
            {
                "segment": [0.0, 1.0],
                "label": "action",
                "score": float("nan"),
            },
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


def test_rounding_induced_zero_duration_is_filtered_before_nms(monkeypatch):
    observed = {}

    def capture_nms(segs, scores, labels, **kwargs):
        observed["count"] = len(segs)
        return segs, scores, labels

    monkeypatch.setattr(test_engine, "batched_nms", capture_nms)
    merged, audit = test_engine.apply_sliding_window_nms(
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
    assert audit["aggregate"]["rounding_induced_invalid_detections"] == 1


def test_unfiltered_invalid_input_fails_closed_before_nms(monkeypatch):
    called = False

    def forbidden_nms(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("NMS must not receive invalid proposals")

    monkeypatch.setattr(test_engine, "batched_nms", forbidden_nms)
    with pytest.raises(test_engine.InvalidProposalError) as caught:
        test_engine.apply_sliding_window_nms(
            {
                "video": [
                    {
                        "segment": [1.0, 1.0],
                        "label": "action",
                        "score": 0.5,
                    }
                ]
            },
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


def test_gathered_multi_rank_results_match_single_merged_input(monkeypatch):
    monkeypatch.setattr(test_engine, "batched_nms", _threshold_nms)
    rank_results = [
        {
            "video": [
                {
                    "segment": [0.0, 1.0],
                    "label": "action",
                    "score": 0.9,
                }
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
    expected_input = {
        "video": rank_results[0]["video"] + rank_results[1]["video"]
    }
    expected = test_engine.apply_sliding_window_nms(
        expected_input,
        _post_cfg(),
    )
    assert gathered == expected
    assert pre_cross == expected_input
    assert audit["aggregate"]["input_detections"] == 2


def test_eval_one_epoch_writes_direct_replay_artifact_contract(
    monkeypatch,
    tmp_path,
):
    class FakeSlidingWindowDataset:
        class_map = ["action"]
        data_list = [("video",)]

    class FakeLoader:
        dataset = FakeSlidingWindowDataset()

        def __iter__(self):
            return iter([{}])

    original_detection = {
        "segment": [0.123456789, 1.987654321],
        "label": "action",
        "score": 0.87654321,
    }

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, **kwargs):
            return {"video": [dict(original_detection)]}

    class FakeEvaluator:
        def evaluate(self):
            return {"average_mAP": 0.4126, "mAP@0.7": 0.149}

        def logging(self, logger):
            return None

    class FakeLogger:
        def info(self, message):
            return None

    def fake_all_gather_object(outputs, local_result):
        outputs[:] = [local_result]

    monkeypatch.setattr(
        test_engine,
        "SlidingWindowDataset",
        FakeSlidingWindowDataset,
    )
    monkeypatch.setattr(
        test_engine,
        "build_decode_replay_collector",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        test_engine.dist,
        "all_gather_object",
        fake_all_gather_object,
    )
    monkeypatch.setattr(test_engine, "batched_nms", _threshold_nms)
    monkeypatch.setattr(
        test_engine,
        "build_evaluator",
        lambda config: FakeEvaluator(),
    )
    monkeypatch.setenv("PHYSTIME_EXPECTED_COMMIT", "commit-v6")
    monkeypatch.setenv("PHYSTIME_EXPECTED_TREE", "tree-v6")

    cfg = ConfigDict(
        work_dir=str(tmp_path),
        inference=ConfigDict(save_raw_prediction=False),
        post_processing=_post_cfg(
            save_dict=True,
            save_pre_cross_window_detections=True,
            save_post_processing_audit=True,
        ),
        evaluation=ConfigDict(type="mAP"),
    )
    test_engine.eval_one_epoch(
        FakeLoader(),
        FakeModel(),
        cfg,
        FakeLogger(),
        rank=0,
        world_size=1,
        epoch=59,
    )

    pre_cross_path = tmp_path / "pre_cross_window_detections.json.gz"
    result_path = tmp_path / "result_detection.json"
    metrics_path = tmp_path / "evaluation_metrics.json"
    audit_path = tmp_path / "post_processing_audit.json"
    with gzip.open(pre_cross_path, "rt", encoding="utf-8") as handle:
        pre_cross = json.load(handle)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert pre_cross["schema_version"] == (
        "opentad_pre_cross_window_detections_v1"
    )
    assert pre_cross["artifact_kind"] == (
        "pre_cross_window_nms_full_precision_detections"
    )
    assert pre_cross["evaluation_epoch"] == 59
    assert pre_cross["git_commit"] == "commit-v6"
    assert pre_cross["git_tree"] == "tree-v6"
    assert pre_cross["results"] == {"video": [original_detection]}
    assert result["evaluation_epoch"] == 59
    assert metrics == {
        "average_mAP": 0.4126,
        "evaluation_epoch": 59,
        "mAP@0.7": 0.149,
    }
    assert audit["schema_version"] == "opentad_post_processing_audit_v1"
    assert audit["evaluation_epoch"] == 59
    assert audit["post_processing"]["schema_version"] == (
        "opentad_cross_window_nms_audit_v1"
    )
    assert audit["pre_cross_window_artifact"]["path"] == str(
        pre_cross_path.resolve()
    )
    assert audit["pre_cross_window_artifact"]["sha256"] == (
        test_engine._sha256_file(pre_cross_path)
    )
