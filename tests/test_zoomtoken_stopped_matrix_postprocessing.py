import pytest
import torch
from mmengine.config import ConfigDict

from opentad.models.detectors.single_stage import SingleStageDetector
from tools.bata.continuous_roi_s2_v3_full200_compute_eval import (
    build_prediction_bundle_payload,
    prediction_rows_to_objects,
)
from tools.bata.continuous_roi_s2_v3_full200_compute_infer import post_nms_with_prediction_uids


def test_official_clipping_rounding_nms_predictions_survive_bundle_validation():
    classes = ["Action", "Other"]
    proposals = [torch.tensor([[-2.0, -1.0], [120.0, 125.0], [0.0, 0.004], [20.0, 40.0]])]
    scores = [torch.tensor([[0.95, 0.0], [0.85, 0.0], [0.7, 0.0], [0.6, 0.0]])]
    meta = dict(video_name="v000", fps=-1, resize_length=100, duration=10.0)
    rows = SingleStageDetector.post_processing(
        None, (proposals, scores), [meta],
        ConfigDict(sliding_window=True, pre_nms_thresh=0.1), classes,
    )["v000"]
    for index, row in enumerate(rows):
        row["prediction_uid"] = ["v000", 0, index, classes.index(row["label"])]
    videos = [f"v{index:03d}" for index in range(211)]
    results = post_nms_with_prediction_uids(
        {"v000": rows}, video_order=videos,
        nms_config=dict(use_soft_nms=True, multiclass=True, sigma=0.5,
                        min_score=0.001, max_seg_num=2000),
    )
    assert results["v000"] == rows
    predictions = prediction_rows_to_objects("v000", results["v000"], class_map=classes)
    assert sum(row.start == row.end for row in predictions) == 3
    assert [row.score for row in predictions] == pytest.approx([0.95, 0.85, 0.7, 0.6])
    payload = build_prediction_bundle_payload(
        arm="D160", seed=4407, population_manifest_sha256="synthetic-population",
        video_order=videos, results=results, class_map=classes,
    )
    assert payload["prediction_count"] == 4
    assert payload["results"]["v000"] == rows
