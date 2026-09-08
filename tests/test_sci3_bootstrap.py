import json

import numpy as np
import pandas as pd
import pytest

from opentad.evaluations.mAP import mAP
from geosparse_research.bootstrap import dataset_map, paired_video_map_bootstrap, resample_video_rows


def tables():
    gt = pd.DataFrame([("hit", 0., 1., 0), ("miss", 0., 1., 0)],
                      columns=["video-id", "t-start", "t-end", "label"])
    pred = pd.DataFrame([("background", 0., 1., 0, .9), ("hit", 0., 1., 0, .8)],
                        columns=["video-id", "t-start", "t-end", "label", "score"])
    return gt, pred


def test_official_dataset_ap_keeps_background_false_positives_and_misses(tmp_path):
    gt, pred = tables()
    filename = tmp_path / "gt.json"
    filename.write_text(json.dumps(dict(database={
        video: dict(subset="validation", annotations=[] if video == "background" else
                    [dict(label="action", segment=[0, 1])]) for video in ["hit", "miss", "background"]})))
    predictions = dict(results={"background": [dict(label="action", segment=[0, 1], score=.9)],
                               "hit": [dict(label="action", segment=[0, 1], score=.8)], "miss": []})
    official = mAP(str(filename), predictions, "validation", np.array([.3, .5, .7]), thread=1)
    official.evaluate()
    actual = dataset_map(gt, pred, [0], [.3, .5, .7])
    assert actual["average_map"] == official.average_mAP == .25
    # Average of the two annotated videos' AP would be .5, a different statistic.
    assert actual["average_map"] != .5


def test_repeated_video_is_an_independent_complete_cluster():
    gt, pred = tables()
    copies = ["hit", "hit", "miss", "background"]
    new_gt, new_pred = resample_video_rows(gt, copies), resample_video_rows(pred, copies)
    assert list(new_gt["video-id"]) == ["0", "1", "2"]
    assert set(new_pred["video-id"]) == {"0", "1", "3"}
    assert dataset_map(new_gt, new_pred, [0], [.5])["average_map"] == pytest.approx(4 / 9)


def test_paired_bootstrap_is_reproducible_and_identical_methods_have_zero_difference():
    gt, pred = tables()
    args = gt, pred, pred, ["hit", "miss", "background"]
    result = paired_video_map_bootstrap(*args, draws=30, seed=17)
    assert result == paired_video_map_bootstrap(*args, draws=30, seed=17)
    assert result["delta"] == 0 and result["interval"]["average_map"] == [0., 0.]
    assert result["valid_draws"] + result["undefined_draws"] == 30
    assert any("background" in row["sampled_videos"] for row in result["draws_ledger"])


def test_missing_classes_are_recorded_without_redrawing_or_dropping_class_from_map():
    gt, pred = tables()
    gt.loc[1, "label"] = 1
    result = paired_video_map_bootstrap(gt, pred, pred, ["hit", "miss", "background"], draws=30, seed=9)
    assert result["undefined_draws"] > 0 and result["valid_draws"] > 0
    expected = np.random.default_rng(9).integers(0, 3, size=(30, 3))
    for row, indices in zip(result["draws_ledger"], expected):
        assert row["sampled_videos"] == [["hit", "miss", "background"][index] for index in indices]
        if not row["defined"]:
            assert row["missing_classes"] and "delta" not in row


def test_empty_predictions_are_zero_recall_and_outside_split_is_rejected():
    gt, pred = tables()
    empty = pred.iloc[:0]
    assert dataset_map(gt, empty, [0], [.5])["average_map"] == 0
    result = paired_video_map_bootstrap(gt, empty, pred, ["hit", "miss", "background"], draws=5)
    assert result["delta"] == .25
    with pytest.raises(ValueError, match="outside"):
        paired_video_map_bootstrap(gt, pred, pred, ["hit", "miss"])
