from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from mmengine.config import Config

from opentad.models import build_detector


ROOT = Path(__file__).resolve().parents[1]
PHYSTIME_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "phystime_tad_i3d_feature_gate0b.py"
SELECTED_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "selected_axis_actionformer_i3d_k384.py"
TIMESTAMP_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "timestamp_selected_axis_actionformer_i3d_k384.py"


def test_phystime_config_accepts_audited_environment_overrides(monkeypatch):
    monkeypatch.setenv("PHYSTIME_OBSERVATION_COUNT", "192")
    monkeypatch.setenv("PHYSTIME_PAIRED_TRAIN", "0")
    monkeypatch.setenv("PHYSTIME_OBSERVATION_MEASURE", "point_gaussian")
    monkeypatch.setenv("PHYSTIME_DISCRETIZATION_WEIGHT", "0")
    monkeypatch.setenv("PHYSTIME_FEATURE_PATH", "/audit/features")
    monkeypatch.setenv("PHYSTIME_ANNOTATION_PATH", "/audit/annotations.json")
    monkeypatch.setenv("PHYSTIME_CLASS_MAP", "/audit/classes.txt")

    cfg = Config.fromfile(PHYSTIME_CONFIG)
    model = build_detector(cfg.model)
    train_types = [step["type"] for step in cfg.dataset.train.pipeline]

    assert cfg.observation_count == 192
    assert cfg.dataset.train.data_path == "/audit/features"
    assert cfg.dataset.train.ann_file == "/audit/annotations.json"
    assert cfg.dataset.train.class_map == "/audit/classes.txt"
    assert "BuildPairedPhysTimeFeatureViews" not in train_types
    assert "SampleIrregularFeatureObservations" in train_types
    assert model.projection.level_attentions[0].observation_measure == "point_gaussian"
    assert model.discretization_loss_weight == 0.0


@pytest.mark.parametrize(
    "config_path, expected_channels, append_timestamps",
    [
        (SELECTED_CONFIG, 2048, False),
        (TIMESTAMP_CONFIG, 2052, True),
    ],
)
def test_selected_axis_baseline_configs_are_buildable_and_explicit(
    monkeypatch, config_path, expected_channels, append_timestamps
):
    monkeypatch.setenv("PHYSTIME_FEATURE_PATH", "/audit/features")
    monkeypatch.setenv("PHYSTIME_ANNOTATION_PATH", "/audit/annotations.json")
    monkeypatch.setenv("PHYSTIME_CLASS_MAP", "/audit/classes.txt")
    cfg = Config.fromfile(config_path)
    model = build_detector(cfg.model)
    transform = next(
        step for step in cfg.dataset.train.pipeline if step["type"] == "BuildSelectedAxisFeatureBaseline"
    )

    assert model.__class__.__name__ == "ActionFormer"
    assert cfg.model.projection.in_channels == expected_channels
    assert transform["append_timestamp_channels"] is append_timestamps
    assert cfg.dataset.train.data_path == "/audit/features"
    assert cfg.model.projection.max_seq_len == 384
