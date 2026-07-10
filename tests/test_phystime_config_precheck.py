import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from mmengine.config import Config

from opentad.models import build_detector


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "adatad" / "thumos" / "phystime_tad_i3d_feature_gate0b.py"


def test_formal_config_builds_independent_phystime_detector_without_selector_route():
    cfg = Config.fromfile(CONFIG_PATH)
    model = build_detector(cfg.model)
    serialized = repr(cfg.model).lower()

    assert model.__class__.__name__ == "PhysTimeTAD"
    assert not hasattr(model, "frame_selector")
    for forbidden in ("actionness", "budget", "ledger", "selected_axis", "x3d", "slowfast"):
        assert forbidden not in serialized
    for split in ("train", "val", "test"):
        pipeline_types = [step["type"] for step in cfg.dataset[split]["pipeline"]]
        if split == "train":
            assert "BuildPairedPhysTimeFeatureViews" in pipeline_types
        else:
            assert "SampleIrregularFeatureObservations" in pipeline_types
            assert "BuildPhysTimeFeatureGeometry" in pipeline_types


def test_precheck_cli_emits_auditable_json(tmp_path):
    output_path = tmp_path / "phystime_precheck.json"
    command = [
        sys.executable,
        str(ROOT / "tools" / "bata" / "run_phystime_tad_precheck.py"),
        "--config",
        str(CONFIG_PATH),
        "--device",
        "cpu",
        "--output",
        str(output_path),
    ]

    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["precheck_pass"] is True
    assert report["model_type"] == "PhysTimeTAD"
    assert report["prediction_time_unit"] == "seconds"
    assert report["optimizer_coverage"] is True
    assert report["projection_gradient_nonzero"] is True
    assert report["classification_gradient_nonzero"] is True
    assert report["regression_gradient_nonzero"] is True
    assert report["endpoint_gradient_nonzero"] is True
    assert report["paired_discretization_loss_active"] is True
    assert report["uses_selector"] is False
    assert report["uses_ledger"] is False
