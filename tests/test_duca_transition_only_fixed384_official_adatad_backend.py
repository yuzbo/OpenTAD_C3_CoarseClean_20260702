from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.bata.validate_duca_transition_only_fixed384_official_adatad_backend import validate_config


CONFIG = (
    "configs/adatad/thumos/"
    "duca_transition_only_fixed384_official_adatad_backend_full_train.py"
)


def test_transition_only_fixed384_config_contract() -> None:
    summary = validate_config(CONFIG)

    assert summary["ok"] is True
    assert summary["task"] == "offline_temporal_action_detection"
    assert summary["selector_variant"] == "transition_only"
    assert summary["budget"] == 384
    assert summary["dense_window_size"] == 768
    assert summary["max_unselected_hole"] == 15
    assert summary["detector_type"] == "ActionFormer"
    assert summary["detector_head"] == "ActionFormerHead"
    assert summary["official_head_config_match"] is True
    assert summary["expected_steps_per_epoch"] == 100
    assert summary["expected_total_steps"] == 13200
    assert summary["workflow_epochs"] == summary["scheduler_epochs"] == 132
    assert summary["paper_claim_allowed"] is False


@pytest.mark.skipif(os.name == "nt", reason="Linux remote runs Torch/official-ASFormer proof")
def test_transition_only_official_actionformer_one_step_gradient_contract() -> None:
    if os.environ.get("DUCA_RUN_OFFICIAL_PROOF_TEST", "0") != "1":
        pytest.skip("set DUCA_RUN_OFFICIAL_PROOF_TEST=1 to run the official one-step proof")
    repo_root = os.environ.get("C3_OFFICIAL_ACTION_SEG_REPOS")
    if not repo_root or not (Path(repo_root) / "ASFormer" / "model.py").is_file():
        pytest.skip("C3_OFFICIAL_ACTION_SEG_REPOS with official ASFormer source is required")
    from tools.bata.run_duca_transition_only_official_adatad_one_step_grad_proof import run_proof

    summary = run_proof(
        CONFIG,
        temporal_len=16,
        budget=8,
        hidden_dim=16,
        feature_dim=16,
        spatial_size=16,
        device="cpu",
    )

    assert summary["ok"] is True
    assert summary["model_type"] == "ActionFormer"
    assert summary["detector_head_type"] == "ActionFormerHead"
    assert summary["selected_count"] == 8
    assert summary["max_unselected_hole_observed"] <= 1
    assert summary["optimizer_exact_coverage"] is True
    assert summary["train_forward"] is True
    assert summary["test_forward"] is True
    assert summary["detector_route_gradients"]["transition_scorer"] > 0.0
    assert summary["detector_route_gradients"]["coarse_probe"] == pytest.approx(0.0)
