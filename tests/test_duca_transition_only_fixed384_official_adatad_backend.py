from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.bata.validate_duca_transition_only_fixed384_official_adatad_backend import validate_config


CONFIG = (
    "configs/adatad/thumos/"
    "duca_transition_only_fixed384_official_adatad_backend_full_train.py"
)
PROOF = Path("tools/bata/run_duca_transition_only_official_adatad_one_step_grad_proof.py")
TRAIN_ENTRYPOINT = Path("tools/train.py")
OPTIMIZER_SOURCE = Path("opentad/cores/optimizer.py")
ACTIONFORMER_SOURCE = Path("opentad/models/detectors/actionformer.py")
LAUNCHER = Path("scripts/run_duca_transition_only_fixed384_official_adatad_backend_gpu1.sh")
FORMAL_GATE = Path("tools/bata/run_duca_transition_only_formal_full_model_gate.py")


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
    assert summary["coarse_hidden_kind"] == "official_asformer_encoder_hidden"
    assert summary["trained_with_thumos_labels"] is True
    assert summary["trained_with_gt_segments"] is True
    assert summary["uses_labels_at_inference"] is False
    assert summary["uses_gt_at_inference"] is False
    assert summary["val_start_epoch"] >= 47
    assert summary["expected_steps_per_epoch"] == 100
    assert summary["expected_total_steps"] == 13200
    assert summary["workflow_epochs"] == summary["scheduler_epochs"] == 132
    assert summary["paper_claim_allowed"] is False


def test_gradient_proof_attributes_selector_gradient_to_real_detector_losses() -> None:
    text = PROOF.read_text(encoding="utf-8")

    assert 'detector_losses["cls_loss"] + detector_losses["reg_loss"]' in text
    assert 'detector_route["inputs"].square().mean()' not in text


def test_optimizer_parameter_freezing_happens_before_ddp_registration() -> None:
    train_text = TRAIN_ENTRYPOINT.read_text(encoding="utf-8")
    optimizer_text = OPTIMIZER_SOURCE.read_text(encoding="utf-8")

    prepare = "prepare_optimizer_parameter_freezing(cfg.optimizer, model, logger)"
    ddp = "model = DistributedDataParallel("
    assert prepare in train_text
    assert train_text.index(prepare) < train_text.index(ddp)
    get_groups_body = optimizer_text.split("def get_backbone_optim_groups", 1)[1]
    assert "param.requires_grad_(False)" not in get_groups_body


def test_direct_and_transition_p0_variants_share_coarse_component_lr_routing() -> None:
    text = ACTIONFORMER_SOURCE.read_text(encoding="utf-8")
    parameter_lr = text.split("def parameter_lr(name):", 1)[1].split("grouped = {}", 1)[0]

    assert "if not transition_only" not in parameter_lr
    assert "if name.startswith(coarse_prefix):" in parameter_lr


def test_transition_only_formal_launcher_requires_clean_tree_and_hashes_inputs() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "git status --porcelain --untracked-files=normal" in text
    assert text.index("git status --porcelain --untracked-files=normal") < text.index('cat > "${RUN_DIR}/manifest.json"')
    assert "sha256sum" in text
    for field in ("git_commit", "config_sha256", "source_sha256", "checkpoint_sha256"):
        assert f'"{field}"' in text
    assert "run_duca_transition_only_formal_full_model_gate.py" in text
    assert 'DUCA_RUN_FORMAL_FULL_MODEL_GATE="${DUCA_RUN_FORMAL_FULL_MODEL_GATE:-0}"' in text
    assert 'formal_full_model_gate.json' in text


def test_formal_gpu_gate_keeps_real_videomae_and_768_to_384_geometry() -> None:
    assert FORMAL_GATE.is_file()
    text = FORMAL_GATE.read_text(encoding="utf-8")

    assert "from tools.bata.run_duca_official_adatad_one_step_grad_proof import" not in text
    assert "model.backbone =" not in text
    assert "dense_window_size" in text
    assert "selected_count" in text
    assert "detector_only_loss" in text
    assert "backbone_adapter" in text


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
    assert summary["detector_route_loss_keys"] == ["cls_loss", "reg_loss"]
