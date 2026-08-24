from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from mmengine.config import Config

import opentad.datasets  # noqa: F401 - registers OpenTAD data transforms.
from opentad.models.backbones.vit_adapter import VisionTransformerAdapter
from opentad.models.builder import build_detector
from opentad.models.detectors.actionformer import ActionFormer
from opentad.models.duca.structured_selection import exact_uniform_positions
from opentad.models.utils.temporal_grid import (
    clip_relative_physical_time_mask,
    global_rank_clip_coordinates,
)
from tools.train import (
    _capture_epoch_boundary_data_loader_state,
    _restore_resumable_optimizer_update_count,
    _validate_epoch_boundary_data_loader_state,
)
from tools.bata.audit_duca_h65_terminal_checkpoint import audit_terminal_checkpoint


CONFIG = Path("configs/adatad/thumos/duca_h65_first_singleclock_cycle4.py")
GATE_ZERO_CONFIG = Path(
    "configs/adatad/thumos/duca_h65_first_singleclock_cycle4_gate_zero.py"
)
LAUNCHER = Path("scripts/run_duca_h65_matched_cycle4_n16r4.sbatch")
TERMINAL_EVAL_LAUNCHER = Path("scripts/run_duca_h65_singleclock_terminal_eval_n16r4.sbatch")


def _clock_detector_stub(gate_zero=False):
    detector = ActionFormer.__new__(ActionFormer)
    nn.Module.__init__(detector)
    detector.single_clock_gate_zero = bool(gate_zero)
    return detector


def _full_uniform(batch=1, valid_len=768, k=384):
    positions = exact_uniform_positions(valid_len, k).repeat(batch, 1)
    mask = torch.ones_like(positions, dtype=torch.bool)
    return positions, mask


def _tiny_clock_model():
    model = VisionTransformerAdapter(
        img_size=32,
        patch_size=16,
        embed_dims=24,
        depth=2,
        num_heads=3,
        mlp_ratio=2,
        num_frames=4,
        tubelet_size=2,
        return_feat_map=True,
        with_cp=False,
        adapter_index=[],
        relative_physical_time_residual=True,
    )
    model.eval()
    return model


def test_actionformer_reconstructs_exact_p_m_l_contract():
    detector = _clock_detector_stub()
    inputs = torch.zeros(2, 1, 3, 384, 2, 2)
    masks = torch.ones(2, 384, dtype=torch.bool)
    masks[1, 382:] = False
    first = exact_uniform_positions(768, 384).tolist()
    second = exact_uniform_positions(700, 382).tolist()
    metas = [
        {
            "irregular_selected_positions": first,
            "irregular_selected_valid_len": 384,
            "irregular_dense_valid_len": 768,
        },
        {
            "irregular_selected_positions": second,
            "irregular_selected_valid_len": 382,
            "irregular_dense_valid_len": 700,
        },
    ]
    contract = detector._single_clock_metadata(inputs, masks, metas)
    assert contract["irregular_selected_positions"].shape == (2, 384)
    assert contract["irregular_selected_positions"][1, 382:].tolist() == [-1, -1]
    assert torch.equal(contract["irregular_selected_mask"], masks)
    assert contract["irregular_dense_valid_len"].tolist() == [768, 700]


def test_actionformer_rejects_nonprefix_mask_and_position_count_drift():
    detector = _clock_detector_stub()
    inputs = torch.zeros(1, 1, 3, 384, 2, 2)
    masks = torch.ones(1, 384, dtype=torch.bool)
    masks[0, 10] = False
    meta = [{"irregular_selected_positions": list(range(383)), "irregular_dense_valid_len": 768}]
    with pytest.raises(ValueError, match="contiguous valid prefix"):
        detector._single_clock_metadata(inputs, masks, meta)
    masks[0, 10] = True
    with pytest.raises(ValueError, match="positions/mask mismatch"):
        detector._single_clock_metadata(inputs, masks, meta)


def test_singleclock_identity_audit_seals_selected_rgb_positions_and_mask():
    def collect(inputs, gate_zero):
        detector = _clock_detector_stub(gate_zero=gate_zero)
        detector.single_clock_admission = True
        detector.enable_single_clock_identity_audit()
        masks = torch.ones(1, 384, dtype=torch.bool)
        metas = [
            {
                "video_name": "video_validation_0000001",
                "window_start_frame": 0,
                "irregular_selected_positions": exact_uniform_positions(768, 384).tolist(),
                "irregular_selected_valid_len": 384,
                "irregular_dense_valid_len": 768,
            }
        ]
        detector._record_single_clock_identity(inputs, masks, metas)
        return detector, detector.single_clock_identity_payload()

    inputs = torch.arange(384 * 3 * 2 * 2, dtype=torch.float32).reshape(1, 1, 3, 384, 2, 2)
    detector_on, on = collect(inputs, gate_zero=False)
    _, gate_zero = collect(inputs.clone(), gate_zero=True)
    assert on["aggregate_sha256"] == gate_zero["aggregate_sha256"]
    assert on["sample_count"] == 1

    changed = inputs.clone()
    changed[..., 7, 0, 0] += 1
    _, changed_payload = collect(changed, gate_zero=False)
    assert changed_payload["aggregate_sha256"] != on["aggregate_sha256"]

    masks = torch.ones(1, 384, dtype=torch.bool)
    metas = [
        {
            "video_name": "video_validation_0000001",
            "window_start_frame": 0,
            "irregular_selected_positions": exact_uniform_positions(768, 384).tolist(),
            "irregular_selected_valid_len": 384,
            "irregular_dense_valid_len": 768,
        }
    ]
    with pytest.raises(RuntimeError, match="duplicate SingleClock identity sample"):
        detector_on._record_single_clock_identity(inputs, masks, metas)


def test_singleclock_identity_audit_normalizes_numpy_window_start():
    detector = _clock_detector_stub(gate_zero=False)
    detector.single_clock_admission = True
    detector.enable_single_clock_identity_audit()
    inputs = torch.zeros(1, 1, 3, 384, 2, 2)
    masks = torch.ones(1, 384, dtype=torch.bool)
    metas = [
        {
            "video_name": "video_validation_0000001",
            "window_start_frame": np.int64(768),
            "irregular_selected_positions": exact_uniform_positions(768, 384).tolist(),
            "irregular_selected_valid_len": 384,
            "irregular_dense_valid_len": 768,
        }
    ]

    detector._record_single_clock_identity(inputs, masks, metas)
    payload = detector.single_clock_identity_payload()

    assert payload["records"][0]["window_start_frame"] == 768
    assert payload["records"][0]["sample_id"].endswith("window_start_frame=768")


def test_tools_test_exposes_singleclock_identity_evidence_switch():
    text = Path("tools/test.py").read_text()
    assert 'parser.add_argument("--single-clock-identity-json"' in text
    assert "enable_single_clock_identity_audit" in text
    assert '"single_clock_identity_sha256"' in text


def test_global_coordinates_preserve_uniform_identity_and_short_padding():
    positions, mask = _full_uniform(batch=2)
    out = global_rank_clip_coordinates(
        positions,
        torch.tensor([768, 768]),
        selected_valid_mask=mask,
    )
    assert out["actual"].shape == (2, 24, 8)
    assert torch.equal(out["actual"], out["canonical"])
    assert bool(out["tubelet_valid_mask"].all())
    dense_per_clip = torch.tensor([768, 768]).repeat_interleave(24)
    assert clip_relative_physical_time_mask(
        out["actual"].flatten(0, 1),
        out["canonical"].flatten(0, 1),
        dense_valid_len=dense_per_clip,
        tubelet_valid_mask=out["tubelet_valid_mask"].flatten(0, 1),
        spatial_tokens=4,
    ) is None

    short_positions = torch.full((1, 384), -1, dtype=torch.long)
    short_positions[0, :382] = exact_uniform_positions(700, 382)
    short_mask = short_positions >= 0
    short = global_rank_clip_coordinates(
        short_positions,
        torch.tensor([700]),
        selected_valid_mask=short_mask,
    )
    assert short["tubelet_valid_mask"].shape == (1, 24, 8)
    assert short["tubelet_valid_mask"][0, -1, -1].item() is False
    assert short["actual"][0, -1, -1].item() == 0.0


def test_bounded_normalized_residual_shape_and_invalid_rows_zero():
    canonical = torch.arange(8, dtype=torch.float32).repeat(2, 1)
    actual = canonical.clone()
    actual[0, 1] += 10000.0
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[0, -1] = False
    residual = clip_relative_physical_time_mask(
        actual,
        canonical,
        dense_valid_len=torch.tensor([768, 768]),
        tubelet_valid_mask=valid,
        spatial_tokens=4,
    )
    assert residual.shape == (2, 1, 32, 32)
    assert float(residual.abs().max()) <= 1.0
    assert torch.count_nonzero(residual[1]).item() == 0
    assert torch.count_nonzero(residual[0, :, -4:, :]).item() == 0
    assert torch.count_nonzero(residual[0, :, :, -4:]).item() == 0


def test_tiny_videomae_singleclock_identity_gate_and_gradient():
    torch.manual_seed(4)
    model = _tiny_clock_model()
    assert model.blocks[0].relative_physical_time_scale is not None
    assert model.blocks[1].relative_physical_time_scale is None
    frames = torch.randn(1, 3, 4, 32, 32)
    canonical = torch.tensor([[0.5, 2.5]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    lengths = torch.tensor([8])

    with torch.no_grad():
        legacy = model(frames)
        uniform = model(
            frames,
            actual_positions=canonical,
            canonical_positions=canonical,
            dense_valid_len=lengths,
            tubelet_valid_mask=valid,
        )
    assert torch.equal(legacy, uniform)

    irregular = canonical.clone()
    irregular[0, 1] += 1.0
    with torch.no_grad():
        theta_zero = model(
            frames,
            actual_positions=irregular,
            canonical_positions=canonical,
            dense_valid_len=lengths,
            tubelet_valid_mask=valid,
        )
        gate_zero = model(
            frames,
            actual_positions=irregular,
            canonical_positions=canonical,
            dense_valid_len=lengths,
            tubelet_valid_mask=valid,
            relative_physical_time_gate_zero=True,
        )
    torch.testing.assert_close(theta_zero, gate_zero, rtol=0.0, atol=0.0)

    model.blocks[0].relative_physical_time_scale.data.fill_(0.2)
    output = model(
        frames,
        actual_positions=irregular,
        canonical_positions=canonical,
        dense_valid_len=lengths,
        tubelet_valid_mask=valid,
    )
    output.square().mean().backward()
    grad = model.blocks[0].relative_physical_time_scale.grad
    assert grad is not None and torch.isfinite(grad) and float(grad.abs()) > 0.0


def test_resolved_actionformer_config_builds_only_block0_clock(monkeypatch):
    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT", "stage1_epoch29.pth")
    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT_SHA256", "a" * 64)
    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT_EPOCH", "29")
    cfg = Config.fromfile(str(CONFIG))
    build_cfg = cfg.model
    build_cfg.backbone.custom.pretrain = None
    model = build_detector(build_cfg)
    assert isinstance(model, ActionFormer)
    assert model.single_clock_admission is True
    vit = model.backbone.model.backbone
    assert isinstance(vit, VisionTransformerAdapter)
    assert vit.relative_physical_time_residual is True
    assert vit.blocks[0].relative_physical_time_scale is not None
    assert all(block.relative_physical_time_scale is None for block in vit.blocks[1:])
    custom = {item["name"]: item for item in cfg.optimizer.backbone.custom}
    assert custom["relative_physical_time_scale"]["lr"] == pytest.approx(2e-4)
    assert custom["relative_physical_time_scale"]["weight_decay"] == 0.0


def test_config_and_launcher_freeze_training_and_environment_contract():
    config_text = CONFIG.read_text()
    assert "checkpoint_interval_epochs = 5" in config_text
    assert "keep_latest=3" in config_text and "final_ema=True" in config_text
    assert "relative_residual_h=lambda" not in config_text
    assert "single_clock_gate_zero=False" in config_text
    assert "tubelet_packed_runtime_route=dict(enabled=False)" in config_text
    assert "require_resumable_training_state=True" in config_text

    text = LAUNCHER.read_text()
    assert "/data/run01/sczc063/yuzibo/thumos14/raw_data/video" in text
    assert "PRECHECK_TARGET" in text and "STAGE1|STAGE2_OFF|STAGE2_ON" in text
    assert 'export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"' in text
    assert 'MASTER_PORT="${MASTER_PORT:-29500}"' in text
    assert "CUDA_VISIBLE_DEVICES" not in text
    assert 'if [[ "${PRE_RUN_ONLY:-0}" == 1 ]]; then' in text
    assert 'PRE_RUN_END_EPOCH="${DUCA_PRE_RUN_END_EPOCH:-1}"' in text
    assert 'PRE_RUN_RESUME="${DUCA_PRE_RUN_RESUME:-}"' in text
    assert 'workflow.end_epoch="$PRE_RUN_END_EPOCH"' in text
    assert "workflow.val_start_epoch=9999" in text


def test_terminal_eval_launcher_freezes_on_gate_zero_and_h65_off_reinference():
    text = TERMINAL_EVAL_LAUNCHER.read_text()
    assert "DUCA_CLOCK_CHECKPOINT_SHA256" in text
    assert "DUCA_H65_OFF_CHECKPOINT_SHA256" in text
    assert "DUCA_EVAL_COMMIT" in text
    assert "audit_duca_h65_terminal_checkpoint.py" in text
    assert "final_on" in text and "final_gate_zero" in text
    assert "ema_on" in text and "ema_gate_zero" in text
    assert "h65_off_final" in text and "h65_off_ema" in text
    assert "--single-clock-identity-json" in text
    assert 'run_eval final_gate_zero "$CLOCK_GATE_ZERO_CONFIG"' in text
    assert 'run_eval ema_gate_zero "$CLOCK_GATE_ZERO_CONFIG"' in text
    assert "single_clock_gate_zero=True" in GATE_ZERO_CONFIG.read_text()
    assert "state_dict_ema" in text


def test_epoch_boundary_data_loader_state_round_trip():
    class Dataset:
        def __len__(self):
            return 200

    class Sampler:
        num_replicas = 1
        rank = 0
        seed = 3407
        drop_last = True
        epoch = 4

    class Loader:
        dataset = Dataset()
        sampler = Sampler()
        batch_size = 2

    loader = Loader()
    snapshot = _capture_epoch_boundary_data_loader_state(loader, 4)
    assert snapshot["next_epoch"] == 5
    assert snapshot["resume_semantics"] == "derive_sampler_from_seed_and_next_epoch"
    _validate_epoch_boundary_data_loader_state(snapshot, loader, 4)
    snapshot["sampler_seed"] = 1
    with pytest.raises(RuntimeError, match="DataLoader contract"):
        _validate_epoch_boundary_data_loader_state(snapshot, loader, 4)


def test_resume_restores_cumulative_successful_optimizer_updates():
    audit = {"successful_optimizer_updates": 0}
    _restore_resumable_optimizer_update_count(
        {"successful_optimizer_updates": 2}, audit
    )
    assert audit["successful_optimizer_updates"] == 2
    with pytest.raises(RuntimeError, match="optimizer update count"):
        _restore_resumable_optimizer_update_count(
            {"successful_optimizer_updates": True}, audit
        )


def test_terminal_checkpoint_audit_accepts_complete_h65_off_recovery_state(tmp_path):
    stage1 = tmp_path / "stage1.pth"
    torch.save({"epoch": 29, "state_dict_ema": {}}, stage1)
    import hashlib

    stage1_sha = hashlib.sha256(stage1.read_bytes()).hexdigest()
    terminal = tmp_path / "terminal.pth"
    torch.save(
        {
            "epoch": 59,
            "state_dict": {},
            "state_dict_ema": {},
            "optimizer": {"state": {}, "param_groups": []},
            "scheduler": {"last_epoch": 6000},
            "grad_scaler": {},
            "rng_state": {"python": (), "numpy": (), "torch_cpu": torch.zeros(1), "torch_cuda": []},
            "data_loader_state": {"completed_epoch": 59, "next_epoch": 60},
            "successful_optimizer_updates": 6000,
        },
        terminal,
    )
    payload = audit_terminal_checkpoint(
        checkpoint_path=terminal,
        config_path="configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py",
        stage1_path=stage1,
        stage1_sha256=stage1_sha,
        family="h65_off",
    )
    assert payload["checkpoint_epoch"] == 59
    assert payload["successful_optimizer_updates"] == 6000
    assert payload["data_loader_next_epoch"] == 60
    assert payload["recovery_state_complete"] is True
    assert payload["recovery_protocol_deviation"] == []


def test_terminal_checkpoint_audit_records_legacy_h65_off_recovery_deviation(tmp_path):
    stage1 = tmp_path / "stage1.pth"
    torch.save({"epoch": 29, "state_dict_ema": {}}, stage1)
    import hashlib

    stage1_sha = hashlib.sha256(stage1.read_bytes()).hexdigest()
    terminal = tmp_path / "terminal.pth"
    registered_zero_clock = {
        "model.backbone.blocks.0.relative_physical_time_scale": torch.tensor(0.0)
    }
    torch.save(
        {
            "epoch": 59,
            "state_dict": registered_zero_clock,
            "state_dict_ema": registered_zero_clock,
            "optimizer": {"state": {}, "param_groups": []},
            "scheduler": {"last_epoch": 6000},
            "grad_scaler": {},
            "successful_optimizer_updates": 6000,
        },
        terminal,
    )
    payload = audit_terminal_checkpoint(
        checkpoint_path=terminal,
        config_path="configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py",
        stage1_path=stage1,
        stage1_sha256=stage1_sha,
        family="h65_off",
    )
    assert payload["checkpoint_epoch"] == 59
    assert payload["recovery_state_complete"] is False
    assert payload["rng_state_complete"] is False
    assert payload["data_loader_state_complete"] is False
    assert payload["data_loader_next_epoch"] is None
    assert payload["h65_off_scalar_identity"] == "registered_zero"
    assert payload["recovery_protocol_deviation"] == [
        "rng_state",
        "data_loader_state",
    ]


def test_terminal_checkpoint_audit_rejects_clock_on_without_recovery_state(tmp_path):
    stage1 = tmp_path / "stage1.pth"
    torch.save({"epoch": 29, "state_dict_ema": {}}, stage1)
    import hashlib

    stage1_sha = hashlib.sha256(stage1.read_bytes()).hexdigest()
    terminal = tmp_path / "terminal.pth"
    clock_state = {
        "model.backbone.relative_physical_time_scale": torch.tensor(0.1)
    }
    torch.save(
        {
            "epoch": 59,
            "state_dict": clock_state,
            "state_dict_ema": clock_state,
            "optimizer": {"state": {}, "param_groups": []},
            "scheduler": {"last_epoch": 6000},
            "grad_scaler": {},
            "successful_optimizer_updates": 6000,
        },
        terminal,
    )
    with pytest.raises(ValueError, match="missing rng_state"):
        audit_terminal_checkpoint(
            checkpoint_path=terminal,
            config_path="configs/adatad/thumos/duca_h65_first_singleclock_cycle4.py",
            stage1_path=stage1,
            stage1_sha256=stage1_sha,
            family="clock_on",
        )
