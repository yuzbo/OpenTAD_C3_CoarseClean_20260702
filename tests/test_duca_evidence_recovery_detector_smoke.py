"""Smoke test to verify full detector instantiation and forward pass for all 8 DUCA Evidence Recovery arms."""
import pytest

try:
    import torch
except OSError as exc:
    pytest.skip(f"torch runtime unavailable in this environment: {exc}", allow_module_level=True)
from mmengine.config import Config

import opentad.datasets  # noqa: F401
import opentad.models  # noqa: F401
from opentad.models.builder import build_detector
from opentad.models.duca.structured_selection import exact_uniform_positions


CONFIG_PATHS = [
    "configs/adatad/thumos/duca_evidence_recovery_matched_h65_60.py",
    "configs/adatad/thumos/duca_evidence_recovery_full.py",
    "configs/adatad/thumos/duca_evidence_recovery_no_coverage.py",
    "configs/adatad/thumos/duca_evidence_recovery_no_time.py",
    "configs/adatad/thumos/duca_evidence_recovery_no_robust.py",
    "configs/adatad/thumos/duca_evidence_recovery_no_merge.py",
    "configs/adatad/thumos/duca_evidence_recovery_no_recovery.py",
    "configs/adatad/thumos/duca_evidence_recovery_h65_selection.py",
]


@pytest.mark.parametrize("cfg_path", CONFIG_PATHS)
def test_detector_forward_train_and_test(cfg_path):
    """Verify that every arm config builds cleanly and executes forward_train and forward_test."""
    cfg = Config.fromfile(cfg_path)
    
    # Build detector
    detector = build_detector(cfg.model)
    detector.eval()



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = detector.to(device)

    # Synthetic batch: 1 sample, 1 clip, 3 channels, 768 frames, 32x32 resolution (divisible by patch_size=16)
    B, C, T, H, W = 1, 3, 768, 32, 32
    inputs = torch.randn(B, 1, C, T, H, W, device=device)

    masks = torch.ones(B, T, dtype=torch.bool, device=device)
    metas = [{"video_name": "test_video_0", "fps": 25.0, "duration": 30.0, "snippet_stride": 4, "offset_frames": 0}]
    if cfg.model.frame_selector.get("use_h65_selection", False):
        h65_positions = exact_uniform_positions(T, cfg.model.frame_selector.budget).tolist()
        metas[0].update(
            {
                "bata_selected_dense_indices": h65_positions,
                "irregular_selected_positions": h65_positions,
                "selected_dense_indices": h65_positions,
                "irregular_dense_valid_len": T,
                "selected_valid_len": cfg.model.frame_selector.budget,
            }
        )
    gt_segments = [torch.tensor([[10.0, 50.0], [100.0, 200.0]], device=device)]
    gt_labels = [torch.tensor([1, 2], device=device)]

    infer_cfg = getattr(cfg, "inference", None)
    post_cfg = getattr(cfg, "post_processing", None)
    ext_cls = [f"class_{idx}" for idx in range(cfg.model.rpn_head.num_classes)]

    # 1. Forward Test (Inference / Detection)
    with torch.no_grad():
        test_out = detector(
            inputs,
            masks=masks,
            metas=metas,
            return_loss=False,
            infer_cfg=infer_cfg,
            post_cfg=post_cfg,
            ext_cls=ext_cls,
        )
        assert test_out is not None

    # 2. Forward Train
    detector.train()
    train_losses = detector(
        inputs,
        masks=masks,
        metas=metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
        return_loss=True,
    )
    assert isinstance(train_losses, dict)
    assert "cost" in train_losses
    assert torch.isfinite(train_losses["cost"]).item()

    del detector, inputs, masks, metas, gt_segments, gt_labels, test_out, train_losses
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

