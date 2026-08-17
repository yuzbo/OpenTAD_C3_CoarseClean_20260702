from types import SimpleNamespace

import torch

from opentad.models.detectors.actionformer import ActionFormer
from opentad.models.detectors.single_stage import SingleStageDetector


class _SpyBackbone:
    def __init__(self):
        self.temporal_inputs = []

    def __call__(self, frames, masks=None):
        time_dim = 3 if frames.ndim == 6 else 2
        k = int(frames.shape[time_dim])
        self.temporal_inputs.append(k)
        batch = int(frames.shape[0])
        return torch.ones(batch, 2, k)


def test_dynamic_heavy_recovery_uses_real_mixed_k_buckets_and_accounting():
    spy = _SpyBackbone()
    owner = SimpleNamespace(backbone=spy)
    inputs = torch.zeros(2, 1, 3, 6, 2, 2)
    masks = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    metas = [
        {"pc_ot_mras_prebackbone_dynamic_budget": {"requested_k": 2}},
        {"pc_ot_mras_prebackbone_dynamic_budget": {"requested_k": 4}},
    ]
    features, out_masks = ActionFormer._forward_backbone_dynamic_k(owner, inputs, masks, metas)
    assert spy.temporal_inputs == [2, 4]
    assert features.shape == (2, 2, 4)
    assert out_masks.tolist() == [[True, True, False, False], [True, True, True, True]]
    assert [m["pc_ot_mras_prebackbone_dynamic_budget"]["executed_k"] for m in metas] == [2, 4]


def test_irregular_selected_positions_reach_postprocessing_conversion():
    detector = SingleStageDetector.__new__(SingleStageDetector)
    predictions = (
        [torch.tensor([[0.5, 1.5]])],
        [torch.tensor([[0.9]])],
    )
    meta = {
        "video_name": "nonuniform",
        "fps": 10.0,
        "snippet_stride": 1.0,
        "offset_frames": 0.0,
        "window_start_frame": 0.0,
        "duration": 20.0,
        "irregular_native_axis": False,
        "irregular_selected_positions": [0.0, 3.0, 9.0],
        "irregular_selected_valid_len": 12.0,
    }
    cfg = SimpleNamespace(pre_nms_thresh=0.001, pre_nms_topk=2000, sliding_window=False, nms=None)
    result = detector.post_processing(predictions, [meta], cfg, ["action"])
    assert result["nonuniform"][0]["segment"] == [0.15, 0.6]
