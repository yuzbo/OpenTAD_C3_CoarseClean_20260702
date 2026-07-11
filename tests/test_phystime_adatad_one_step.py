import torch
import torch.nn as nn

from opentad.models.detectors.phystime_tad import PhysTimeTAD
from opentad.models.utils.post_processing import convert_to_seconds


class TinyAdapterBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = nn.Conv3d(3, 8, kernel_size=1)

    def forward(self, frames, masks=None):
        frames = frames[:, 0]
        return self.adapter(frames).mean(dim=(-1, -2))


def build_tiny_model():
    model = PhysTimeTAD(
        projection=dict(
            type="PhysTimeMeasureProjection",
            in_channels=8,
            out_channels=8,
            attention_channels=4,
            base_spacing_sec=0.5,
            num_levels=2,
        ),
        rpn_head=dict(
            type="PhysTimeHead",
            num_classes=2,
            in_channels=8,
            feat_channels=8,
            num_convs=1,
            regression_ranges_sec=[(0.0, 2.0), (2.0, 1.0e8)],
            loss_normalizer=10,
            endpoint_loss_weight=0.25,
            loss=dict(cls_loss=dict(type="FocalLoss"), reg_loss=dict(type="DIOULoss")),
        ),
    )
    model.backbone = TinyAdapterBackbone()
    return model


def make_batch():
    observation_count = 16
    timestamps = [0.25 + 0.5 * index for index in range(observation_count)]
    supports = [[value - 0.25, value + 0.25] for value in timestamps]
    return dict(
        inputs=torch.randn(1, 1, 3, observation_count, 8, 8),
        masks=torch.ones(1, observation_count, dtype=torch.bool),
        metas=[
            dict(
                video_name="synthetic",
                duration=8.0,
                fps=20.0,
                snippet_stride=4,
                irregular_native_axis=True,
                remap_gt_to_selected_axis=False,
                gt_remapped_to_selected_axis=False,
                gt_time_unit="seconds",
                prediction_time_unit="seconds",
                phystime_timestamps_sec=timestamps,
                phystime_support_intervals_sec=supports,
                phystime_duration_sec=8.0,
                phystime_domain_start_sec=0.0,
                phystime_domain_end_sec=8.0,
                phystime_support_provenance="original_raw_dense_cells",
            )
        ],
        gt_segments=[torch.tensor([[2.0, 4.0]])],
        gt_labels=[torch.tensor([1])],
    )


def test_phystime_adatad_cost_reaches_adapter_projection_and_all_head_branches():
    model = build_tiny_model().train()
    losses = model(return_loss=True, **make_batch())

    assert torch.isfinite(losses["cost"])
    losses["cost"].backward()
    required = {
        "adapter": model.backbone.adapter.weight,
        "projection": model.projection.level_attentions[0].value_proj.weight,
        "classification": model.rpn_head.cls_head.weight,
        "regression": model.rpn_head.reg_head.weight,
        "endpoint": model.rpn_head.endpoint_head.weight,
    }
    for name, parameter in required.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_phystime_adatad_predictions_stay_in_seconds_and_export_original_frames():
    model = build_tiny_model().eval()
    batch = make_batch()

    with torch.no_grad():
        proposals, scores = model.forward_test(
            batch["inputs"],
            batch["masks"],
            batch["metas"],
        )

    assert torch.isfinite(proposals[0]).all()
    assert torch.isfinite(scores[0]).all()
    assert batch["metas"][0]["prediction_time_unit"] == "seconds"
    clamped_seconds = convert_to_seconds(proposals[0].clone(), batch["metas"][0])
    assert torch.all(clamped_seconds >= 0.0)
    assert torch.all(clamped_seconds <= 8.0)
    original_frame_numbers = torch.round(clamped_seconds * batch["metas"][0]["fps"])
    assert torch.equal(original_frame_numbers, torch.round(clamped_seconds * 20.0))
    assert "selected_axis_to_true_time_dense_index" not in batch["metas"][0]
