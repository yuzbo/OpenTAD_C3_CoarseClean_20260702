import pytest
import torch

from opentad.models.chronotransport import ChronoAction, ScheduleQuantileRiskPredictor


def test_r2_window_head_has_fixed_d23_encoder_and_mean_max_pooling():
    predictor = ScheduleQuantileRiskPredictor(signal_dims=6, num_groups=3, hidden_dims=64)
    assert predictor.action_embedding.embedding_dim == 8
    assert predictor.group_embedding.embedding_dim == 8
    assert predictor.action_embedding.weight.data_ptr() != predictor.group_embedding.weight.data_ptr()
    assert predictor.cell_input_dims == 23
    assert predictor.cell_encoder[0].in_features == 23
    assert predictor.cell_encoder[0].out_features == 64
    assert predictor.cell_encoder[2].in_features == 64
    assert predictor.cell_encoder[2].out_features == 64
    assert predictor.window_head[0].normalized_shape == (128,)
    assert predictor.window_head[1].in_features == 128

    signals = torch.randn(2, 48, 3, 6)
    actions = torch.randint(0, 3, (16, 48, 3))
    actions[:, 0] = int(ChronoAction.RECOMPUTE)
    output = predictor(signals, actions)
    assert output.shape == (2, 16)
    assert torch.isfinite(output).all()
    assert torch.all(output >= 0)


def test_true_age_feature_is_not_clamped_to_transport_embedding_cap():
    actions = torch.full((1, 48, 1), int(ChronoAction.HOLD), dtype=torch.long)
    actions[:, 0] = int(ChronoAction.RECOMPUTE)
    age = ScheduleQuantileRiskPredictor.candidate_age(actions)
    assert age[0, -1, 0, 0].item() == 47
    normalized = ScheduleQuantileRiskPredictor.normalized_candidate_age(actions)
    assert normalized[0, -1, 0, 0].item() == pytest.approx(47 / 48)


def test_window_head_gradients_reach_signals_and_both_embeddings():
    predictor = ScheduleQuantileRiskPredictor(signal_dims=6, num_groups=3, hidden_dims=64)
    signals = torch.randn(1, 48, 3, 6, requires_grad=True)
    actions = torch.randint(0, 3, (2, 48, 3))
    actions[:, 0] = int(ChronoAction.RECOMPUTE)
    predictor(signals, actions).sum().backward()
    assert signals.grad is not None and signals.grad.abs().sum() > 0
    assert predictor.action_embedding.weight.grad is not None
    assert predictor.group_embedding.weight.grad is not None


def test_conformal_ranks_are_exact_for_gate3_sizes():
    for size, expected_rank in ((30, 28), (140, 127)):
        prediction = torch.zeros(size)
        target = torch.arange(1, size + 1, dtype=torch.float32)
        value = ScheduleQuantileRiskPredictor.conformal_offset(prediction, target, coverage=0.9)
        assert value.item() == expected_rank
