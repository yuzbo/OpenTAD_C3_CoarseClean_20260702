import torch

from opentad.models.duca.phase_fields import (
    adaptive_phase_quotas,
    compute_phase_fields,
    select_exact_uniform_positions,
    select_phase_positions,
)


def test_phase_fields_are_fp32_masked_and_constant_derivatives_are_zero():
    logits = torch.zeros(2, 16, dtype=torch.float16)
    valid = torch.ones(2, 16, dtype=torch.bool)
    valid[1, 10:] = False

    fields = compute_phase_fields(logits=logits, valid_mask=valid, sigmas=(1.5, 3.0))

    assert fields.core.dtype == torch.float32
    assert fields.onset.shape == logits.shape
    assert fields.offset.shape == logits.shape
    assert torch.count_nonzero(fields.onset).item() == 0
    assert torch.count_nonzero(fields.offset).item() == 0
    assert torch.count_nonzero(fields.curvature).item() == 0
    assert fields.valid_mask.equal(valid)
    assert fields.core[1, 10:].sum().item() == 0.0


def test_phase_fields_detect_onset_and_offset_polarity():
    logits = torch.full((1, 32), -4.0)
    logits[:, 10:20] = 4.0
    valid = torch.ones(1, 32, dtype=torch.bool)

    fields = compute_phase_fields(logits=logits, valid_mask=valid, sigmas=(1.5, 3.0))

    assert int(fields.onset[0].argmax().item()) in range(8, 13)
    assert int(fields.offset[0].argmax().item()) in range(18, 23)


def test_adaptive_quotas_sum_to_budget_and_selection_is_exact_k_sorted_unique():
    logits = torch.linspace(-2.0, 2.0, steps=48).unsqueeze(0)
    fields = compute_phase_fields(logits=logits, valid_mask=torch.ones(1, 48, dtype=torch.bool))

    quotas = adaptive_phase_quotas(fields, total_budget=24)[0]
    assert sum(quotas.values()) == 24
    assert all(value >= 0 for value in quotas.values())

    selection = select_phase_positions(
        fields,
        total_budget=24,
        quota_mode="adaptive",
        fixed_quota={"scaffold": 8, "onset": 4, "offset": 4, "core": 8},
        adaptive_minima={"scaffold": 6, "onset": 2, "offset": 2, "core": 4},
        adaptive_caps={"scaffold": 16, "onset": 10, "offset": 10, "core": 16},
        temporal_nms_radius=1,
    )

    assert selection["selected_positions"].shape == (1, 24)
    selected = selection["selected_positions"][0].tolist()
    assert selected == sorted(set(selected))
    assert selection["selected_mask"].sum().item() == 24
    assert sum(selection["phase_actual_counts"][0].values()) == 24
    assert selection["diagnostics"]["exact_k"] is True


def test_exact_uniform_selection_is_original_time_exact_k():
    valid = torch.ones(2, 20, dtype=torch.bool)
    valid[1, 12:] = False

    selection = select_exact_uniform_positions(valid, total_budget=8)

    assert selection["selected_positions"].shape == (2, 8)
    assert selection["selected_mask"].sum(dim=1).tolist() == [8, 8]
    for row, max_valid in zip(selection["selected_positions"], [20, 12]):
        values = row.tolist()
        assert values == sorted(set(values))
        assert min(values) >= 0
        assert max(values) < max_valid
