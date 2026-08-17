import torch

from opentad.models.selectors.pc_ot_mras_prebackbone_frame_selector import (
    PCOTMRASPreBackboneFrameSelector, pack_variable_temporal_batch, temporal_work_units,
)


def test_semantic_budget_receipt_is_per_video_for_batch_gt_one():
    selector = PCOTMRASPreBackboneFrameSelector.__new__(PCOTMRASPreBackboneFrameSelector)
    selector.dynamic_k_min = 2
    selector.dynamic_k_max = 8
    selector.dynamic_k_step = 2
    selector.dynamic_k_threshold = 0.5
    out = selector._semantic_budget_from_predictions(
        torch.tensor([[.9, .0, .0, .0], [.0, .8, .8, .0]]),
        torch.zeros(2, 4), torch.ones(2, 4, dtype=torch.bool))
    assert len(out["metadata"]) == 2
    assert [r["requested_k"] for r in out["metadata"]] == [2, 2]
    assert all(set(("requested_k", "effective_k", "executed_k", "min_k", "max_k")) <= r.keys() for r in out["metadata"])


def test_variable_k_is_packed_without_padding_and_work_differs():
    x = torch.randn(2, 8, 3)
    lengths = torch.tensor([2, 6])
    rows = pack_variable_temporal_batch(x, lengths)
    assert [row.shape[0] for row in rows] == [2, 6]
    assert temporal_work_units(torch.tensor([2]), channels=3) != temporal_work_units(torch.tensor([6]), channels=3)
