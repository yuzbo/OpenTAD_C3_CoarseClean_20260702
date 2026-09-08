from dataclasses import replace

import pytest
import torch

from geosparse_ext.data import video_batch
from geosparse_ext.geometry import canonical_atoms, native_layout
from geosparse_ext.matrix import base
from geosparse_ext.routing import make_plan
from geosparse_ext.sparse import heavy_macs
from geosparse_research.heavy_cap import allocate_heavy_prefix
from test_geosparse_detector import batch, detector


@pytest.mark.parametrize('route', ['A', 'B', 'C'])
@torch.no_grad()
def test_actual_executor_consumes_capped_plan_and_preserves_detector_mask(route):
    torch.manual_seed(41)
    model = detector(route, query_length=16).eval()
    data = batch(); data['masks'][0, -3:] = False
    video = video_batch(data['inputs'], data['masks'], data['metas'], 'cpu')
    _, plan, _, _, _ = model.geosparse(video)
    original = plan.selected_native.clone()
    capped, receipt = allocate_heavy_prefix(plan, native_layout(video).valid, route, width=16, layers=2, max_fraction=.5)
    state, actual, _, _, _ = model.geosparse(video, forced_plan=capped)
    measured = heavy_macs(model.geosparse.encoder.trace, 16)
    assert measured == receipt['rows'][0]['allocated_heavy_macs'] <= receipt['rows'][0]['cap_macs']
    assert measured > 0 and measured < receipt['rows'][0]['full_valid_heavy_macs']
    assert torch.equal(actual.selected_native, capped.selected_native)
    assert torch.equal(state.valid, data['masks']) and torch.isfinite(state.features).all()
    assert torch.equal(plan.selected_native, original)
    assert not capped.learned_sample.any() and not capped.log_prob.any() and capped.budget_id.item() == -1
    assert all(not x.numel() for x in capped.sampling_order)


@pytest.mark.parametrize('route', ['A', 'B', 'C'])
@pytest.mark.parametrize('fraction', [.3, .4, .6])
def test_native_768x160_cap_uses_parent_cost_and_actual_tail(route, fraction):
    atoms = canonical_atoms(384, 10, 10)
    valid = torch.ones(2, 384, 10, 10, dtype=torch.bool)
    valid[0, 379:] = False
    valid[1] = False
    gain = torch.linspace(-2., 2., len(atoms)).repeat(2, 1)
    plan = make_plan(gain, atoms, valid, 8, (10, 10), base(route, selector='uniform'), training=False)
    capped, receipt = allocate_heavy_prefix(plan, valid, route, width=768, layers=12, max_fraction=fraction)
    for row in receipt['rows']:
        assert row['allocated_heavy_macs'] <= row['cap_macs']
        assert row['next_prefix_macs'] is None or row['next_prefix_macs'] > row['cap_macs']
    assert not capped.selected_native[1].any() and receipt['rows'][1]['full_valid_heavy_macs'] == 0
    assert not (capped.selected_native.reshape_as(valid) & ~valid).any()
    assert torch.equal(capped.predicted_gain, gain)


def test_equal_token_counts_do_not_define_a_heavy_budget():
    atoms = canonical_atoms(16, 2, 2)
    valid = torch.ones(1, 16, 2, 2, dtype=torch.bool)
    plan = make_plan(torch.zeros(1, 16), atoms, valid, 8, (2, 2), base('A', selector='uniform'), training=False)
    concentrated = torch.full((1, 16), -9.); concentrated[0, :2] = torch.tensor([2., 1.])
    distributed = torch.full((1, 16), -9.); distributed[0, [0, 8]] = torch.tensor([2., 1.])
    cap = 51200 / 524288
    _, a = allocate_heavy_prefix(replace(plan, predicted_gain=concentrated), valid, 'A', width=16, layers=2, max_fraction=cap)
    _, b = allocate_heavy_prefix(replace(plan, predicted_gain=distributed), valid, 'A', width=16, layers=2, max_fraction=cap)
    assert a['rows'][0]['selected_atoms'] == 1
    assert b['rows'][0]['selected_atoms'] == 2 and b['rows'][0]['allocated_heavy_macs'] == 51200


@pytest.mark.parametrize('route', ['A', 'B', 'C'])
def test_zero_full_and_infeasible_coarse_floor(route):
    atoms = canonical_atoms(16, 2, 2); valid = torch.ones(1, 16, 2, 2, dtype=torch.bool)
    plan = make_plan(torch.zeros(1, 16), atoms, valid, 8, (2, 2), base(route, selector='uniform'), training=False)
    full, record = allocate_heavy_prefix(plan, valid, route, width=16, layers=2, max_fraction=1.)
    assert full.selected_native.all() and record['rows'][0]['actual_heavy_fraction'] == 1
    if route == 'C':
        with pytest.raises(ValueError, match='INFEASIBLE'):
            allocate_heavy_prefix(plan, valid, route, width=16, layers=2, max_fraction=0.)
        coarse, record = allocate_heavy_prefix(plan, valid, route, width=16, layers=2, max_fraction=106496/524288)
        assert not coarse.selected_native.any() and record['rows'][0]['allocated_heavy_macs'] > 0
    else:
        zero, record = allocate_heavy_prefix(plan, valid, route, width=16, layers=2, max_fraction=0.)
        assert not zero.selected_native.any() and record['rows'][0]['allocated_heavy_macs'] == 0


def test_nonfinite_scores_and_overlapping_atoms_are_rejected():
    atoms = canonical_atoms(16, 2, 2); valid = torch.ones(1, 16, 2, 2, dtype=torch.bool)
    plan = make_plan(torch.zeros(1, 16), atoms, valid, 8, (2, 2), base('A', selector='uniform'), training=False)
    gain = plan.predicted_gain.clone(); gain[0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite signed'):
        allocate_heavy_prefix(replace(plan, predicted_gain=gain), valid, 'A', width=16, layers=2, max_fraction=.5)
    bad = atoms.clone(); bad[1] = bad[0]
    with pytest.raises(ValueError, match='partition'):
        allocate_heavy_prefix(replace(plan, atom_to_native=bad), valid, 'A', width=16, layers=2, max_fraction=.5)
