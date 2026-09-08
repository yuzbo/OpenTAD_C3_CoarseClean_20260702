import copy

import pytest
import torch

from geosparse_ext.data import video_batch
from geosparse_research.heavy_cap_precheck import probe_executor
from test_geosparse_detector import batch, detector


@pytest.mark.parametrize('budget_mode', ['fixed', 'dynamic'])
def test_real_executor_precheck_covers_zero_half_full_and_original_tail(budget_mode):
    torch.manual_seed(113)
    model = detector('A', query_length=16).eval()
    model.geosparse.config['budget_mode'] = budget_mode
    original = copy.deepcopy(model.geosparse.config)
    saved = {name: value.clone() for name, value in model.state_dict().items()}
    data = batch()
    data['masks'][0, -1] = False
    video = video_batch(data['inputs'], data['masks'], data['metas'], 'cpu')
    result = probe_executor(model.geosparse, video)
    assert result['status'] == 'PASS' and not result['uses_torch_dispatch']
    zero, half, full = result['cases']
    assert zero['actual_heavy_macs'] == 0
    assert 0 < half['actual_heavy_macs'] <= result['source_full_heavy_macs'] // 2
    assert full['actual_heavy_macs'] == result['source_full_heavy_macs'] and full['full_reference_equal']
    assert all(row['valid_positions'] == 15 for row in result['cases'])
    assert model.geosparse.config == original
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[name], atol=0, rtol=0)


def test_source_full_shortcut_is_rejected_instead_of_ignoring_cap():
    model = detector('A').eval()
    model.geosparse.config.update(selector='none', budget=1.)
    data = batch()
    video = video_batch(data['inputs'], data['masks'], data['metas'], 'cpu')
    with pytest.raises(ValueError, match='learned selector'):
        probe_executor(model.geosparse, video)


def test_config_restored_on_actual_forward_exception(monkeypatch):
    model = detector('A').eval()
    model.geosparse.config['budget_mode'] = 'dynamic'
    original = copy.deepcopy(model.geosparse.config)
    data = batch()
    video = video_batch(data['inputs'], data['masks'], data['metas'], 'cpu')
    def fail(*args, **kwargs):
        raise RuntimeError('injected executor failure')
    monkeypatch.setattr(model.geosparse.encoder, 'forward', fail)
    with pytest.raises(RuntimeError, match='injected executor failure'):
        probe_executor(model.geosparse, video)
    assert model.geosparse.config == original
