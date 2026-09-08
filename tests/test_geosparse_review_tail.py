"""Review N02: actual 768-position B forward, with synthetic pixel inputs."""
import copy

import pytest
import torch

from geosparse_ext.data import video_batch
from geosparse_ext.evidence import spatial_slots
from geosparse_ext.geometry import native_layout
from test_geosparse_detector import batch, detector


def tail_batch():
    data = batch()
    t = 768
    data["inputs"] = torch.rand(1, 1, 3, t, 32, 32) * 255
    data["masks"] = torch.arange(t)[None] < 755
    data["metas"][0].update(duration=t / 25)
    data["metas"][0]["geosparse"].update(
        source_frame_id=list(range(t)), regular_s=(torch.arange(t) / 25).tolist(),
        intervals_s=torch.stack((torch.arange(t) / 25, (torch.arange(t) + 1) / 25), -1).numpy())
    return data


@pytest.mark.parametrize("temporal_atom", [1, 4, 8])
def test_acquisition_keeps_755_valid_frames_on_768_queries(monkeypatch, temporal_atom):
    torch.manual_seed(31)
    model = detector("B", input_positions=768, query_length=768).train()
    model.epoch = 6
    model.geosparse.config.update(selector="uniform", budget=0., quota="global_zero_allowed",
                                  allow_global_zero=True, temporal_atom_tubelets=temporal_atom)
    data = tail_batch()
    # Choose the last eligible atom: for atom4 it contains two valid tubelets
    # and two padding tubelets. Atom1 is the unchanged main-method control.
    monkeypatch.setattr(torch, "randint", lambda high, size: torch.tensor(high - 1))
    observations, evidence = [], []

    def capture(module, args, output):
        state, plan = output[:2]
        observations.append((state.valid.clone(), plan, copy.deepcopy(module.encoder.trace)))

    h1 = model.geosparse.register_forward_hook(capture)
    h2 = model.geosparse.receiver.register_forward_pre_hook(lambda module, args: evidence.append(args[2]))
    try:
        losses = model.forward_train(**data)
        losses["cost"].backward()
    finally:
        h1.remove()
        h2.remove()
    assert len(observations) == 2 and len(evidence) == 2
    native_valid = data["masks"].reshape(1, 384, 2).any(-1).repeat_interleave(4, 1)
    for mask, plan, trace in observations:
        assert torch.equal(mask, data["masks"])
        assert not (plan.selected_native.flatten(1) & ~native_valid).any()
        assert torch.equal(plan.execution_order[0], torch.where(plan.selected_native[0].flatten())[0])
        assert sum(row["qkv_tokens"] for row in trace) == int(plan.realized_token_count[0]) * 2
    added = observations[1][1]
    members = added.atom_to_native[model.latest_acquisition["atom"]]
    assert int(added.realized_token_count[0]) == int(native_valid[0, members].sum())
    assert torch.equal(evidence[1].valid, evidence[1].support_valid.any(-1))
    assert evidence[1].features.shape[1] == int(added.realized_token_count[0])
    assert torch.isfinite(evidence[1].features).all() and torch.isfinite(losses["cost"])
    assert "acquisition_loss" in losses


def test_evidence_rejects_padding_selection_even_in_a_forced_plan():
    data = tail_batch()
    video = video_batch(data["inputs"], data["masks"], data["metas"], "cpu")
    layout = native_layout(video)
    features = torch.randn(1, 16, 384, 2, 2)
    selected = torch.ones_like(layout.valid)
    evidence = spatial_slots(features, selected, layout, slots=4)
    assert evidence.features.shape[1] == 378 * 4
    assert evidence.valid.all() and evidence.support_valid.any(-1).all()
    assert int(evidence.support_valid[:, -4:].sum()) == 4  # odd final pair
