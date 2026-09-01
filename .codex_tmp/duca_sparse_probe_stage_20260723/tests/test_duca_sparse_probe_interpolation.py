from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from opentad.models.duca.acquisition import C3CoarseProbeActionnessSource
from tools.bata.aggregate_duca_sparse_probe_tad import aggregate


ROOT = Path(__file__).resolve().parents[1]


def test_sparse_probe_linear_reconstruction_preserves_linear_hidden():
    valid = torch.ones(1, 8, dtype=torch.bool)
    anchors = C3CoarseProbeActionnessSource._sparse_probe_positions(valid, 3)
    assert anchors.tolist() == [0, 3, 6, 7]
    sparse = torch.stack((anchors.float(), 2.0 * anchors.float()), dim=-1)[None]
    dense = C3CoarseProbeActionnessSource._reconstruct_sparse_sequence(
        sparse,
        anchor_positions=anchors,
        anchor_valid=valid[:, anchors],
        dense_valid=valid,
        mode="hidden_linear",
    )
    expected = torch.stack(
        (torch.arange(8).float(), 2.0 * torch.arange(8).float()), dim=-1
    )[None]
    assert torch.allclose(dense, expected)


def test_sparse_probe_forward_only_computes_anchor_frames_and_returns_dense_evidence():
    class FakeProbe:
        def __init__(self):
            self.observed_length = None

        def __call__(self, frames, valid, return_hidden=False):
            self.observed_length = int(frames.shape[1])
            logits = frames.mean(dim=(2, 3, 4))
            hidden = torch.stack((logits, 2.0 * logits), dim=-1)
            return {
                "logits": logits,
                "hidden": hidden,
                "hidden_kind": "official_asformer_encoder_hidden",
            }

    source = C3CoarseProbeActionnessSource.__new__(C3CoarseProbeActionnessSource)
    nn.Module.__init__(source)
    source.temporal_probe_stride = 3
    source.temporal_interpolation_mode = "hidden_linear"
    source.return_hidden_features = True
    source.require_hidden_features = True
    source.policy_hidden_gradient_scope = "none"
    source.frozen = False
    source.probe_model = "fake"
    source.probe = FakeProbe()
    source.calibration_bias = 0.0
    source.calibration_temperature = 1.0
    source.source_name = "fake_sparse_probe"
    source._prepare_probe_inputs = lambda value: value.permute(0, 2, 1, 3, 4)
    source._provenance = lambda: {
        "thumos_trained": True,
        "uses_labels": True,
        "uses_teacher": False,
        "uses_gt": False,
        "uses_prediction_cache": False,
    }
    source._estimate_probe_profile = lambda inputs, logits, latency: {
        "estimated_macs": 10,
        "estimated_flops": 20,
        "input_shape": list(inputs.shape),
        "output_shape": list(logits.shape),
        "latency_ms": {"coarse_probe_ms": latency},
    }
    frames = torch.arange(8).float()[None, None, :, None, None].expand(1, 3, 8, 2, 2)
    output = source(frames, valid_mask=torch.ones(1, 8, dtype=torch.bool))
    assert source.probe.observed_length == 4
    assert output["logits"].shape == (1, 8)
    assert output["coarse_hidden_features"].shape == (1, 8, 2)
    assert torch.allclose(output["logits"], torch.arange(8).float()[None])
    assert output["compute_profile"]["sparse_anchor_count"] == 4
    assert output["compute_profile"]["dense_output_length"] == 8


def test_sparse_probe_configs_and_submitter_cover_four_source_intervals():
    p0 = (ROOT / "configs/adatad/thumos/duca_sparse_probe_hidden_linear_frontend_pretrain_fixed384.py").read_text()
    tad = (ROOT / "configs/adatad/thumos/duca_sparse_probe_hidden_linear_g0_fixed384_official60.py").read_text()
    runner = (ROOT / "scripts/run_duca_independent_official60_gpu1.sh").read_text()
    submitter = (ROOT / "scripts/submit_duca_sparse_probe_tad_suite.sh").read_text()
    for stride in (1, 2, 3, 4):
        assert f"sparse_probe_hidden_linear_d{stride}" in runner
    for text in (p0, tad):
        assert 'temporal_interpolation_mode="hidden_linear"' in text
        assert "selector_receives_anchor_mask=False" in text
        assert "selector_receives_anchor_distance=False" in text
    assert "#SBATCH --gpus=4" in submitter
    assert "for stride in 1 2 3 4" in submitter
    assert "--expected-checkpoint-epoch 59" in runner


def test_sparse_probe_aggregate_requires_all_four_terminal_completions(tmp_path):
    for stride in (1, 2, 3, 4):
        arm = tmp_path / f"d{stride}"
        arm.mkdir()
        (arm / "completion.json").write_text(
            json.dumps(
                {
                    "ok": True,
                    "variant": f"sparse_probe_hidden_linear_d{stride}",
                    "official_validation_comparable": True,
                    "metrics": {"average_mAP": 60.0 + stride},
                }
            ),
            encoding="utf-8",
        )
    result = aggregate(tmp_path, tmp_path / "summary.json")
    assert result["ok"]
    assert [row["interval_source_frames"] for row in result["rows"]] == [4, 8, 12, 16]
