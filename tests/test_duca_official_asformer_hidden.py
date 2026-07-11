from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

if os.name == "nt":
    pytest.skip("local Windows torch/c10.dll import is unstable; Linux remote runs this suite", allow_module_level=True)

import torch

from opentad.models.duca.transition_only import ASFORMER_ENCODER_HIDDEN_KIND
from tools.bata import train_lowres_action_probe as probe_module


_FAKE_ASFORMER = r'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden, classes):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, hidden, 1)
        self.dropout = nn.Dropout(0.25)
        self.conv_out = nn.Conv1d(hidden, classes, 1)
        self.calls = 0
    def forward(self, x, mask):
        self.calls += 1
        feature = self.dropout(torch.tanh(self.conv_1x1(x)))
        return self.conv_out(feature) * mask[:, :1], feature

class Decoder(nn.Module):
    def __init__(self, classes, hidden):
        super().__init__()
        self.proj = nn.Conv1d(classes, hidden, 1)
        self.dropout = nn.Dropout(0.25)
        self.out = nn.Conv1d(hidden, classes, 1)
    def forward(self, x, fencoder, mask):
        feature = self.dropout(torch.tanh(self.proj(x)) + fencoder)
        return self.out(feature) * mask[:, :1], feature

class MyTransformer(nn.Module):
    def __init__(self, num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate):
        super().__init__()
        self.encoder = Encoder(input_dim, num_f_maps, num_classes)
        self.decoders = nn.ModuleList([Decoder(num_classes, num_f_maps) for _ in range(num_decoders)])
    def forward(self, x, mask):
        out, feature = self.encoder(x, mask)
        outputs = out.unsqueeze(0)
        for decoder in self.decoders:
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, :1], feature * mask[:, :1], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs

class Trainer:
    pass
'''


def _make_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_path = tmp_path / "ASFormer" / "model.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(_FAKE_ASFORMER, encoding="utf-8")
    monkeypatch.setenv("C3_OFFICIAL_ACTION_SEG_REPOS", str(tmp_path))
    probe_module._OFFICIAL_ACTION_SEG_MODULE_CACHE.clear()
    probe = probe_module.C3OfficialActionSegmentationProbe(
        backend="official_asformer",
        spatial_size=8,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        hidden_output_kind=ASFORMER_ENCODER_HIDDEN_KIND,
    )
    return probe, source_path


def test_official_asformer_hidden_capture_preserves_logits_and_rng(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe, source_path = _make_probe(tmp_path, monkeypatch)
    frames = torch.randn(2, 6, 3, 8, 8)
    valid = torch.tensor([[True] * 6, [True, True, True, True, False, False]])
    probe.train()

    torch.manual_seed(17)
    calls_before = probe.official_temporal.encoder.calls
    logits_only = probe(frames, valid, return_hidden=False)
    rng_without_hidden = torch.random.get_rng_state().clone()
    calls_without_hidden = probe.official_temporal.encoder.calls - calls_before

    torch.manual_seed(17)
    calls_before = probe.official_temporal.encoder.calls
    with_hidden = probe(frames, valid, return_hidden=True)
    rng_with_hidden = torch.random.get_rng_state().clone()
    calls_with_hidden = probe.official_temporal.encoder.calls - calls_before

    assert torch.equal(logits_only, with_hidden["logits"])
    assert torch.equal(rng_without_hidden, rng_with_hidden)
    assert calls_without_hidden == frames.shape[0]
    assert calls_with_hidden == frames.shape[0]
    assert with_hidden["hidden_kind"] == ASFORMER_ENCODER_HIDDEN_KIND
    assert with_hidden["hidden"].shape == (2, 6, 16)
    assert torch.equal(with_hidden["hidden"][1, 4:], torch.zeros_like(with_hidden["hidden"][1, 4:]))
    assert with_hidden["official_source_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert with_hidden["official_source_normalized_lf_sha256"] == hashlib.sha256(
        source_path.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
