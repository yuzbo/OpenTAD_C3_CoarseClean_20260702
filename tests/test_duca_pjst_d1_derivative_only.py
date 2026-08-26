"""Focused regression tests for PJST-D1 (derivative-only physical tubelet).

These tests are fully runnable on Linux/N16R4 (OpenTAD environment).  On local
Windows the known Torch ``c10.dll`` load failure is the only reason this module
skips; every other failure is a real defect.
"""

import hashlib
import inspect
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

if sys.platform == "win32":
    # The local Windows user-site Torch can crash natively while loading c10.dll
    # (WinError 1114), which is not cleanly catchable in-process.  Probe import in
    # a subprocess and skip only for that known environment issue; on Linux/N16R4
    # the full test suite runs normally.
    probe = subprocess.run([sys.executable, "-c", "import torch"], capture_output=True)
    if probe.returncode != 0:
        pytest.skip("local Windows Torch c10.dll load failure (known); full run on Linux/N16R4", allow_module_level=True)

import torch
import torch.nn as nn

from opentad.models.backbones import backbone_wrapper as bw_module
from opentad.models.backbones.backbone_wrapper import BackboneWrapper
from opentad.models.detectors.single_stage import SingleStageDetector
from opentad.models.duca.structured_selection import exact_uniform_positions
from opentad.models.utils import temporal_grid as tg

ROOT = Path(__file__).resolve().parents[1]


def _uniform_positions(length, count):
    return exact_uniform_positions(length, count)


def _irregular_positions(length, count, *, seed=0):
    """Strictly increasing in-range positions that differ from exact uniform."""
    base = _uniform_positions(length, count)
    # nudge a few interior positions while preserving strict monotonicity.
    out = base.clone()
    for idx in range(2, min(count, 24), 7):
        if out[idx] + 1 < out[idx + 1] and out[idx] + 1 < length:
            out[idx] = out[idx] + 1
    if torch.equal(out, base):
        out[-2] = base[-2] + 1
    assert bool((out[1:] > out[:-1]).all()) and bool((out < length).all())
    return out


def _build_metadata(positions_rows, dense_lengths):
    """Build [B,384] positions / [B] lengths / [B,384] prefix mask for a batch."""
    rows = []
    for row in positions_rows:
        full = torch.full((384,), -1, dtype=torch.int64)
        full[: len(row)] = torch.as_tensor(row, dtype=torch.int64)
        rows.append(full)
    positions = torch.stack(rows)
    lengths = torch.as_tensor(dense_lengths, dtype=torch.int64)
    mask = positions >= 0
    return positions, lengths, mask


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------

def test_global_to_packed_layout_and_order():
    pos, lengths, mask = _build_metadata([_uniform_positions(768, 384), _irregular_positions(768, 384)], [768, 768])
    meta = tg.pjst_pair_metadata(pos, lengths, mask)
    assert meta["pair_scale"].shape == (2, 24, 8)
    assert meta["pair_valid"].shape == (2, 24, 8)
    assert meta["actual_delta"].shape == (2, 24, 8)
    # global pair order: pair (c, r) uses global indices (16c+2r, 16c+2r+1)
    p0 = pos.reshape(2, 24, 16)[:, :, 0::2]
    p1 = pos.reshape(2, 24, 16)[:, :, 1::2]
    assert torch.equal(meta["actual_delta"], p1 - p0)

    packed = tg.pack_pjst_pair_metadata(meta, clips=24)
    assert packed["pair_scale"].shape == (48, 8)
    assert packed["pair_valid"].shape == (48, 8)
    assert packed["exact_uniform_identity"].shape == (48,)
    # batch-major: sample 0's 24 clips come first, then sample 1's.
    assert torch.equal(packed["pair_scale"][:24], meta["pair_scale"][0])
    assert torch.equal(packed["pair_scale"][24:], meta["pair_scale"][1])


def test_exact_uniform_identity_flag():
    uniform = _uniform_positions(768, 384)
    irregular = _irregular_positions(768, 384)
    pos, lengths, mask = _build_metadata([uniform, irregular], [768, 768])
    meta = tg.pjst_pair_metadata(pos, lengths, mask)
    assert bool(meta["exact_uniform_identity"][0])
    assert not bool(meta["exact_uniform_identity"][1])


def test_uniform_mixed_batch_byte_identity():
    uniform = _uniform_positions(768, 384)
    irregular = _irregular_positions(768, 384)
    pos, lengths, mask = _build_metadata([uniform, irregular], [768, 768])
    meta = tg.pjst_pair_metadata(pos, lengths, mask)
    packed = tg.pack_pjst_pair_metadata(meta, clips=24)

    x = torch.randn(48, 3, 16, 4, 4)
    y = tg.apply_pjst_derivative_only(
        x, packed["pair_scale"], packed["pair_valid"], packed["exact_uniform_identity"]
    )
    # uniform sample's 24 clips stay byte-identical.
    assert torch.equal(y[:24], x[:24])
    # irregular sample's 24 clips may change but keep the same shape/dtype.
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_invalid_partial_pair_byte_identity():
    # short valid prefix of length 5 -> only pairs 0 and 1 (frames 0..3) valid.
    pos, lengths, mask = _build_metadata([_irregular_positions(768, 384)], [768])
    pos[0, 5:] = -1
    mask[0, 5:] = False
    meta = tg.pjst_pair_metadata(pos, lengths, mask)
    packed = tg.pack_pjst_pair_metadata(meta, clips=24)
    x = torch.randn(24, 3, 16, 4, 4)
    y = tg.apply_pjst_derivative_only(
        x, packed["pair_scale"], packed["pair_valid"], packed["exact_uniform_identity"]
    )
    # pair_valid only True for pair 0 (frames 0,1) and pair 1 (frames 2,3).
    assert bool(packed["pair_valid"][0, 0]) and bool(packed["pair_valid"][0, 1])
    assert not bool(packed["pair_valid"][0, 2])
    # frames 4..15 (invalid/partial pairs) stay byte-identical.
    assert torch.equal(y[0, :, 4:], x[0, :, 4:])


def test_explicit_formula():
    x = torch.randn(1, 3, 16, 4, 4)
    pair_scale = torch.full((1, 8), 2.0)
    pair_valid = torch.ones((1, 8), dtype=torch.bool)
    exact_uniform = torch.zeros((1,), dtype=torch.bool)
    y = tg.apply_pjst_derivative_only(x, pair_scale, pair_valid, exact_uniform)
    z = x.float().reshape(1, 3, 8, 2, 4, 4)
    x_minus, x_plus = z[:, :, :, 0], z[:, :, :, 1]
    m = (x_minus + x_plus) * 0.5
    v = 2.0 * (x_plus - x_minus) * 0.5
    expected = torch.stack((m - v, m + v), dim=3)
    assert torch.allclose(y.reshape(1, 3, 8, 2, 4, 4), expected, rtol=1e-6, atol=1e-6)


def test_constant_pair_invariance():
    x = torch.randn(1, 3, 16, 4, 4)
    x[:, :, 1] = x[:, :, 0]  # constant pair 0 (frames 0 and 1)
    pair_scale = torch.full((1, 8), 5.0)  # arbitrary gap
    pair_valid = torch.ones((1, 8), dtype=torch.bool)
    exact_uniform = torch.zeros((1,), dtype=torch.bool)
    y = tg.apply_pjst_derivative_only(x, pair_scale, pair_valid, exact_uniform)
    # v = 0 for the constant pair -> frames 0..1 unchanged at any gap.
    assert torch.equal(y[:, :, :2], x[:, :, :2])
    # non-constant pairs still receive the derivative transport.
    assert not torch.equal(y[:, :, 2:], x[:, :, 2:])


def test_gap_scaling_halves_derivative():
    # s = delta_can / delta_act; doubling the actual gap halves s, which exactly
    # halves the odd/derivative contribution v = s * (x+ - x-) / 2.
    x = torch.randn(1, 3, 16, 4, 4)
    pair_valid = torch.ones((1, 8), dtype=torch.bool)
    exact_uniform = torch.zeros((1,), dtype=torch.bool)
    y_full = tg.apply_pjst_derivative_only(x, torch.ones(1, 8), pair_valid, exact_uniform)
    y_half = tg.apply_pjst_derivative_only(x, torch.full((1, 8), 0.5), pair_valid, exact_uniform)

    def view(t):
        return t.reshape(1, 3, 8, 2, 4, 4)

    v_full = (view(y_full)[:, :, :, 1] - view(y_full)[:, :, :, 0]) / 2
    v_half = (view(y_half)[:, :, :, 1] - view(y_half)[:, :, :, 0]) / 2
    assert torch.allclose(v_half, 0.5 * v_full, rtol=1e-6, atol=1e-6)


def test_dtype_restoration():
    x = torch.randn(1, 3, 16, 4, 4, dtype=torch.float16)
    pair_scale = torch.full((1, 8), 2.0)
    pair_valid = torch.ones((1, 8), dtype=torch.bool)
    exact_uniform = torch.zeros((1,), dtype=torch.bool)
    y = tg.apply_pjst_derivative_only(x, pair_scale, pair_valid, exact_uniform)
    assert y.dtype == torch.float16
    # written frames are float16 (restored dtype), not float32.
    assert y.dtype == x.dtype


def test_finite_nonzero_input_gradient():
    x = torch.randn(1, 3, 16, 4, 4, requires_grad=True)
    pair_scale = torch.full((1, 8), 2.0)
    pair_valid = torch.ones((1, 8), dtype=torch.bool)
    exact_uniform = torch.zeros((1,), dtype=torch.bool)
    y = tg.apply_pjst_derivative_only(x, pair_scale, pair_valid, exact_uniform)
    y.square().sum().backward()
    assert x.grad is not None
    assert bool(torch.isfinite(x.grad).all())
    assert bool((x.grad != 0).any())


def test_uniform_row_bypasses_transform():
    x = torch.randn(24, 3, 16, 4, 4)
    pair_scale = torch.ones((24, 8))
    pair_valid = torch.ones((24, 8), dtype=torch.bool)
    exact_uniform = torch.ones((24,), dtype=torch.bool)
    y = tg.apply_pjst_derivative_only(x, pair_scale, pair_valid, exact_uniform)
    assert torch.equal(y, x)


# ---------------------------------------------------------------------------
# Backbone / detector reachability
# ---------------------------------------------------------------------------

class _RecordingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, masks=None, metas=None, **kwargs):
        self.calls.append({"masks": masks, "metas": metas, **kwargs})
        return x


class _MockDataPreprocessor:
    def preprocess(self, frames_list, data_samples=None, training=False):
        frames = torch.stack(frames_list)
        b = frames.shape[0]
        return frames.reshape(b * 24, 1, 3, 16, *frames.shape[-2:]), None


class _MockModel:
    """Stands in for the mmaction Recognizer3D held by BackboneWrapper."""

    def __init__(self):
        self.backbone = _RecordingBackbone()
        self.data_preprocessor = _MockDataPreprocessor()


def _make_wrapper(pjst):
    wrapper = BackboneWrapper.__new__(BackboneWrapper)
    wrapper.pjst_derivative_only = bool(pjst)
    wrapper.model = _MockModel()
    wrapper.norm_eval = False
    wrapper.pre_processing_pipeline = None
    wrapper.post_processing_pipeline = None
    wrapper.freeze_backbone = False
    wrapper.use_temporal_checkpointing = False
    return wrapper


def test_off_does_not_pass_pjst_kwargs():
    wrapper = _make_wrapper(False)
    frames = torch.randn(4, 3, 16, 4, 4)
    wrapper._backbone_forward(frames, None, None, None, None, False, None, None, None)
    kwargs = wrapper.model.backbone.calls[-1]
    assert "pjst_pair_scale" not in kwargs
    assert "pjst_pair_valid" not in kwargs
    assert "pjst_exact_uniform_identity" not in kwargs


def test_on_passes_pjst_kwargs():
    wrapper = _make_wrapper(True)
    frames = torch.randn(4, 3, 16, 4, 4)
    scale = torch.ones(4, 8)
    valid = torch.ones(4, 8, dtype=torch.bool)
    uniform = torch.ones(4, dtype=torch.bool)
    wrapper._backbone_forward(frames, None, None, None, None, False, scale, valid, uniform)
    kwargs = wrapper.model.backbone.calls[-1]
    assert kwargs["pjst_pair_scale"] is scale
    assert kwargs["pjst_pair_valid"] is valid
    assert kwargs["pjst_exact_uniform_identity"] is uniform


def test_backbone_wrapper_on_extracts_metas():
    wrapper = _make_wrapper(True)
    frames = torch.randn(2, 1, 3, 384, 4, 4)
    uniform = _uniform_positions(768, 384)
    metas = [
        {"irregular_selected_positions": uniform.tolist(), "irregular_selected_count": 384,
         "irregular_selected_valid_len": 384, "irregular_dense_valid_len": 768},
        {"irregular_selected_positions": _irregular_positions(768, 384).tolist(), "irregular_selected_count": 384,
         "irregular_selected_valid_len": 384, "irregular_dense_valid_len": 768},
    ]
    wrapper.forward(frames, metas=metas)
    kwargs = wrapper.model.backbone.calls[-1]
    assert kwargs["pjst_pair_scale"].shape == (48, 8)
    assert kwargs["pjst_pair_valid"].shape == (48, 8)
    assert kwargs["pjst_exact_uniform_identity"].shape == (48,)


def test_single_stage_metas_reach_backbone():
    det = SingleStageDetector.__new__(SingleStageDetector)
    nn.Module.__init__(det)
    det.backbone = _RecordingBackbone()
    frames = torch.randn(1, 3, 4, 4)
    masks = torch.ones(1, 4)
    metas = [{"irregular_selected_positions": [0, 1, 2, 3]}]
    det._call_backbone(frames, masks, metas)
    assert det.backbone.calls[-1]["metas"] is metas


def test_single_stage_forward_train_passes_metas():
    # Build a minimal detector whose backbone accepts metas and records it.
    det = SingleStageDetector.__new__(SingleStageDetector)
    nn.Module.__init__(det)
    det.backbone = _RecordingBackbone()
    det.frame_selector = None
    det.projection = None
    det.neck = None
    det.rpn_head = None
    # with_projection/neck/rpn_head are False via hasattr checks on missing attrs.
    frames = torch.randn(1, 1, 3, 4, 4, 4)
    masks = torch.ones(1, 4)
    metas = [{"video_name": "v1", "irregular_selected_positions": [0, 1, 2, 3]}]
    gt_segments = torch.zeros(1, 0, 2)
    gt_labels = torch.zeros(1, 0, dtype=torch.long)
    losses = det.forward_train(frames, masks, metas, gt_segments, gt_labels)
    assert det.backbone.calls[-1]["metas"] is metas


def test_chunk_dim_2_on_rejection_off_unchanged():
    on_wrapper = _make_wrapper(True)
    with pytest.raises(ValueError):
        on_wrapper.temporal_checkpointing(
            torch.randn(4, 3, 16, 4, 4), 2, 2,
            None, None, None, None, False,
            torch.ones(4, 8), torch.ones(4, 8, dtype=torch.bool), torch.ones(4, dtype=torch.bool),
        )
    off_wrapper = _make_wrapper(False)
    off_wrapper.temporal_checkpointing(torch.randn(4, 3, 16, 4, 4), 2, 2)


def test_chunk_dim_0_metadata_slicing(monkeypatch):
    calls = []

    def passthrough(fn, *args, **kwargs):
        calls.append(args)
        return fn(*args)

    monkeypatch.setattr(bw_module.cp, "checkpoint", passthrough)
    wrapper = _make_wrapper(True)
    wrapper.pjst_derivative_only = True
    scale = torch.arange(48 * 8, dtype=torch.float32).reshape(48, 8)
    valid = torch.ones(48, 8, dtype=torch.bool)
    uniform = torch.ones(48, dtype=torch.bool)
    wrapper.temporal_checkpointing(
        torch.randn(48, 3, 16, 4, 4), 2, 0,
        None, None, None, None, False, scale, valid, uniform,
    )
    # two chunks -> two checkpoint calls; each mini-metadata slice is contiguous.
    assert len(calls) == 2
    assert len(wrapper.model.backbone.calls) == 2
    # args are (frames, actual, canonical, dense_len, tubelet_valid, pjst_scale,
    # pjst_valid, pjst_uniform); chunk 0 -> scale[:24], chunk 1 -> scale[24:].
    assert torch.equal(calls[0][5], scale[:24])
    assert torch.equal(calls[1][5], scale[24:])


# ---------------------------------------------------------------------------
# Static / config / launcher / remap-order checks
# ---------------------------------------------------------------------------

def test_remap_occurs_exactly_once_before_filtering():
    # Inspect only the post_processing method source, not later helper definitions
    # (which would also match the remap name and break the exactly-once count).
    pp = inspect.getsource(SingleStageDetector.post_processing)
    remap_line = pp.index("_remap_selector_segments_for_post_processing(")
    threshold_line = pp.index("keep_idxs1 = pred_prob > pre_nms_thresh")
    nms_line = pp.index("batched_nms(")
    assert remap_line < threshold_line < nms_line
    # exactly one remap call inside post_processing
    assert pp.count("_remap_selector_segments_for_post_processing(") == 1


def test_single_patch_embed_call():
    src = (ROOT / "opentad/models/backbones/vit_adapter.py").read_text(encoding="utf-8")
    fwd = src.split("def forward", 1)[1].split("def _freeze_layers", 1)[0]
    assert fwd.count("self.patch_embed(x)[0]") == 1
    assert fwd.index("apply_pjst_derivative_only") < fwd.index("self.patch_embed(x)[0]")


def _write_fixture_checkpoint(dir_path, content=b"duca-stage1-epoch29-fixture\n"):
    p = Path(dir_path) / "duca_stage1_epoch29.ckpt"
    p.write_bytes(content)
    return p, hashlib.sha256(content).hexdigest()


def _run_validator(*extra_args):
    return subprocess.run(
        [sys.executable, str(ROOT / "tools/bata/validate_duca_pjst_d1_derivative_only.py"), *extra_args],
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )


def test_config_sole_distinction_and_freeze_via_validator(tmp_path):
    ckpt, digest = _write_fixture_checkpoint(tmp_path)
    proc = _run_validator("--stage1", str(ckpt), "--sha256", digest, "--epoch", "29")
    assert proc.returncode == 0, proc.stderr
    assert "PASS PJST-D1 matched configs" in proc.stdout


def test_validator_requires_explicit_checkpoint_and_sha():
    proc = _run_validator()
    assert proc.returncode != 0


def test_validator_missing_checkpoint_file_fails(tmp_path):
    missing = tmp_path / "does_not_exist.ckpt"
    proc = _run_validator("--stage1", str(missing), "--sha256", "0" * 64)
    assert proc.returncode != 0


def test_validator_non_regular_checkpoint_fails(tmp_path):
    # A directory is not a regular file and must not pass admission.
    proc = _run_validator("--stage1", str(tmp_path), "--sha256", "0" * 64)
    assert proc.returncode != 0


def test_validator_malformed_digest_fails(tmp_path):
    ckpt, _ = _write_fixture_checkpoint(tmp_path)
    proc = _run_validator("--stage1", str(ckpt), "--sha256", "not-a-sha")
    assert proc.returncode != 0


def test_validator_wrong_digest_fails(tmp_path):
    ckpt, _ = _write_fixture_checkpoint(tmp_path)
    proc = _run_validator("--stage1", str(ckpt), "--sha256", "a" * 64)
    assert proc.returncode != 0
    assert "mismatch" in (proc.stderr + proc.stdout)


def test_validator_wrong_epoch_fails(tmp_path):
    ckpt, digest = _write_fixture_checkpoint(tmp_path)
    proc = _run_validator("--stage1", str(ckpt), "--sha256", digest, "--epoch", "30")
    assert proc.returncode != 0


def test_validator_correct_fixture_passes(tmp_path):
    ckpt, digest = _write_fixture_checkpoint(tmp_path)
    proc = _run_validator("--stage1", str(ckpt), "--sha256", digest, "--epoch", "29")
    assert proc.returncode == 0, proc.stderr
    assert "PASS PJST-D1 matched configs" in proc.stdout


def test_validator_unreadable_checkpoint_fails(tmp_path):
    if sys.platform == "win32":
        pytest.skip("POSIX file permissions not applicable on Windows")
    ckpt, digest = _write_fixture_checkpoint(tmp_path)
    os.chmod(ckpt, 0)
    try:
        if os.access(ckpt, os.R_OK):
            pytest.skip("running as root; unreadable file check not applicable")
        proc = _run_validator("--stage1", str(ckpt), "--sha256", digest)
        assert proc.returncode != 0
    finally:
        os.chmod(ckpt, 0o644)


def test_launcher_precheck_fail_closed():
    src = (ROOT / "scripts/run_duca_pjst_d1_matched_cycle3_n16r4.sbatch").read_text(encoding="utf-8")
    validator_call = "tools/bata/validate_duca_pjst_d1_derivative_only.py"
    call_idx = src.index(validator_call)
    precheck = src[:call_idx]
    # Fail-closed guards must precede the validator invocation.
    assert '[[ "$STAGE1_EPOCH" == 29 ]]' in precheck
    assert '[[ -f "$STAGE1_CHECKPOINT" ]]' in precheck
    assert '[[ -r "$STAGE1_CHECKPOINT" ]]' in precheck
    assert '=~ ^[0-9a-fA-F]{64}$' in precheck
    # Launcher passes explicit path/digest/epoch to the validator.
    assert '--stage1 "$STAGE1_CHECKPOINT"' in src
    assert '--sha256 "$STAGE1_SHA"' in src
    assert '--epoch 29' in src


def test_launcher_syntax():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available on this host")
    launcher = ROOT / "scripts/run_duca_pjst_d1_matched_cycle3_n16r4.sbatch"
    proc = subprocess.run([bash, "-n", str(launcher)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_launcher_binds_dataset_env_and_drops_stale_data_overrides():
    src = (ROOT / "scripts/run_duca_pjst_d1_matched_cycle3_n16r4.sbatch").read_text(encoding="utf-8")
    exec_call = 'exec "$PYTHON" tools/train.py'
    exec_idx = src.index(exec_call)
    before_train = src[:exec_idx]
    # The four exact dataset variables are exported through the config's real
    # THUMOS14_* interface before tools/train.py loads cfg.dataset.*.
    assert 'export THUMOS14_ANNOTATION_PATH="$ANNOTATION_PATH"' in before_train
    assert 'export THUMOS14_CLASS_MAP="$CATEGORY_PATH"' in before_train
    assert 'export THUMOS14_TRAIN_DATA_PATH="$VIDEO_ROOT"' in before_train
    assert 'export THUMOS14_TEST_DATA_PATH="$VIDEO_ROOT"' in before_train
    # No stale data.train/data.val/data.test cfg-options remain anywhere; these
    # silently built an unused cfg.data tree that never reached cfg.dataset.*.
    for stale in (
        "data.train.ann_file",
        "data.val.ann_file",
        "data.test.ann_file",
        "data.train.category_file",
        "data.val.category_file",
        "data.test.category_file",
    ):
        assert stale not in src
    # The real custom pretrain binding is retained (not moved to data_preprocessor).
    assert 'model.backbone.custom.pretrain="$ADATAD_PRETRAIN"' in src
