# PhysTime-AdaTAD 1.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a raw-video, pre-backbone sparse PhysTime detector on the official OpenTAD AdaTAD VideoMAE-S stack, with matched selected-axis and physical-grid head baselines.

**Architecture:** A deterministic GT-independent sampler selects K raw frames from a logical 768-position window before `DecordDecode`. The official VideoMAE-S adapter processes only those K frames. Three matched configurations then use selected-rank ActionFormer, physical-grid ActionFormer, or `PhysTimeMeasureProjection + PhysTimeHead`; the PhysTime branch keeps GT and predictions on the original physical timeline in seconds.

**Tech Stack:** Python, PyTorch, MMEngine, MMAction2, OpenTAD, VideoMAE-S, Decord, pytest, Slurm.

---

## File Map

- Create `opentad/datasets/transforms/phystime_raw.py`: raw selected-frame geometry and seconds conversion only.
- Modify `opentad/datasets/transforms/__init__.py`: register the raw geometry transform.
- Modify `opentad/datasets/transforms/formatting.py`: preserve raw PhysTime metadata through `Collect`.
- Create `configs/adatad/thumos/phystime_adatad_sparse_k384.py`: main raw-video PhysTime configuration.
- Create `configs/adatad/thumos/selected_axis_adatad_sparse_k384.py`: original selected-rank head baseline.
- Create `configs/adatad/thumos/physical_grid_adatad_sparse_k384.py`: existing physical-grid head baseline.
- Create `tests/test_phystime_raw_frame_geometry.py`: geometry and no-leak unit contract.
- Create `tests/test_phystime_adatad_configs.py`: matched sampling and model registry contract.
- Create `tests/test_phystime_adatad_one_step.py`: end-to-end synthetic raw-video gradient contract.
- Create `tools/bata/run_phystime_adatad_real_gate.py`: real THUMOS CUDA gate.
- Create `tools/bata/validate_phystime_adatad_track.py`: fail-closed deployment validator.
- Create `scripts/run_phystime_adatad_gate_gpu1.sh`: focused gate launcher.
- Create `scripts/run_phystime_adatad_full_train_gpu1.sh`: formal training launcher.
- Create `scripts/submit_phystime_adatad_head_comparison.sh`: gate-dependent three-head queue.
- Modify `docs/evaluation/results.md`: record only verified raw-video gate and experiment states.

### Task 1: Raw-Frame Physical Geometry

**Files:**
- Create: `tests/test_phystime_raw_frame_geometry.py`
- Create: `opentad/datasets/transforms/phystime_raw.py`
- Modify: `opentad/datasets/transforms/__init__.py`
- Modify: `opentad/datasets/transforms/formatting.py`

- [ ] **Step 1: Write the failing raw geometry tests**

```python
import numpy as np
import pytest
import torch

from opentad.datasets.transforms.phystime_raw import BuildPhysTimeRawFrameGeometry


def make_sample(with_gt=True):
    sample = {
        "frame_inds": np.array([100, 108, 132, 140], dtype=np.int64),
        "selected_dense_indices": np.array([0, 2, 8, 10], dtype=np.float32),
        "masks": torch.ones(4, dtype=torch.bool),
        "snippet_stride": 4,
        "avg_fps": 20.0,
        "duration": 20.0,
        "irregular_native_axis": True,
        "remap_gt_to_selected_axis": False,
        "gt_remapped_to_selected_axis": False,
    }
    if with_gt:
        sample["gt_segments"] = np.array([[1.0, 6.0]], dtype=np.float32)
    return sample


def test_raw_geometry_uses_original_video_time_and_dense_cell_support():
    out = BuildPhysTimeRawFrameGeometry(convert_gt_to_seconds=True)(make_sample())
    assert out["phystime_timestamps_sec"] == pytest.approx([5.0, 5.4, 6.6, 7.0])
    assert out["phystime_support_intervals_sec"][0] == pytest.approx([4.9, 5.1])
    assert torch.allclose(out["gt_segments"], torch.tensor([[5.2, 6.2]]))
    assert out["prediction_time_unit"] == "seconds"
    assert out["gt_time_unit"] == "seconds"
    assert out["phystime_support_provenance"] == "original_raw_dense_cells"


def test_raw_geometry_forbids_selected_axis_ground_truth():
    sample = make_sample()
    sample["remap_gt_to_selected_axis"] = True
    with pytest.raises(ValueError, match="selected-axis"):
        BuildPhysTimeRawFrameGeometry(convert_gt_to_seconds=True)(sample)


def test_raw_geometry_test_mode_does_not_require_ground_truth():
    out = BuildPhysTimeRawFrameGeometry(convert_gt_to_seconds=False)(make_sample(with_gt=False))
    assert "gt_time_unit" not in out
    assert out["prediction_time_unit"] == "seconds"
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m pytest tests/test_phystime_raw_frame_geometry.py -q
```

Expected: collection fails because `opentad.datasets.transforms.phystime_raw` does not exist.

- [ ] **Step 3: Implement `BuildPhysTimeRawFrameGeometry`**

```python
import numpy as np
import torch

from ..builder import PIPELINES


@PIPELINES.register_module()
class BuildPhysTimeRawFrameGeometry:
    def __init__(self, convert_gt_to_seconds=True):
        self.convert_gt_to_seconds = bool(convert_gt_to_seconds)

    def __call__(self, results):
        if results.get("remap_gt_to_selected_axis") or results.get("gt_remapped_to_selected_axis"):
            raise ValueError("PhysTime raw geometry forbids selected-axis ground truth")
        frame_inds = np.asarray(results["frame_inds"], dtype=np.float64).reshape(-1)
        mask = torch.as_tensor(results["masks"], dtype=torch.bool).reshape(-1)
        valid_count = int(mask.sum().item())
        if valid_count <= 0 or not torch.equal(mask, torch.arange(mask.numel()) < valid_count):
            raise ValueError("PhysTime raw masks must contain a non-empty valid prefix")
        selected_frames = frame_inds[:valid_count]
        selected_dense = np.asarray(results["selected_dense_indices"], dtype=np.float64)[:valid_count]
        if selected_dense.size != valid_count or np.any(np.diff(selected_frames) < 0):
            raise ValueError("PhysTime raw indices must be aligned and sorted")
        fps = float(results.get("avg_fps", results.get("fps", 0.0)))
        stride = float(results["snippet_stride"])
        duration = float(results["duration"])
        if fps <= 0 or stride <= 0 or duration <= 0:
            raise ValueError("PhysTime raw geometry requires positive fps, stride, and duration")
        centers = selected_frames / fps
        half_width = 0.5 * stride / fps
        supports = np.stack(
            [np.maximum(centers - half_width, 0.0), np.minimum(centers + half_width, duration)],
            axis=-1,
        )
        dense_origin_frame = float(selected_frames[0] - selected_dense[0] * stride)
        if self.convert_gt_to_seconds:
            gt = torch.as_tensor(results["gt_segments"], dtype=torch.float32)
            results["gt_segments"] = (gt * stride + dense_origin_frame) / fps
            results["gt_time_unit"] = "seconds"
        results.update(
            phystime_timestamps_sec=centers.astype(np.float32).tolist(),
            phystime_support_intervals_sec=supports.astype(np.float32).tolist(),
            phystime_duration_sec=duration,
            phystime_domain_start_sec=max(dense_origin_frame / fps, 0.0),
            phystime_domain_end_sec=min((dense_origin_frame + results["irregular_dense_valid_len"] * stride) / fps, duration),
            phystime_support_provenance="original_raw_dense_cells",
            phystime_selected_raw_frame_indices=selected_frames.astype(np.int64).tolist(),
            phystime_sampling_uses_gt=False,
            irregular_native_axis=True,
            remap_gt_to_selected_axis=False,
            gt_remapped_to_selected_axis=False,
            prediction_time_unit="seconds",
        )
        return results
```

Register the class in `transforms/__init__.py` and add the new keys to the metadata allowlist in `formatting.py`.

- [ ] **Step 4: Run focused tests and verify GREEN**

```bash
python -m pytest tests/test_phystime_raw_frame_geometry.py tests/test_phystime_geometry.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add opentad/datasets/transforms/phystime_raw.py opentad/datasets/transforms/__init__.py opentad/datasets/transforms/formatting.py tests/test_phystime_raw_frame_geometry.py
git commit -m "Add raw-frame PhysTime geometry"
```

### Task 2: Matched Sparse AdaTAD Configurations

**Files:**
- Create: `tests/test_phystime_adatad_configs.py`
- Create: `configs/adatad/thumos/phystime_adatad_sparse_k384.py`
- Create: `configs/adatad/thumos/selected_axis_adatad_sparse_k384.py`
- Create: `configs/adatad/thumos/physical_grid_adatad_sparse_k384.py`

- [ ] **Step 1: Write failing configuration tests**

```python
from pathlib import Path

from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "selected": ROOT / "configs/adatad/thumos/selected_axis_adatad_sparse_k384.py",
    "physical": ROOT / "configs/adatad/thumos/physical_grid_adatad_sparse_k384.py",
    "phystime": ROOT / "configs/adatad/thumos/phystime_adatad_sparse_k384.py",
}


def load_frame_step(cfg, split):
    return next(step for step in cfg.dataset[split].pipeline if step["type"] == "LoadFrames")


def test_sparse_configs_share_raw_sampling_contract():
    cfgs = {name: Config.fromfile(path) for name, path in CONFIGS.items()}
    for split in ("train", "val", "test"):
        steps = {name: load_frame_step(cfg, split) for name, cfg in cfgs.items()}
        assert {step["method"] for step in steps.values()} == {"random_fixed_subsample"}
        assert {step["target_len"] for step in steps.values()} == {384}
        assert {step["source_len"] for step in steps.values() if split == "train"} == {768}
    assert cfgs["phystime"].model.type == "PhysTimeTAD"
    assert cfgs["phystime"].model.backbone.backbone.type == "VisionTransformerAdapter"
    assert cfgs["phystime"].model.projection.type == "PhysTimeMeasureProjection"
    assert cfgs["phystime"].model.rpn_head.type == "PhysTimeHead"
    assert cfgs["selected"].model.rpn_head.type == "ActionFormerHead"
    assert cfgs["physical"].model.rpn_head.physical_grid_actionformer.enabled is True
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
python -m pytest tests/test_phystime_adatad_configs.py -q
```

Expected: failure because the three configuration files do not exist.

- [ ] **Step 3: Create the selected-axis and physical-grid baselines**

Each configuration must inherit `e2e_thumos_videomae_s_768x1_160_adapter.py`, set `window_size=384`, `dense_window_size=768`, and use this `LoadFrames` contract:

```python
dict(
    type="LoadFrames",
    num_clips=1,
    method="random_fixed_subsample",
    method_base="random_trunc",
    keep_ratio=0.5,
    target_len=384,
    source_len=768,
    trunc_thresh=0.75,
    crop_ratio=[0.9, 1.0],
    scale_factor=1,
    remap_gt_to_selected_axis=True,  # selected baseline only
)
```

Validation and test use `method_base="sliding_window"`, `target_len=384`, and `remap_gt_to_selected_axis` matching the head. The physical-grid configuration sets the flag to false and enables the existing strict `physical_grid_actionformer` block.

- [ ] **Step 4: Create the PhysTime-AdaTAD configuration**

Use the same raw sampling with `remap_gt_to_selected_axis=False`, append `BuildPhysTimeRawFrameGeometry` before image decoding, and define:

```python
model = dict(
    _delete_=True,
    type="PhysTimeTAD",
    discretization_loss_weight=0.0,
    backbone=dict(
        type="mmaction.Recognizer3D",
        backbone=dict(
            type="VisionTransformerAdapter",
            img_size=224,
            patch_size=16,
            embed_dims=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            num_frames=16,
            drop_path_rate=0.1,
            norm_cfg=dict(type="LN", eps=1e-6),
            return_feat_map=True,
            with_cp=True,
            total_frames=384,
            adapter_index=list(range(12)),
        ),
        data_preprocessor=dict(
            type="mmaction.ActionDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape="NCTHW",
        ),
        custom=dict(
            pretrain="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
            pre_processing_pipeline=[dict(type="Rearrange", keys=["frames"], ops="b n c (t1 t) h w -> (b t1) n c t h w", t1=24)],
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t1 t)", t1=24),
                dict(type="Interpolate", keys=["feats"], size=384),
            ],
            norm_eval=False,
            freeze_backbone=False,
        ),
    ),
    projection=dict(
        type="PhysTimeMeasureProjection",
        in_channels=384,
        out_channels=512,
        attention_channels=128,
        observation_measure="support_overlap",
        base_spacing_sec=0.5,
        num_levels=6,
        dropout=0.1,
    ),
    rpn_head=dict(
        type="PhysTimeHead",
        num_classes=20,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        regression_ranges_sec=[(0.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, 32.0), (32.0, 1.0e8)],
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample_radius=1.5,
        cls_prior_prob=0.01,
        endpoint_loss_weight=0.25,
        loss=dict(cls_loss=dict(type="FocalLoss"), reg_loss=dict(type="DIOULoss")),
    ),
)
```

Copy the official AdaTAD optimizer block so the VideoMAE trunk remains at learning rate zero and adapters use 2e-4.

- [ ] **Step 5: Run configuration and registry tests**

```bash
python -m pytest tests/test_phystime_adatad_configs.py tests/test_phystime_config_precheck.py -q
```

Expected: all tests pass and all three configurations build.

- [ ] **Step 6: Commit Task 2**

```bash
git add configs/adatad/thumos/phystime_adatad_sparse_k384.py configs/adatad/thumos/selected_axis_adatad_sparse_k384.py configs/adatad/thumos/physical_grid_adatad_sparse_k384.py tests/test_phystime_adatad_configs.py
git commit -m "Add matched sparse AdaTAD head configs"
```

### Task 3: Same-Observation Audit

**Files:**
- Modify: `tests/test_phystime_adatad_configs.py`
- Create: `tools/bata/validate_phystime_adatad_track.py`

- [ ] **Step 1: Write the failing checksum test**

```python
import hashlib
import numpy as np

from opentad.datasets.transforms.end_to_end import LoadFrames


def checksum(values):
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def test_all_heads_select_identical_frames_for_same_window():
    sample_results = {
        "video_name": "video_test_0000001",
        "total_frames": 4000,
        "snippet_stride": 4,
        "window_size": 768,
        "feature_start_idx": 20,
        "feature_end_idx": 787,
    }
    selected = []
    for remap in (True, False, False):
        loader = LoadFrames(
            method="random_fixed_subsample",
            method_base="sliding_window",
            target_len=384,
            keep_ratio=0.5,
            remap_gt_to_selected_axis=remap,
        )
        selected.append(loader(dict(sample_results))["frame_inds"])
    assert len({checksum(item) for item in selected}) == 1
```

- [ ] **Step 2: Run the checksum test and verify its fixture/validator is missing**

```bash
python -m pytest tests/test_phystime_adatad_configs.py::test_all_heads_select_identical_frames_for_same_window -q
```

Expected: failure because the formal configuration files and track validator are not implemented.

- [ ] **Step 3: Implement the fail-closed validator**

The validator loads all three configs and rejects any mismatch in dataset paths, `LoadFrames` parameters, pretrained checkpoint, backbone structure, optimizer schedule, workflow, NMS, or seed-facing sampling fields. It emits JSON containing each resolved config hash and the shared sampling-contract hash.

```python
def require_equal(name, values):
    normalized = [json.dumps(value, sort_keys=True, default=str) for value in values]
    if len(set(normalized)) != 1:
        raise RuntimeError(f"matched PhysTime-AdaTAD contract differs for {name}: {normalized}")
```

- [ ] **Step 4: Verify validator and tests**

```bash
python tools/bata/validate_phystime_adatad_track.py --output /tmp/phystime_adatad_contract.json
python -m pytest tests/test_phystime_adatad_configs.py -q
```

Expected: validator exits zero and records one shared sampling hash.

- [ ] **Step 5: Commit Task 3**

```bash
git add tools/bata/validate_phystime_adatad_track.py tests/test_phystime_adatad_configs.py
git commit -m "Enforce matched PhysTime-AdaTAD observations"
```

### Task 4: End-to-End Gradient and Seconds Round Trip

**Files:**
- Create: `tests/test_phystime_adatad_one_step.py`
- Modify: `opentad/models/detectors/phystime_tad.py` only if the failing test exposes an integration defect.

- [ ] **Step 1: Write the failing one-step test**

```python
import torch
import torch.nn as nn

from opentad.models.detectors.phystime_tad import PhysTimeTAD


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
    k = 16
    timestamps = [0.25 + 0.5 * idx for idx in range(k)]
    supports = [[value - 0.25, value + 0.25] for value in timestamps]
    return dict(
        inputs=torch.randn(1, 1, 3, k, 8, 8),
        masks=torch.ones(1, k, dtype=torch.bool),
        metas=[dict(
            video_name="synthetic",
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
        )],
        gt_segments=[torch.tensor([[2.0, 4.0]])],
        gt_labels=[torch.tensor([1])],
    )


def test_phystime_adatad_cost_reaches_adapters_projection_and_head():
    model = build_tiny_model().train()
    batch = make_batch()
    losses = model(return_loss=True, **batch)
    assert torch.isfinite(losses["cost"])
    losses["cost"].backward()
    parameters = dict(model.named_parameters())
    required = {
        "adapter": next(p for n, p in parameters.items() if "adapter" in n and p.requires_grad),
        "projection": model.projection.level_attentions[0].value_proj.weight,
        "classification": model.rpn_head.cls_head.weight,
        "regression": model.rpn_head.reg_head.weight,
        "endpoint": model.rpn_head.endpoint_head.weight,
    }
    for name, parameter in required.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
```

Add assertions that `forward_test` returns finite seconds and that
`round(proposal_sec * fps)` produces original-video frame numbers without a
selected-axis inverse map.

- [ ] **Step 2: Run the one-step test and verify RED**

```bash
python -m pytest tests/test_phystime_adatad_one_step.py -q
```

Expected: failure at the first missing raw-video integration contract.

- [ ] **Step 3: Make the minimal integration correction**

Keep `PhysTimeTAD._extract_observations()` as the single backbone entry. Do not
add a second detector wrapper. If metadata forwarding or optimizer accounting
fails, correct only that boundary and retain the existing projection/head APIs.

- [ ] **Step 4: Verify GREEN and existing detector tests**

```bash
python -m pytest tests/test_phystime_adatad_one_step.py tests/test_phystime_detector.py tests/test_phystime_head.py -q
```

Expected: all tests pass with nonzero adapter and head gradients.

- [ ] **Step 5: Commit Task 4**

```bash
git add tests/test_phystime_adatad_one_step.py opentad/models/detectors/phystime_tad.py
git commit -m "Prove PhysTime-AdaTAD gradient flow"
```

### Task 5: Real THUMOS CUDA Gate

**Files:**
- Create: `tools/bata/run_phystime_adatad_real_gate.py`
- Create: `scripts/run_phystime_adatad_gate_gpu1.sh`
- Create: `tests/test_phystime_adatad_gate_contract.py`

- [ ] **Step 1: Write failing gate contract tests**

```python
def test_gate_requires_raw_video_and_all_three_configs():
    from pathlib import Path

    text = Path("tools/bata/run_phystime_adatad_real_gate.py").read_text()
    assert "phystime_adatad_sparse_k384.py" in text
    assert "selected_axis_adatad_sparse_k384.py" in text
    assert "physical_grid_adatad_sparse_k384.py" in text
    assert "decoded_frame_count" in text
    assert "selected_index_checksum" in text
    assert "adapter_gradient_nonzero" in text
    assert "prediction_time_unit" in text
```

- [ ] **Step 2: Run the gate test and verify RED**

```bash
python -m pytest tests/test_phystime_adatad_gate_contract.py -q
```

Expected: failure because the real gate tool is absent.

- [ ] **Step 3: Implement the real gate**

The gate builds one real training sample per config, verifies the frame-index
checksums match, collates the PhysTime batch, builds the real VideoMAE-S model,
runs CUDA forward/backward/inference, uses `build_optimizer` for adapter and
detector coverage, and writes:

```json
{
  "gate_pass": true,
  "input_source": "raw_thumos_mp4",
  "logical_window": 768,
  "decoded_frame_count": 384,
  "backbone_feature_length": 384,
  "selected_index_checksum_match": true,
  "adapter_gradient_nonzero": true,
  "projection_gradient_nonzero": true,
  "classification_gradient_nonzero": true,
  "regression_gradient_nonzero": true,
  "endpoint_gradient_nonzero": true,
  "prediction_time_unit": "seconds",
  "uses_preextracted_features": false
}
```

- [ ] **Step 4: Run static gate tests**

```bash
python -m pytest tests/test_phystime_adatad_gate_contract.py -q
bash -n scripts/run_phystime_adatad_gate_gpu1.sh
```

Expected: all tests pass and the launcher is valid Bash.

- [ ] **Step 5: Commit Task 5**

```bash
git add tools/bata/run_phystime_adatad_real_gate.py scripts/run_phystime_adatad_gate_gpu1.sh tests/test_phystime_adatad_gate_contract.py
git commit -m "Add real-video PhysTime-AdaTAD gate"
```

### Task 6: Formal Training and Cost Launchers

**Files:**
- Create: `scripts/run_phystime_adatad_full_train_gpu1.sh`
- Create: `scripts/submit_phystime_adatad_head_comparison.sh`
- Create: `tests/test_phystime_adatad_deployment.py`

- [ ] **Step 1: Write failing deployment tests**

```python
def test_submission_is_gate_dependent_and_has_three_heads():
    from pathlib import Path

    text = Path("scripts/submit_phystime_adatad_head_comparison.sh").read_text()
    assert "afterok:${gate_job}" in text
    assert "phystime_adatad_sparse_k384" in text
    assert "selected_axis_adatad_sparse_k384" in text
    assert "physical_grid_adatad_sparse_k384" in text
    assert "prepare_phystime_thumos_i3d" not in text
    assert "PHYSTIME_FEATURE_PATH" not in text
```

- [ ] **Step 2: Run deployment tests and verify RED**

```bash
python -m pytest tests/test_phystime_adatad_deployment.py -q
```

Expected: failure because the scripts are absent.

- [ ] **Step 3: Implement the formal launcher**

The training launcher requires a raw-video config, raw THUMOS paths, the
VideoMAE-S checkpoint, and a passed real-gate JSON. It records commit, resolved
config hash, selected policy, K, runtime environment, peak memory, and
wall-clock time. It must fail if any feature-path environment variable is set.

- [ ] **Step 4: Implement the gate-dependent submission script**

Submit one real gate followed by exactly three K=384 jobs. All three depend on
the same gate and use seed 42. The manifest records that the decoded-frame
budget is 384 of a logical 768 positions and that Phase 2 is held.

- [ ] **Step 5: Verify launchers**

```bash
bash -n scripts/run_phystime_adatad_full_train_gpu1.sh scripts/submit_phystime_adatad_head_comparison.sh
python -m pytest tests/test_phystime_adatad_deployment.py -q
```

Expected: all checks pass.

- [ ] **Step 6: Commit Task 6**

```bash
git add scripts/run_phystime_adatad_full_train_gpu1.sh scripts/submit_phystime_adatad_head_comparison.sh tests/test_phystime_adatad_deployment.py
git commit -m "Deploy PhysTime-AdaTAD head comparison"
```

### Task 7: Full Verification and Remote Deployment

**Files:**
- Modify: `docs/evaluation/results.md`

- [ ] **Step 1: Run the complete focused suite locally or on the remote Torch environment**

```bash
python -m py_compile opentad/datasets/transforms/phystime_raw.py tools/bata/run_phystime_adatad_real_gate.py tools/bata/validate_phystime_adatad_track.py
python -m pytest tests/test_phystime_geometry.py tests/test_phystime_measure_attention.py tests/test_phystime_head.py tests/test_phystime_detector.py tests/test_phystime_raw_frame_geometry.py tests/test_phystime_adatad_configs.py tests/test_phystime_adatad_one_step.py tests/test_phystime_adatad_gate_contract.py tests/test_phystime_adatad_deployment.py -q
```

Expected: zero failures.

- [ ] **Step 2: Push the implementation branch and create an immutable remote snapshot**

```bash
git push origin codex/phystime-adatad-1
```

Clone the exact commit under
`/data/run01/sczc063/yuzibo/projects/opentad_phystime_adatad_<commit>_20260711`.

- [ ] **Step 3: Submit and wait for the real raw-video gate**

```bash
bash scripts/submit_phystime_adatad_head_comparison.sh
```

Expected: the script records one real-gate job and exactly three dependent
training jobs; no I3D data job exists.

- [ ] **Step 4: Wait for the real gate and verify dependency release**

```bash
sacct -j "$GATE_JOB" --format=JobID,State,ExitCode,Elapsed -X
cat "$RUN_ROOT/real_gate/real_gate.json"
squeue -j "$SELECTED_JOB,$PHYSICAL_JOB,$PHYSTIME_JOB" -o '%.18i %.2t %.30R'
```

Expected: gate state is `COMPLETED`, `gate_pass=true`, and the three jobs no
longer report an unsatisfied gate dependency.

- [ ] **Step 5: Record verified status**

Update `docs/evaluation/results.md` with the implementation commit, gate job,
three training job IDs, and only metrics already present in authoritative logs.

- [ ] **Step 6: Commit the deployment record**

```bash
git add docs/evaluation/results.md
git commit -m "Record PhysTime-AdaTAD deployment"
git push origin codex/phystime-adatad-1
```

## Plan Self-Review

- Spec coverage: raw sampling, seconds/original-frame mapping, official AdaTAD,
  matched heads, gradient proof, real gate, deployment, and claim boundaries
  all map to explicit tasks.
- Placeholder scan: no deferred implementation markers are present.
- Type consistency: the plan consistently uses
  `BuildPhysTimeRawFrameGeometry`, `[B, 384, K]`, K=384, a logical 768-position
  window, and `prediction_time_unit="seconds"`.
- Scope control: learned selection, dynamic budgets, paired consistency, and
  robustness curves remain outside the first implementation/deployment gate.
