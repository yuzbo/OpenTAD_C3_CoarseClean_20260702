from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.models import build_head
from opentad.models.detectors.single_stage import SingleStageDetector
from opentad.models.detectors.tridet import TriDet
from opentad.models.selectors.duca_rime_frame_selector import (
    DucaRimeFrameSelector,
)
from opentad.models.utils.truetime_geometry import (
    SELECTED_AXIS,
    TRUE_TIME_AXIS,
    truetime_map_from_metadata,
)
from tools.bata.calibrate_duca_numeric_null import calibrate_numeric_null
from tools.bata.duca_acquisition_gate_schema import (
    ADMISSION_SCHEMA,
    SELECTED_AXIS_CONTRACT,
    validate_duca_acquisition_admission_v2,
)
from tools.bata.duca_evidence_io import (
    canonical_sha256,
    verify_content_sha256,
    with_content_sha256,
    write_json_exclusive_atomic,
)
from tools.bata.duca_gate_diagnostics import (
    implemented_uniform_axis_geometry_report,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_CALIBRATION_SCHEMA = "duca_numeric_null_calibration_v1"
_SCIENTIFIC_SCHEMA = "duca_acquisition_scientific_protocol_v1"
_CALIBRATION_SEEDS = (3407, 5801, 8123)


class AcquisitionRuntimeGateFailure(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AcquisitionRuntimeGateFailure(
            f"DUCA acquisition runtime gate failed: {message}"
        )


def _sha256_file(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"artifact is missing: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


def _driver_version() -> str:
    try:
        value = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            encoding="utf-8",
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AcquisitionRuntimeGateFailure(
            f"could not identify the NVIDIA driver: {exc}"
        ) from exc
    _require(bool(value), "NVIDIA driver version is empty")
    return value.splitlines()[0].strip()


def _assert_external_output(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    try:
        target.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise AcquisitionRuntimeGateFailure(
            "runtime evidence must be written outside the Git worktree"
        )
    _require(not target.exists(), f"refusing to overwrite runtime evidence: {target}")
    return target


def _bind_runtime(
    expected_commit: str,
    expected_branch: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = str(expected_commit).lower()
    _require(_COMMIT.fullmatch(expected) is not None, "exact commit is required")
    _require(_git_output("rev-parse", "HEAD") == expected, "Git commit drift")
    _require(
        not _git_output("status", "--porcelain", "--untracked-files=normal"),
        "tracked/untracked tree must be clean",
    )
    branch = _git_output("rev-parse", "--abbrev-ref", "HEAD")
    _require(branch == str(expected_branch), "Git branch drift")
    _require(os.environ.get("SLURM_JOB_ID") is not None, "Slurm is required")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    _require(torch.cuda.device_count() == 1, "one logical Slurm GPU is required")
    _require(int(os.environ.get("WORLD_SIZE", "1")) == 1, "WORLD_SIZE must be one")
    _require(int(os.environ.get("LOCAL_RANK", "0")) == 0, "LOCAL_RANK must be zero")
    torch.cuda.set_device(0)
    deterministic = {
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    }
    identity = {
        "remote": _git_output("remote", "get-url", "origin"),
        "branch": branch,
        "git_commit": expected,
        "git_tree": _git_output("rev-parse", "HEAD^{tree}"),
        "tracked_tree_clean": True,
    }
    runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
        "gpu_name": torch.cuda.get_device_name(0),
        "driver": _driver_version(),
        "amp_enabled": True,
        "amp_dtype": "float16",
        "deterministic_flags": deterministic,
        "slurm_job_id": str(os.environ["SLURM_JOB_ID"]),
        "logical_device": "cuda:0",
    }
    return identity, runtime


def _configure_config_environment(
    *,
    train_block_list: str | Path,
    development_block_list: str | Path,
    targets_jsonl: str | Path,
    budget_protocol: str | Path,
) -> dict[str, str]:
    paths = {
        "DUCA_RIME_TRAIN_BLOCK_LIST": str(
            Path(train_block_list).expanduser().resolve()
        ),
        "DUCA_RIME_DEVELOPMENT_BLOCK_LIST": str(
            Path(development_block_list).expanduser().resolve()
        ),
        "DUCA_RIME_TARGETS_JSONL": str(Path(targets_jsonl).expanduser().resolve()),
        "DUCA_RIME_BUDGET_PROTOCOL_JSON": str(
            Path(budget_protocol).expanduser().resolve()
        ),
    }
    for label, path in paths.items():
        _require(Path(path).is_file(), f"{label} is missing: {path}")
    paths["DUCA_RIME_TARGETS_SHA256"] = _sha256_file(
        paths["DUCA_RIME_TARGETS_JSONL"]
    )
    paths["DUCA_RIME_BUDGET_PROTOCOL_SHA256"] = _sha256_file(
        paths["DUCA_RIME_BUDGET_PROTOCOL_JSON"]
    )
    for key, value in paths.items():
        os.environ[key] = value
    return paths


def _normal_head_config(config: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(config))
    physical = output.pop("physical_grid_actionformer", None)
    _require(
        physical is None,
        "selected/standard head config unexpectedly enables physical-grid logic",
    )
    return output


def _load_selected_standard_pair(
    *,
    selected_path: str | Path,
    standard_path: str | Path,
    expected_backend: str,
) -> tuple[Config, Config, dict[str, Any]]:
    selected_resolved = Path(selected_path).expanduser().resolve()
    standard_resolved = Path(standard_path).expanduser().resolve()
    _require(selected_resolved.is_file(), f"selected config missing: {selected_resolved}")
    _require(standard_resolved.is_file(), f"standard config missing: {standard_resolved}")
    selected = Config.fromfile(str(selected_resolved))
    standard = Config.fromfile(str(standard_resolved))
    selected_contract = selected.get("duca_rime_contract", {})
    selected_mode = selected.model.frame_selector.get(
        "detector_coordinate_mode",
        None,
    )
    _require(
        selected_mode == "selected_axis_plugin",
        f"{expected_backend} selected config is not a selected-axis plugin",
    )
    _require(
        selected.model.rpn_head.get("physical_grid_actionformer", None) is None,
        f"{expected_backend} selected config enables the physical head",
    )
    _require(
        selected_contract.get("pre_backbone_plugin") is True
        and selected_contract.get("detector_head_modified") is False
        and selected_contract.get("physical_head_enabled") is False
        and selected_contract.get("paper_mainline_allowed") is True
        and selected_contract.get("admission_schema") == ADMISSION_SCHEMA,
        f"{expected_backend} selected config contract drift",
    )
    selected_head = _normal_head_config(selected.model.rpn_head)
    standard_head = _normal_head_config(standard.model.rpn_head)
    _require(
        selected_head == standard_head,
        f"{expected_backend} selected and standard head configs differ",
    )
    return selected, standard, {
        "selected_path": str(selected_resolved),
        "selected_sha256": _sha256_file(selected_resolved),
        "standard_path": str(standard_resolved),
        "standard_sha256": _sha256_file(standard_resolved),
        "head_config_sha256": canonical_sha256(selected_head),
    }


def _checkpoint_state_mappings(payload: Any) -> list[tuple[str, Mapping[str, Any]]]:
    _require(isinstance(payload, Mapping), "checkpoint payload must be a mapping")
    output: list[tuple[str, Mapping[str, Any]]] = []
    for key in ("state_dict_ema", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            output.append((key, value))
    if payload and all(isinstance(key, str) for key in payload):
        output.append(("root", payload))
    _require(bool(output), "checkpoint has no supported state mapping")
    return output


def _extract_rpn_head_state(
    checkpoint_path: str | Path,
    *,
    expected_keys: set[str],
) -> tuple[dict[str, torch.Tensor], str]:
    resolved = Path(checkpoint_path).expanduser().resolve()
    _require(resolved.is_file(), f"checkpoint missing: {resolved}")
    try:
        payload = torch.load(str(resolved), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(resolved), map_location="cpu")
    diagnostics = []
    for source_name, source in _checkpoint_state_mappings(payload):
        output: dict[str, torch.Tensor] = {}
        for raw_key, value in source.items():
            key = str(raw_key)
            while key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("rpn_head."):
                head_key = key[len("rpn_head.") :]
            elif key.startswith("model.rpn_head."):
                head_key = key[len("model.rpn_head.") :]
            else:
                continue
            _require(torch.is_tensor(value), f"non-tensor rpn_head value: {raw_key}")
            _require(head_key not in output, f"duplicate rpn_head key: {head_key}")
            output[head_key] = value.detach().cpu()
        missing = sorted(expected_keys - set(output))
        unexpected = sorted(set(output) - expected_keys)
        diagnostics.append(
            {
                "source": source_name,
                "key_count": len(output),
                "missing_count": len(missing),
                "unexpected_count": len(unexpected),
            }
        )
        if not missing and not unexpected:
            return output, source_name
    raise AcquisitionRuntimeGateFailure(
        "checkpoint has no complete exact rpn_head state mapping: "
        f"{resolved}; candidates={diagnostics}"
    )


def _build_restored_head_pair(
    selected: Config,
    standard: Config,
    *,
    checkpoint_path: str | Path,
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    selected_head = build_head(copy.deepcopy(selected.model.rpn_head))
    standard_head = build_head(copy.deepcopy(standard.model.rpn_head))
    _require(
        set(selected_head.state_dict()) == set(standard_head.state_dict()),
        "standard/candidate head state keys differ before checkpoint loading",
    )
    checkpoint_state, checkpoint_mapping = _extract_rpn_head_state(
        checkpoint_path,
        expected_keys=set(selected_head.state_dict()),
    )
    selected_result = selected_head.load_state_dict(checkpoint_state, strict=True)
    standard_result = standard_head.load_state_dict(checkpoint_state, strict=True)
    _require(
        not selected_result.missing_keys
        and not selected_result.unexpected_keys
        and not standard_result.missing_keys
        and not standard_result.unexpected_keys,
        "standard/candidate head checkpoint compatibility is not strict",
    )
    _require(
        list(selected_head.state_dict()) == list(standard_head.state_dict()),
        "standard/candidate head state keys differ",
    )
    for key, value in selected_head.state_dict().items():
        _require(
            value.shape == standard_head.state_dict()[key].shape,
            f"standard/candidate head state shape differs for {key}",
        )
    for head in (selected_head, standard_head):
        _require(
            getattr(head, "physical_grid_enabled", None) is False,
            "restored standard head unexpectedly enables physical-grid logic",
        )
        head.to("cuda:0").eval()
    return selected_head, standard_head, {
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_mapping": checkpoint_mapping,
        "state_key_count": len(checkpoint_state),
        "strict_load": True,
    }


def _named_tensors(value: Any, prefix: str = "root") -> dict[str, torch.Tensor]:
    if torch.is_tensor(value):
        return {prefix: value}
    output: dict[str, torch.Tensor] = {}
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            output.update(_named_tensors(value[key], f"{prefix}.{key}"))
        return output
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            output.update(_named_tensors(item, f"{prefix}.{index}"))
        return output
    return output


def tensor_tree_max_abs_error(left: Any, right: Any) -> float:
    left_tensors = _named_tensors(left)
    right_tensors = _named_tensors(right)
    _require(
        set(left_tensors) == set(right_tensors),
        "numeric-null output structures differ",
    )
    maximum = 0.0
    for key in sorted(left_tensors):
        lhs = left_tensors[key].detach().float()
        rhs = right_tensors[key].detach().float()
        _require(lhs.shape == rhs.shape, f"numeric-null shape differs for {key}")
        if lhs.numel():
            error = float((lhs - rhs).abs().max().item())
            _require(math.isfinite(error), f"numeric-null error is non-finite for {key}")
            maximum = max(maximum, error)
    return maximum


def _head_feature_fixture(
    head: nn.Module,
    seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    generator = torch.Generator(device="cuda:0")
    generator.manual_seed(int(seed))
    channels = int(head.in_channels)
    strides = tuple(float(value) for value in head.prior_generator.strides)
    _require(bool(strides), "head prior generator has no feature levels")
    _require(
        all(math.isfinite(value) and value > 0.0 for value in strides),
        "head prior strides must be finite and positive",
    )
    base_stride = min(strides)
    lengths = tuple(
        max(1, int(math.ceil(64.0 * base_stride / stride))) for stride in strides
    )
    scale = getattr(head, "scale", None)
    _require(
        scale is None or len(scale) == len(lengths),
        "head scale/prior feature-level count drift",
    )
    features = [
        torch.randn(
            1,
            channels,
            length,
            generator=generator,
            device="cuda:0",
            dtype=torch.float32,
        )
        for length in lengths
    ]
    masks = [
        torch.ones(1, length, device="cuda:0", dtype=torch.bool)
        for length in lengths
    ]
    return features, masks


def _run_head_null(
    candidate: nn.Module,
    standard: nn.Module,
    *,
    backend: str,
    seed: int,
    autocast_enabled: bool,
) -> dict[str, Any]:
    _require(
        tuple(candidate.prior_generator.strides)
        == tuple(standard.prior_generator.strides),
        f"{backend} candidate/standard prior strides differ",
    )
    features, masks = _head_feature_fixture(candidate, seed)
    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=bool(autocast_enabled),
        cache_enabled=False,
    ):
        candidate_output = candidate.forward_test(features, masks, metas=None)
        standard_output = standard.forward_test(features, masks, metas=None)
    error = tensor_tree_max_abs_error(candidate_output, standard_output)
    return {
        "run_id": f"{backend.lower()}-seed{int(seed)}",
        "backend": str(backend),
        "seed": int(seed),
        "split_scope": "train_only_calibration",
        "uses_official_final": False,
        "autocast_enabled": bool(autocast_enabled),
        "amp_dtype": "float16" if autocast_enabled else None,
        "metric_errors": {"output_max_abs": error},
    }


def _state_value(value: Any) -> Any:
    if torch.is_tensor(value):
        tensor = value.detach().contiguous().cpu()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {
            "kind": "tensor",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "kind": "ndarray",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {str(key): _state_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_state_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _capture_runtime_state(model: nn.Module) -> dict[str, Any]:
    return {
        "parameters": {
            name: value.detach().to(device="cpu", copy=True)
            for name, value in model.named_parameters()
        },
        "buffers": {
            name: value.detach().to(device="cpu", copy=True)
            for name, value in model.named_buffers()
        },
        "module_training": {
            name: bool(module.training) for name, module in model.named_modules()
        },
        "custom_replay": {
            name: copy.deepcopy(module.capture_amp_replay_state())
            for name, module in model.named_modules()
            if callable(getattr(module, "capture_amp_replay_state", None))
            and callable(getattr(module, "restore_amp_replay_state", None))
        },
        "cpu_rng": torch.random.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
    }


def _restore_runtime_state(model: nn.Module, state: Mapping[str, Any]) -> None:
    modules = dict(model.named_modules())
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    _require(
        set(parameters) == set(state["parameters"])
        and set(buffers) == set(state["buffers"])
        and set(modules) == set(state["module_training"]),
        "module state surface drifted during the runtime gate",
    )
    with torch.no_grad():
        for name, value in state["parameters"].items():
            parameters[name].copy_(value)
        for name, value in state["buffers"].items():
            buffers[name].copy_(value)
    for name, training in state["module_training"].items():
        modules[name].training = bool(training)
    for name, snapshot in state["custom_replay"].items():
        _require(name in modules, f"custom replay module disappeared: {name}")
        modules[name].restore_amp_replay_state(snapshot)
    torch.random.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])


def _runtime_state_sha256(state: Mapping[str, Any]) -> str:
    return canonical_sha256(_state_value(state))


def _build_uniform_runtime_selector() -> DucaRimeFrameSelector:
    selector = DucaRimeFrameSelector(
        in_channels=3,
        rime_arm="uniform_mixed_k",
        candidate_budgets=(192, 384),
        candidate_costs=(192.0, 384.0),
        fixed_budget=384,
        dense_window_size=768,
        target_mean_cost=288.0,
        execution_quantum=16,
        require_frozen_protocol=False,
        mixed_k_schedule_counts=(1, 1),
        mixed_k_schedule_seed=3407,
        detector_bridge_gradient_scale=0.0,
        actionness_source_cfg=None,
        detector_coordinate_mode="selected_axis_plugin",
    )
    return selector.to("cuda:0").eval()


def _selector_window(
    selector: DucaRimeFrameSelector,
    *,
    role: str,
    valid_len: int,
    seed: int,
) -> tuple[dict[str, Any], float]:
    generator = torch.Generator(device="cuda:0")
    generator.manual_seed(int(seed))
    inputs = torch.randn(
        1,
        3,
        768,
        2,
        2,
        generator=generator,
        device="cuda:0",
        dtype=torch.float32,
    )
    masks = torch.zeros(1, 768, device="cuda:0", dtype=torch.bool)
    masks[:, : int(valid_len)] = True
    metas = [
        {
            "frame_inds": list(range(768)),
            "avg_fps": 1.0,
            "video_name": role,
            "window_start_frame": 0,
        }
    ]
    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
        cache_enabled=False,
    ):
        output = selector.forward_test(inputs, masks, metas)
    meta = output["metas"][0]
    positions = [
        int(value)
        for value in meta["selected_axis_to_true_time_dense_index"]
    ]
    effective = len(positions)
    requested = int(meta["duca_requested_k"])
    backbone_k = int(output["inputs"].shape[2])
    active = int(output["masks"][0].sum().item())
    padded = int(meta["duca_padded_k"])
    _require(
        requested >= effective > 0
        and effective == backbone_k == active == padded,
        f"{role} violates exact-K/no-padding",
    )
    _require(
        positions == sorted(set(positions))
        and positions[0] >= 0
        and positions[-1] < int(valid_len),
        f"{role} selected positions are invalid",
    )
    time_map = truetime_map_from_metadata(meta, require_inverse_map=True)
    true_segments = torch.tensor(
        [[
            min(16.0, float(valid_len) / 4.0),
            max(17.0, float(valid_len) * 0.75),
        ]],
        device="cuda:0",
    ).clamp(max=float(valid_len))
    selected_segments = time_map.remap_segments(
        true_segments,
        source_coordinate_space=TRUE_TIME_AXIS,
        target_coordinate_space=SELECTED_AXIS,
    )
    roundtrip = time_map.remap_segments(
        selected_segments,
        source_coordinate_space=SELECTED_AXIS,
        target_coordinate_space=TRUE_TIME_AXIS,
    )
    roundtrip_error = float((roundtrip - true_segments).abs().max().item())
    _require(math.isfinite(roundtrip_error), f"{role} roundtrip is non-finite")

    probe = torch.tensor(
        [[0.25, max(0.5, float(effective) - 0.25)]],
        dtype=torch.float32,
    )
    once, once_meta = SingleStageDetector._remap_selector_segments_for_post_processing(
        probe,
        meta,
    )
    twice, twice_meta = SingleStageDetector._remap_selector_segments_for_post_processing(
        once,
        once_meta,
    )
    _require(
        once_meta["detector_output_coordinate_space"] == TRUE_TIME_AXIS
        and twice_meta["detector_output_coordinate_space"] == TRUE_TIME_AXIS
        and torch.equal(once, twice),
        f"{role} proposal mapping was not exactly once",
    )
    return {
        "role": role,
        "valid_len": int(valid_len),
        "requested_k": requested,
        "effective_k": effective,
        "backbone_input_k": backbone_k,
        "active_mask_count": active,
        "padded_k": padded,
        "selected_positions": positions,
        "implemented_map": implemented_uniform_axis_geometry_report(
            valid_len=int(valid_len),
            positions=positions,
        ),
        "selector_contract": meta.get("duca_contract"),
        "detector_output_coordinate_space": meta.get(
            "detector_output_coordinate_space"
        ),
        "prediction_inverse_map_required": meta.get(
            "detector_prediction_inverse_map_required"
        ),
        "first_mapping_output_coordinate_space": once_meta.get(
            "detector_output_coordinate_space"
        ),
        "second_mapping_is_noop": True,
    }, roundtrip_error


def _gt_training_remap_smoke(selector: DucaRimeFrameSelector) -> dict[str, Any]:
    generator = torch.Generator(device="cuda:0")
    generator.manual_seed(9127)
    inputs = torch.randn(
        1,
        3,
        768,
        2,
        2,
        generator=generator,
        device="cuda:0",
        dtype=torch.float32,
    )
    masks = torch.ones(1, 768, device="cuda:0", dtype=torch.bool)
    metas = [
        {
            "frame_inds": list(range(768)),
            "avg_fps": 1.0,
            "video_name": "training_gt_remap",
            "duca_stateless_epoch": 0,
            "duca_stateless_sample_index": 0,
        }
    ]
    original = torch.tensor([[64.0, 512.0]], device="cuda:0")
    selector.train()
    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
        cache_enabled=False,
    ):
        output = selector.forward_train(
            inputs,
            masks,
            metas,
            gt_segments=[original],
            gt_labels=[torch.tensor([1], device="cuda:0")],
        )
    selector.eval()
    meta = output["metas"][0]
    _require(
        meta.get("gt_remapped_to_selected_axis") is True,
        "training GT was not remapped to selected-axis coordinates",
    )
    time_map = truetime_map_from_metadata(meta, require_inverse_map=True)
    recovered = time_map.remap_segments(
        output["gt_segments"][0],
        source_coordinate_space=SELECTED_AXIS,
        target_coordinate_space=TRUE_TIME_AXIS,
    )
    error = float((recovered - original).abs().max().item())
    _require(math.isfinite(error), "training GT roundtrip is non-finite")
    return {
        "gt_remapped_to_selected_axis": True,
        "roundtrip_max_abs_error": error,
        "label_identity_preserved": bool(
            torch.equal(output["gt_labels"][0], torch.tensor([1], device="cuda:0"))
        ),
    }


def _mapping_precedes_nms() -> dict[str, bool]:
    output = {}
    for label, owner in (
        ("actionformer", SingleStageDetector),
        ("tridet", TriDet),
    ):
        source = inspect.getsource(owner.post_processing)
        remap_index = source.find("_remap_selector_segments_for_post_processing")
        nms_index = source.find("batched_nms")
        _require(
            remap_index >= 0 and nms_index >= 0 and remap_index < nms_index,
            f"{label} does not remap selected-axis proposals before official NMS",
        )
        output[label] = True
    return output


def _load_calibration(
    path: str | Path,
    *,
    expected_commit: str,
    expected_runtime_fingerprint: str,
) -> tuple[dict[str, Any], str]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"numeric calibration missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "numeric calibration must be an object")
    verify_content_sha256(payload)
    _require(
        payload.get("schema") == _CALIBRATION_SCHEMA
        and payload.get("status") == "frozen"
        and payload.get("git_commit") == expected_commit
        and payload.get("uses_official_final") is False
        and payload.get("frozen_before_candidate_development") is True
        and payload.get("runtime_fingerprint_sha256")
        == expected_runtime_fingerprint,
        "numeric calibration identity/scope drift",
    )
    thresholds = payload.get("thresholds")
    _require(isinstance(thresholds, Mapping), "numeric thresholds must be a mapping")
    _require("output_max_abs" in thresholds, "numeric calibration lacks output threshold")
    for key, value in thresholds.items():
        numeric = float(value)
        _require(
            math.isfinite(numeric) and numeric >= 0.0,
            f"numeric calibration threshold is invalid: {key}",
        )
    return dict(payload), _sha256_file(resolved)


def _artifact_binding(path: str | Path) -> dict[str, str]:
    resolved = Path(path).expanduser().resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
    }


def validate_scientific_protocol(
    payload: Mapping[str, Any],
    *,
    expected_commit: str,
    expected_identity: Mapping[str, Any] | None = None,
    expected_candidate_output_root: str | Path | None = None,
) -> dict[str, Any]:
    output = dict(payload)
    if "content_sha256" in output:
        verify_content_sha256(output)
    margin = float(output.get("noninferiority_margin", float("nan")))
    _require(
        output.get("schema") == _SCIENTIFIC_SCHEMA
        and output.get("status") == "frozen"
        and output.get("git_commit") == expected_commit
        and output.get("frozen_before_candidate_development") is True
        and output.get("uses_official_final") is False
        and output.get("paper_claim_allowed") is False
        and output.get("phase4_submission_enabled") is False
        and output.get("official_final_sealed") is True,
        "scientific protocol identity/scope drift",
    )
    _require(
        bool(str(output.get("primary_endpoint", "")).strip()),
        "scientific protocol primary endpoint is missing",
    )
    _require(
        math.isfinite(margin) and margin >= 0.0,
        "scientific protocol NI margin must be finite and non-negative",
    )
    _require(
        output.get("multiplicity_procedure") in {"holm", "closed_testing"},
        "scientific multiplicity procedure is unsupported",
    )
    guardrails = output.get("guardrails")
    stopping = output.get("stopping_rules")
    _require(
        isinstance(guardrails, Sequence)
        and not isinstance(guardrails, (str, bytes))
        and bool(guardrails),
        "scientific guardrails are missing",
    )
    _require(
        isinstance(stopping, Sequence)
        and not isinstance(stopping, (str, bytes))
        and bool(stopping),
        "scientific stopping rules are missing",
    )
    anchor = output.get("preregistration_anchor")
    _require(isinstance(anchor, Mapping), "scientific preregistration anchor is missing")
    _require(
        anchor.get("schema") == "duca_acquisition_preregistration_anchor_v1"
        and anchor.get("git_commit") == expected_commit
        and anchor.get("candidate_output_root_absent") is True
        and anchor.get("candidate_results_observed") is False,
        "scientific preregistration anchor scope drift",
    )
    if expected_identity is not None:
        for key in ("remote", "branch", "git_commit", "git_tree", "repo_root"):
            _require(
                str(anchor.get(key)) == str(expected_identity.get(key)),
                f"scientific preregistration identity drift: {key}",
            )
    if expected_candidate_output_root is not None:
        _require(
            Path(str(anchor.get("candidate_output_root"))).expanduser().resolve()
            == Path(expected_candidate_output_root).expanduser().resolve(),
            "scientific preregistration candidate output root drift",
        )
    margin_source = output.get("margin_source")
    _require(isinstance(margin_source, Mapping), "scientific margin source is missing")
    margin_path = Path(str(margin_source.get("path", ""))).expanduser().resolve()
    _require(margin_path.is_file(), f"scientific margin source missing: {margin_path}")
    _require(
        _sha256_file(margin_path) == margin_source.get("sha256"),
        "scientific margin source raw SHA-256 drift",
    )
    margin_payload = json.loads(margin_path.read_text(encoding="utf-8"))
    _require(isinstance(margin_payload, Mapping), "scientific margin source is invalid")
    verify_content_sha256(margin_payload)
    _require(
        margin_payload.get("content_sha256") == margin_source.get("content_sha256"),
        "scientific margin source content SHA-256 drift",
    )
    return output


def _load_scientific_protocol(
    path: str | Path,
    *,
    expected_commit: str,
    expected_identity: Mapping[str, Any],
    expected_candidate_output_root: str | Path,
) -> tuple[dict[str, Any], str]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"scientific protocol missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "scientific protocol must be an object")
    return (
        validate_scientific_protocol(
            payload,
            expected_commit=expected_commit,
            expected_identity=expected_identity,
            expected_candidate_output_root=expected_candidate_output_root,
        ),
        _sha256_file(resolved),
    )


def _runtime_fingerprint(runtime: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            key: runtime[key]
            for key in (
                "python",
                "torch",
                "cuda_runtime",
                "cudnn",
                "gpu_name",
                "driver",
                "amp_enabled",
                "amp_dtype",
                "deterministic_flags",
            )
        }
    )


def run_runtime_gate(
    *,
    expected_commit: str,
    expected_branch: str,
    selected_actionformer_config: str | Path,
    standard_actionformer_config: str | Path,
    actionformer_checkpoint: str | Path,
    selected_tridet_config: str | Path,
    standard_tridet_config: str | Path,
    tridet_checkpoint: str | Path,
    train_block_list: str | Path,
    development_block_list: str | Path,
    targets_jsonl: str | Path,
    budget_protocol: str | Path,
    data_manifest: str | Path,
    split_assignment: str | Path,
    code_gate_receipt: str | Path,
    calibration_output: str | Path | None,
    numeric_calibration: str | Path | None,
    scientific_protocol: str | Path | None,
    evidence_output: str | Path | None,
    safety_multiplier: float,
    absolute_floor: float,
) -> dict[str, Any]:
    identity, runtime = _bind_runtime(expected_commit, expected_branch)
    identity["repo_root"] = str(ROOT)
    runtime_fingerprint = _runtime_fingerprint(runtime)
    configured_paths = _configure_config_environment(
        train_block_list=train_block_list,
        development_block_list=development_block_list,
        targets_jsonl=targets_jsonl,
        budget_protocol=budget_protocol,
    )
    af_selected, af_standard, af_config = _load_selected_standard_pair(
        selected_path=selected_actionformer_config,
        standard_path=standard_actionformer_config,
        expected_backend="ActionFormer",
    )
    td_selected, td_standard, td_config = _load_selected_standard_pair(
        selected_path=selected_tridet_config,
        standard_path=standard_tridet_config,
        expected_backend="TriDet",
    )
    af_candidate_head, af_standard_head, af_checkpoint = _build_restored_head_pair(
        af_selected,
        af_standard,
        checkpoint_path=actionformer_checkpoint,
    )
    td_candidate_head, td_standard_head, td_checkpoint = _build_restored_head_pair(
        td_selected,
        td_standard,
        checkpoint_path=tridet_checkpoint,
    )
    selector = _build_uniform_runtime_selector()
    modules = nn.ModuleDict(
        {
            "selector": selector,
            "actionformer_candidate": af_candidate_head,
            "actionformer_standard": af_standard_head,
            "tridet_candidate": td_candidate_head,
            "tridet_standard": td_standard_head,
        }
    )
    modules.eval()
    baseline_state = _capture_runtime_state(modules)
    state_before_sha256 = _runtime_state_sha256(baseline_state)
    try:
        amp_rows = []
        for backend, candidate, standard in (
            ("ActionFormer", af_candidate_head, af_standard_head),
            ("TriDet", td_candidate_head, td_standard_head),
        ):
            for seed in _CALIBRATION_SEEDS:
                amp_rows.append(
                    _run_head_null(
                        candidate,
                        standard,
                        backend=backend,
                        seed=seed,
                        autocast_enabled=True,
                    )
                )
        if calibration_output is not None:
            output_path = _assert_external_output(calibration_output)
            calibration = calibrate_numeric_null(
                amp_rows,
                git_commit=expected_commit,
                safety_multiplier=safety_multiplier,
                absolute_floor=absolute_floor,
            )
            calibration.pop("content_sha256", None)
            calibration.update(
                {
                    "runtime_fingerprint_sha256": runtime_fingerprint,
                    "runtime": runtime,
                    "backends": ["ActionFormer", "TriDet"],
                    "candidate_performance_observed": False,
                    "calibration_only": True,
                    "producer": {
                        "schema": "duca_acquisition_runtime_producer_v2",
                        "module": "tools.bata.run_duca_acquisition_runtime_gate_v2",
                        "script": _artifact_binding(Path(__file__)),
                        "launcher": _artifact_binding(
                            ROOT / "scripts" / "run_duca_acquisition_admission_v2.sh"
                        ),
                        "slurm_job_id": runtime["slurm_job_id"],
                        "git_commit": expected_commit,
                        "git_tree": identity["git_tree"],
                    },
                    "artifact_bindings": {
                        "code_gate_receipt": _artifact_binding(code_gate_receipt),
                        "actionformer_checkpoint": _artifact_binding(
                            actionformer_checkpoint
                        ),
                        "tridet_checkpoint": _artifact_binding(tridet_checkpoint),
                        "train_block_list": _artifact_binding(train_block_list),
                        "development_block_list": _artifact_binding(
                            development_block_list
                        ),
                        "targets_jsonl": _artifact_binding(targets_jsonl),
                        "budget_protocol": _artifact_binding(budget_protocol),
                        "data_manifest": _artifact_binding(data_manifest),
                        "split_assignment": _artifact_binding(split_assignment),
                    },
                }
            )
            calibration = with_content_sha256(calibration)
            write_json_exclusive_atomic(output_path, calibration)
            return calibration

        _require(
            numeric_calibration is not None
            and scientific_protocol is not None
            and evidence_output is not None,
            "admission mode requires calibration, scientific protocol and evidence output",
        )
        output_path = _assert_external_output(evidence_output)
        calibration, calibration_sha = _load_calibration(
            numeric_calibration,
            expected_commit=expected_commit,
            expected_runtime_fingerprint=runtime_fingerprint,
        )
        scientific, scientific_sha = _load_scientific_protocol(
            scientific_protocol,
            expected_commit=expected_commit,
            expected_identity=identity,
            expected_candidate_output_root=output_path.parent,
        )
        thresholds = {
            str(key): float(value)
            for key, value in calibration["thresholds"].items()
        }
        amp_null_runs = []
        for row in amp_rows:
            errors = {
                str(key): float(value)
                for key, value in row["metric_errors"].items()
            }
            within = all(
                key in thresholds
                and math.isfinite(value)
                and value <= thresholds[key]
                for key, value in errors.items()
            )
            amp_null_runs.append(
                {
                    **row,
                    "frozen_thresholds": thresholds,
                    "within_frozen_thresholds": bool(within),
                }
            )
        _require(
            all(row["within_frozen_thresholds"] for row in amp_null_runs),
            "AMP null exceeded a frozen threshold",
        )
        diagnostic_rows = [
            _run_head_null(
                af_candidate_head,
                af_standard_head,
                backend="ActionFormer",
                seed=_CALIBRATION_SEEDS[0],
                autocast_enabled=False,
            ),
            _run_head_null(
                td_candidate_head,
                td_standard_head,
                backend="TriDet",
                seed=_CALIBRATION_SEEDS[0],
                autocast_enabled=False,
            ),
        ]
        windows = []
        roundtrip_errors = []
        for role, valid_len, seed in (
            ("full_window", 768, 3407),
            ("short_window", 192, 5801),
        ):
            row, error = _selector_window(
                selector,
                role=role,
                valid_len=valid_len,
                seed=seed,
            )
            windows.append(row)
            roundtrip_errors.append(error)
        gt_remap = _gt_training_remap_smoke(selector)
        roundtrip_errors.append(float(gt_remap["roundtrip_max_abs_error"]))
        nms_order = _mapping_precedes_nms()
    finally:
        _restore_runtime_state(modules, baseline_state)
    final_state = _capture_runtime_state(modules)
    state_after_sha256 = _runtime_state_sha256(final_state)
    _require(
        state_after_sha256 == state_before_sha256,
        "runtime gate did not restore model/RNG/debug state",
    )

    config_bundle = {
        "actionformer": af_config,
        "tridet": td_config,
    }
    checkpoint_bundle = {
        "actionformer": af_checkpoint,
        "tridet": td_checkpoint,
    }
    identity.update(
        {
            "config_sha256": canonical_sha256(config_bundle),
            "checkpoint_sha256": canonical_sha256(checkpoint_bundle),
            "data_manifest_sha256": _sha256_file(data_manifest),
            "split_assignment_sha256": _sha256_file(split_assignment),
            "config_bundle": config_bundle,
            "checkpoint_bundle": checkpoint_bundle,
        }
    )
    producer = {
        "schema": "duca_acquisition_runtime_producer_v2",
        "module": "tools.bata.run_duca_acquisition_runtime_gate_v2",
        "script": _artifact_binding(Path(__file__)),
        "launcher": _artifact_binding(
            ROOT / "scripts" / "run_duca_acquisition_admission_v2.sh"
        ),
        "slurm_job_id": runtime["slurm_job_id"],
        "git_commit": expected_commit,
        "git_tree": identity["git_tree"],
        "finalized_in_runtime_producer": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    artifact_bindings = {
        "code_gate_receipt": _artifact_binding(code_gate_receipt),
        "selected_actionformer_config": _artifact_binding(
            selected_actionformer_config
        ),
        "standard_actionformer_config": _artifact_binding(
            standard_actionformer_config
        ),
        "actionformer_checkpoint": _artifact_binding(actionformer_checkpoint),
        "selected_tridet_config": _artifact_binding(selected_tridet_config),
        "standard_tridet_config": _artifact_binding(standard_tridet_config),
        "tridet_checkpoint": _artifact_binding(tridet_checkpoint),
        "train_block_list": _artifact_binding(train_block_list),
        "development_block_list": _artifact_binding(development_block_list),
        "targets_jsonl": _artifact_binding(targets_jsonl),
        "budget_protocol": _artifact_binding(budget_protocol),
        "data_manifest": _artifact_binding(data_manifest),
        "split_assignment": _artifact_binding(split_assignment),
        "numeric_calibration": _artifact_binding(numeric_calibration),
        "scientific_protocol": _artifact_binding(scientific_protocol),
    }
    _require(
        artifact_bindings["targets_jsonl"]["sha256"]
        == configured_paths["DUCA_RIME_TARGETS_SHA256"]
        and artifact_bindings["budget_protocol"]["sha256"]
        == configured_paths["DUCA_RIME_BUDGET_PROTOCOL_SHA256"],
        "configured runtime artifacts drifted before admission finalization",
    )
    roles = [row["role"] for row in windows]
    execution_rows = [
        {
            key: row[key]
            for key in (
                "role",
                "requested_k",
                "effective_k",
                "backbone_input_k",
                "active_mask_count",
                "padded_k",
                "selected_positions",
            )
        }
        for row in windows
    ]
    payload = {
        "schema": ADMISSION_SCHEMA,
        "status": "passed",
        "admission_effect": True,
        "producer": producer,
        "artifact_bindings": artifact_bindings,
        "identity": identity,
        "runtime": runtime,
        "coordinate_contract": {
            "mode": "selected_axis_plugin",
            "selector_contract": SELECTED_AXIS_CONTRACT,
            "detector_output_coordinate_space": "selected_axis_index",
            "inverse_map_before_official_nms": True,
            "mapping_applied_exactly_once": True,
            "physical_head_enabled": False,
            "gt_remapped_to_selected_axis": bool(
                gt_remap["gt_remapped_to_selected_axis"]
            ),
            "standard_detector_head_unchanged": True,
            "nms_order_checks": nms_order,
            "gt_label_identity_preserved": bool(
                gt_remap["label_identity_preserved"]
            ),
        },
        "execution": {
            "window_roles": roles,
            "requested_k": [row["requested_k"] for row in windows],
            "effective_k": [row["effective_k"] for row in windows],
            "backbone_input_k": [row["backbone_input_k"] for row in windows],
            "active_mask_count": [row["active_mask_count"] for row in windows],
            "padded_k": [row["padded_k"] for row in windows],
            "positions_sha256": canonical_sha256(
                [row["selected_positions"] for row in windows]
            ),
            "bucket_order_sha256": canonical_sha256(execution_rows),
            "rows": execution_rows,
        },
        "geometry": {
            "windows": [
                {
                    "valid_len": row["valid_len"],
                    "selected_positions": row["selected_positions"],
                    "implemented_map": row["implemented_map"],
                }
                for row in windows
            ],
            "roundtrip_max_abs_error": max(roundtrip_errors),
            "mapping_applied_exactly_once": True,
            "gt_training_remap": gt_remap,
        },
        "standard_detector_restoration": {
            "actionformer": {
                "status": "passed",
                "physical_head_enabled": False,
                "selector_disabled_null_passed": all(
                    row["within_frozen_thresholds"]
                    for row in amp_null_runs
                    if row["backend"] == "ActionFormer"
                ),
                "standard_head_state_dict_compatible": True,
                "standard_config_sha256": af_config["standard_sha256"],
                "head_config_sha256": af_config["head_config_sha256"],
                "checkpoint_state_key_count": af_checkpoint["state_key_count"],
            },
            "tridet": {
                "status": "passed",
                "physical_head_enabled": False,
                "selector_disabled_null_passed": all(
                    row["within_frozen_thresholds"]
                    for row in amp_null_runs
                    if row["backend"] == "TriDet"
                ),
                "standard_head_state_dict_compatible": True,
                "standard_config_sha256": td_config["standard_sha256"],
                "head_config_sha256": td_config["head_config_sha256"],
                "checkpoint_state_key_count": td_checkpoint["state_key_count"],
            },
        },
        "numeric": {
            "calibration_manifest_sha256": calibration_sha,
            "calibration_content_sha256": calibration["content_sha256"],
            "runtime_fingerprint_sha256": runtime_fingerprint,
            "amp_null_runs": amp_null_runs,
            "state_before_sha256": state_before_sha256,
            "state_after_sha256": state_after_sha256,
            "autocast_disabled_non_admission_replay": {
                "admission_effect": False,
                "rows": diagnostic_rows,
            },
        },
        "gates": {
            "structural_gate_passed": True,
            "numeric_gate_passed": True,
            "scientific_protocol_preregistered": True,
            "scientific_protocol_sha256": scientific_sha,
            "scientific_protocol_content_sha256": scientific.get(
                "content_sha256"
            ),
            "legacy_scalar_loss_equivalence_required": False,
        },
        "scientific_scope": {
            "uses_official_final": False,
            "paper_claim_allowed": False,
            "phase4_submission_enabled": False,
            "official_final_sealed": True,
            "primary_endpoint": scientific["primary_endpoint"],
            "noninferiority_margin": float(
                scientific["noninferiority_margin"]
            ),
            "multiplicity_procedure": scientific["multiplicity_procedure"],
        },
        "predecessor_evidence": {
            "recovery_v6_job": "1201417",
            "historical_status": "failed_under_v1",
            "historical_outcome_reclassified": False,
            "legacy_v1_read_only": True,
            "v1_universal_loss_equivalence_premise_valid": False,
        },
    }
    payload = with_content_sha256(payload)
    validate_duca_acquisition_admission_v2(
        payload,
        expected_commit=expected_commit,
        require_passed=True,
    )
    write_json_exclusive_atomic(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Slurm/CUDA selected-axis structural and numeric gate that "
            "produces a DUCA acquisition admission-v2 evidence draft."
        )
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--selected-actionformer-config", required=True)
    parser.add_argument("--standard-actionformer-config", required=True)
    parser.add_argument("--actionformer-checkpoint", required=True)
    parser.add_argument("--selected-tridet-config", required=True)
    parser.add_argument("--standard-tridet-config", required=True)
    parser.add_argument("--tridet-checkpoint", required=True)
    parser.add_argument("--train-block-list", required=True)
    parser.add_argument("--development-block-list", required=True)
    parser.add_argument("--targets-jsonl", required=True)
    parser.add_argument("--budget-protocol", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--split-assignment", required=True)
    parser.add_argument("--code-gate-receipt", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--calibration-output")
    mode.add_argument("--evidence-output")
    parser.add_argument("--numeric-calibration")
    parser.add_argument("--scientific-protocol")
    parser.add_argument("--safety-multiplier", type=float, default=2.0)
    parser.add_argument("--absolute-floor", type=float, default=1.0e-7)
    args = parser.parse_args(argv)
    payload = run_runtime_gate(
        expected_commit=args.expected_commit,
        expected_branch=args.expected_branch,
        selected_actionformer_config=args.selected_actionformer_config,
        standard_actionformer_config=args.standard_actionformer_config,
        actionformer_checkpoint=args.actionformer_checkpoint,
        selected_tridet_config=args.selected_tridet_config,
        standard_tridet_config=args.standard_tridet_config,
        tridet_checkpoint=args.tridet_checkpoint,
        train_block_list=args.train_block_list,
        development_block_list=args.development_block_list,
        targets_jsonl=args.targets_jsonl,
        budget_protocol=args.budget_protocol,
        data_manifest=args.data_manifest,
        split_assignment=args.split_assignment,
        code_gate_receipt=args.code_gate_receipt,
        calibration_output=args.calibration_output,
        numeric_calibration=args.numeric_calibration,
        scientific_protocol=args.scientific_protocol,
        evidence_output=args.evidence_output,
        safety_multiplier=args.safety_multiplier,
        absolute_floor=args.absolute_floor,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
