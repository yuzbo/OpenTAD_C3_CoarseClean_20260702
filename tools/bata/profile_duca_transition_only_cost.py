from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from mmengine.config import Config

DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "duca_transition_only_fixed384_official_adatad_backend_full_train.py"
)
SCHEMA_VERSION = "duca-transition-only-cost-v1"


def build_selector(cfg: Mapping[str, Any]) -> Any:
    """Lazy-load OpenTAD so schema/unit tests do not require detector dependencies."""

    from opentad.models import selectors as _registered_selectors  # noqa: F401
    from opentad.models.builder import build_selector as registry_build_selector

    return registry_build_selector(cfg)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items() if str(key) != "_delete_"}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return copy.deepcopy(value)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance(root: Path) -> dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "repository_root": str(root.resolve()),
        "commit": commit,
        "working_tree_dirty": None if status is None else bool(status),
    }


def _config_provenance(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"transition-only config does not exist: {resolved}")
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _parameter_counts(module: Any) -> dict[str, int]:
    if module is None or not hasattr(module, "parameters"):
        return {"total": 0, "trainable": 0}
    parameters = list(module.parameters())
    return {
        "total": int(sum(int(param.numel()) for param in parameters)),
        "trainable": int(sum(int(param.numel()) for param in parameters if param.requires_grad)),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("latency samples cannot be empty")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(percentile)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _temporal_dim(inputs: torch.Tensor) -> int:
    if inputs.ndim == 5:
        return int(inputs.shape[2])
    if inputs.ndim == 6:
        return int(inputs.shape[3])
    raise ValueError("raw-video inputs must be [B,C,T,H,W] or [B,N,C,T,H,W]")


def _selection_geometry(
    selected_positions: torch.Tensor,
    masks: torch.Tensor,
) -> dict[str, Any]:
    if selected_positions.ndim != 2:
        raise ValueError("structured selector positions must be [B,K]")
    if masks.ndim != 2 or masks.shape[0] != selected_positions.shape[0]:
        raise ValueError("valid masks must be [B,T] and align with selected positions")

    counts: list[int] = []
    max_holes: list[int] = []
    for batch_idx in range(int(selected_positions.shape[0])):
        valid_len = int(masks[batch_idx].long().sum().item())
        positions = selected_positions[batch_idx]
        positions = positions[(positions >= 0) & (positions < valid_len)].detach().cpu().long()
        if int(positions.numel()) == 0:
            counts.append(0)
            max_holes.append(valid_len)
            continue
        positions = torch.unique(positions, sorted=True)
        values = [int(value) for value in positions.tolist()]
        holes = [values[0]]
        holes.extend(right - left - 1 for left, right in zip(values[:-1], values[1:]))
        holes.append(valid_len - values[-1] - 1)
        counts.append(len(values))
        max_holes.append(max(holes))
    if len(set(counts)) != 1:
        raise RuntimeError(f"fixed-budget selector returned inconsistent selected counts: {counts}")
    return {
        "selected_count": counts[0],
        "selected_count_per_sample": counts,
        "max_gap": max(max_holes),
        "max_gap_per_sample": max_holes,
        "max_gap_definition": "maximum_unselected_hole_including_prefix_and_suffix",
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cost report contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.ndim == 0:
            return _json_safe(value.detach().cpu().item())
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def build_transition_only_selector(
    config_path: str | Path,
    *,
    temporal_len: Optional[int] = None,
    budget: Optional[int] = None,
    probe_spatial_size: Optional[int] = None,
) -> tuple[Any, dict[str, Any]]:
    """Build the real frame selector while keeping smoke scaling explicit."""

    cfg = Config.fromfile(str(config_path))
    contract = cfg.duca_transition_only_contract
    if _field(contract, "selector_variant") != "transition_only":
        raise ValueError("config contract must declare selector_variant='transition_only'")
    if _field(contract, "coarse_hidden_kind") != "official_asformer_encoder_hidden":
        raise ValueError("config must consume official ASFormer encoder hidden state")

    selector_cfg = _plain(cfg.model.frame_selector)
    if selector_cfg.get("selector_variant") != "transition_only":
        raise ValueError("frame_selector must be the transition_only variant")
    source_cfg = dict(selector_cfg.get("actionness_source_cfg") or {})
    if source_cfg.get("probe_model") != "official-action-seg":
        raise ValueError("transition-only cost audit requires the official action-segmentation probe")
    if source_cfg.get("official_action_seg_backend") != "official_asformer":
        raise ValueError("transition-only cost audit requires official_asformer")

    configured_temporal_len = int(selector_cfg["dense_window_size"])
    configured_budget = int(selector_cfg["budget"])
    configured_spatial_size = int(source_cfg.get("spatial_size", 64))
    profile_temporal_len = configured_temporal_len if temporal_len is None else int(temporal_len)
    profile_budget = configured_budget if budget is None else int(budget)
    profile_spatial_size = int(
        configured_spatial_size if probe_spatial_size is None else probe_spatial_size
    )
    if profile_temporal_len <= 0:
        raise ValueError("temporal_len must be positive")
    if profile_budget <= 0:
        raise ValueError("budget must be positive")
    if profile_budget > profile_temporal_len:
        raise ValueError("budget cannot exceed temporal_len")
    if profile_spatial_size <= 0:
        raise ValueError("probe_spatial_size must be positive")

    selector_cfg.update(
        {
            "dense_window_size": profile_temporal_len,
            "budget": profile_budget,
            "profile_runtime": False,
            "profile_sync_cuda": False,
        }
    )
    source_cfg["spatial_size"] = profile_spatial_size
    selector_cfg["actionness_source_cfg"] = source_cfg
    selector = build_selector(selector_cfg)
    metadata = {
        "configured_temporal_len": configured_temporal_len,
        "configured_budget": configured_budget,
        "configured_probe_spatial_size": configured_spatial_size,
        "profile_temporal_len": profile_temporal_len,
        "profile_budget": profile_budget,
        "profile_probe_spatial_size": profile_spatial_size,
        "max_unselected_hole": int(selector_cfg["max_unselected_hole"]),
        "scaled_smoke": False,
    }
    # Compare against the original source value after the build dictionary is finalized.
    metadata["scaled_smoke"] = bool(
        profile_temporal_len != configured_temporal_len
        or profile_budget != configured_budget
        or profile_spatial_size != configured_spatial_size
    )
    return selector, metadata


def profile_selector(
    selector: Any,
    *,
    inputs: torch.Tensor,
    masks: torch.Tensor,
    warmup: int,
    repeats: int,
    device: torch.device,
    config_provenance: Mapping[str, Any],
    build_metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Measure the offline full-window pre-backbone selector and nothing downstream."""

    warmup = int(warmup)
    repeats = int(repeats)
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requested but torch.cuda.is_available() is false")
    temporal_len = _temporal_dim(inputs)
    if masks.shape != (int(inputs.shape[0]), temporal_len):
        raise ValueError("masks must be [B,T] and align with raw-video inputs")

    selector = selector.to(device)
    selector.eval()
    inputs = inputs.to(device)
    masks = masks.to(device=device, dtype=torch.bool)
    metas = [{"video_name": f"duca_transition_only_cost_{idx:03d}"} for idx in range(int(inputs.shape[0]))]

    def run_once() -> Mapping[str, Any]:
        output = selector.forward_test(inputs=inputs, masks=masks, metas=metas)
        if not isinstance(output, Mapping):
            raise RuntimeError("frame selector forward_test must return a mapping")
        return output

    with torch.inference_mode():
        for _ in range(warmup):
            run_once()
        _sync(device)

        if device.type == "cuda":
            baseline_allocated = int(torch.cuda.memory_allocated(device))
            baseline_reserved = int(torch.cuda.memory_reserved(device))
            torch.cuda.reset_peak_memory_stats(device)
        else:
            baseline_allocated = baseline_reserved = 0

        samples_ms: list[float] = []
        last_output: Optional[Mapping[str, Any]] = None
        for _ in range(repeats):
            _sync(device)
            start_ns = time.perf_counter_ns()
            last_output = run_once()
            _sync(device)
            samples_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000.0)

    if last_output is None:
        raise RuntimeError("profiler did not execute the selector")
    selector_outputs = last_output.get("selector_outputs")
    if not isinstance(selector_outputs, Mapping):
        raise RuntimeError("selector output is missing selector_outputs")
    compute_profile = selector_outputs.get("compute_profile")
    if not isinstance(compute_profile, Mapping):
        raise RuntimeError("selector output is missing its static compute_profile")
    if compute_profile.get("estimated_macs") is None or compute_profile.get("estimated_flops") is None:
        raise RuntimeError("static compute profile must report estimated MACs and FLOPs")
    grid = selector_outputs.get("grid")
    selected_positions = getattr(grid, "selected_positions", None)
    if selected_positions is None:
        raise RuntimeError("structured selector output is missing grid.selected_positions")

    geometry = _selection_geometry(selected_positions, masks)
    actionness_provenance = selector_outputs.get("provenance")
    if not isinstance(actionness_provenance, Mapping):
        raise RuntimeError("selector output is missing coarse-probe provenance")
    scorer = getattr(getattr(selector, "adapter", None), "transition_scorer", None)
    coarse_probe = getattr(selector, "raw_actionness_source", None)

    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        cuda_memory = {
            "available": True,
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "incremental_peak_allocated_bytes": max(0, peak_allocated - baseline_allocated),
            "note": "PyTorch allocator memory for inputs, selector parameters, and selector intermediates; detector excluded",
        }
    else:
        cuda_memory = {
            "available": False,
            "baseline_allocated_bytes": None,
            "baseline_reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "incremental_peak_allocated_bytes": None,
        }

    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor() or platform.machine() or "cpu"
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "task": "offline_temporal_action_detection",
        "online_tad": False,
        "timing_mode": "offline_full_window_inference",
        "accounting_scope": {
            "name": "selector-only/pre-backbone-only",
            "selector_only": True,
            "pre_backbone_only": True,
            "includes": [
                "raw-video tensor input",
                "low-resolution frame preparation",
                "official ASFormer coarse probe",
                "transition descriptor and shared scorer",
                "exact-K/max-gap structured selection",
                "selected-frame gather",
            ],
            "excludes": [
                "video decode and storage I/O",
                "detector backbone",
                "detector projection and neck",
                "detector head and losses",
                "post-processing and NMS",
            ],
            "detector_backbone_included": False,
            "detector_head_included": False,
            "detector_cost_estimated": False,
            "full_stack_cost_claim_allowed": False,
        },
        "input_shape": [int(value) for value in inputs.shape],
        "input_dtype": str(inputs.dtype),
        **geometry,
        "static_estimate": {
            "estimated_macs": int(compute_profile["estimated_macs"]),
            "estimated_flops": int(compute_profile["estimated_flops"]),
            "is_lower_bound": bool(compute_profile.get("estimated_flops_are_lower_bound", False)),
            "complete_memory_accounting": bool(compute_profile.get("complete_memory_accounting", False)),
            "source": "selector_forward_compute_profile",
            "components": compute_profile.get("components", {}),
            "note": "Static estimates are implementation-provided analytic estimates; measured latency is authoritative for uncovered operators and DP control flow.",
        },
        "latency_ms": {
            "path": "DucaOnlineFrameSelector.forward_test",
            "warmup_count": warmup,
            "sample_count": repeats,
            "median": float(statistics.median(samples_ms)),
            "p90": float(_percentile(samples_ms, 0.90)),
            "min": float(min(samples_ms)),
            "max": float(max(samples_ms)),
            "samples": samples_ms,
            "cuda_synchronized": device.type == "cuda",
        },
        "cuda_peak_memory": cuda_memory,
        "parameters": {
            "selector_total": _parameter_counts(selector),
            "coarse_probe": _parameter_counts(coarse_probe),
            "transition_scorer": _parameter_counts(scorer),
        },
        "provenance": {
            "repository": _git_provenance(ROOT),
            "config": dict(config_provenance),
            "build": dict(build_metadata or {}),
            "selector_class": f"{selector.__class__.__module__}.{selector.__class__.__qualname__}",
            "coarse_probe": dict(actionness_provenance),
            "selection_path": selector_outputs.get("selection_path"),
            "selector_variant": selector_outputs.get("selector_variant", "transition_only"),
        },
        "runtime": {
            "device": str(device),
            "device_name": device_name,
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    return _json_safe(report)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _make_inputs(
    *,
    batch_size: int,
    temporal_len: int,
    height: int,
    width: int,
    dtype: str,
    device: torch.device,
) -> torch.Tensor:
    shape = (int(batch_size), 3, int(temporal_len), int(height), int(width))
    if dtype == "uint8":
        return torch.randint(0, 256, shape, dtype=torch.uint8, device=device)
    if dtype == "float32":
        return torch.rand(shape, dtype=torch.float32, device=device) * 255.0
    raise ValueError(f"unsupported input dtype: {dtype}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile DUCA transition-only pre-backbone selector cost without detector cost"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--official-repos-root", default=os.environ.get("C3_OFFICIAL_ACTION_SEG_REPOS"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--temporal-len", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--probe-spatial-size", type=int, default=None)
    parser.add_argument("--input-dtype", choices=("float32", "uint8"), default="float32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--output", default="", help="optional JSON output path; JSON is always printed")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if int(args.batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(args.height) <= 0 or int(args.width) <= 0:
        raise ValueError("height and width must be positive")
    if args.official_repos_root:
        official_root = Path(args.official_repos_root).expanduser().resolve()
        if not (official_root / "ASFormer" / "model.py").is_file():
            raise FileNotFoundError(f"official ASFormer source is missing under {official_root}")
        os.environ["C3_OFFICIAL_ACTION_SEG_REPOS"] = str(official_root)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    selector, build_metadata = build_transition_only_selector(
        config_path,
        temporal_len=args.temporal_len,
        budget=args.budget,
        probe_spatial_size=args.probe_spatial_size,
    )
    temporal_len = int(build_metadata["profile_temporal_len"])
    device = _resolve_device(str(args.device))
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    inputs = _make_inputs(
        batch_size=int(args.batch_size),
        temporal_len=temporal_len,
        height=int(args.height),
        width=int(args.width),
        dtype=str(args.input_dtype),
        device=device,
    )
    masks = torch.ones(int(args.batch_size), temporal_len, dtype=torch.bool, device=device)
    report = profile_selector(
        selector,
        inputs=inputs,
        masks=masks,
        warmup=int(args.warmup),
        repeats=int(args.repeats),
        device=device,
        config_provenance=_config_provenance(config_path),
        build_metadata=build_metadata,
    )
    payload = json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
