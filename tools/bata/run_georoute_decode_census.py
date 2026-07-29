#!/usr/bin/env python3
"""Run a full exact-index development decode census for GeoRoute."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.datasets import build_dataset  # noqa: E402


CENSUS_SCHEMA = "georoute_exact_index_decode_census_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip() or f"git {' '.join(args)} failed"
        )
    return completed.stdout.strip()


def _require_source(expected_commit: str) -> str:
    actual = _git_output("rev-parse", "HEAD").lower()
    if actual != expected_commit.lower():
        raise RuntimeError(
            "decode-census source commit differs from --expected-commit"
        )
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("decode census requires a clean source snapshot")
    return actual


def _window_descriptor(
    index: int, row: list[Any] | tuple[Any, ...]
) -> dict[str, Any]:
    if not isinstance(row, (list, tuple)) or len(row) < 4:
        raise ValueError("decode census data row must contain at least four fields")
    centers = row[3]
    if not hasattr(centers, "__len__") or len(centers) == 0:
        raise ValueError("decode census window-center list must be non-empty")
    descriptor = {
        "dataset_index": int(index),
        "video_id": str(row[0]),
        "window_center_count": int(len(centers)),
        "window_center_first": float(centers[0]),
        "window_center_last": float(centers[-1]),
    }
    descriptor["descriptor_sha256"] = _canonical_sha256(descriptor)
    return descriptor


def _validate_sample(
    sample: Mapping[str, Any],
    *,
    expected_source_hw: tuple[int, int] = (180, 320),
    expected_scout_hw: tuple[int, int] = (96, 96),
    expected_frames: int = 768,
) -> dict[str, Any]:
    if not isinstance(sample, Mapping):
        raise TypeError("decoded dataset item must be a mapping")
    inputs = sample.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "source",
        "scout",
    }:
        raise ValueError(
            "decoded GeoRoute item must contain exactly source/scout inputs"
        )
    source = inputs["source"]
    scout = inputs["scout"]
    valid_array_types = (np.ndarray, torch.Tensor)
    if not isinstance(source, valid_array_types) or not isinstance(
        scout, valid_array_types
    ):
        raise TypeError("decoded GeoRoute views must be arrays or tensors")
    source_is_uint8 = (
        source.dtype == torch.uint8
        if isinstance(source, torch.Tensor)
        else source.dtype == np.uint8
    )
    scout_is_uint8 = (
        scout.dtype == torch.uint8
        if isinstance(scout, torch.Tensor)
        else scout.dtype == np.uint8
    )
    if not source_is_uint8 or not scout_is_uint8:
        raise TypeError("decoded GeoRoute views must remain uint8")
    if source.ndim != 5 or scout.ndim != 5:
        raise ValueError("decoded per-item GeoRoute views must be five-dimensional")
    if source.shape[:3] != scout.shape[:3]:
        raise ValueError("decoded source/scout clip, channel and time differ")
    if int(source.shape[2]) != int(expected_frames):
        raise ValueError("decoded GeoRoute time dimension differs from window size")
    if tuple(source.shape[-2:]) != tuple(expected_source_hw):
        raise ValueError("decoded source spatial shape differs from frozen support")
    if tuple(scout.shape[-2:]) != tuple(expected_scout_hw):
        raise ValueError("decoded scout spatial shape differs from frozen support")
    return {
        "source_shape": [int(value) for value in source.shape],
        "scout_shape": [int(value) for value in scout.shape],
        "source_dtype": str(source.dtype),
        "scout_dtype": str(scout.dtype),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bound-config", type=Path, required=True)
    parser.add_argument("--expected-bound-config-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--source-experiment-commit", required=True)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GeoRoute decode census must run inside Slurm")
    if int(args.passes) < 1:
        raise ValueError("decode census requires at least one complete pass")
    runtime_commit = _require_source(args.expected_commit)
    config_path = args.bound_config.resolve()
    output_path = args.output.resolve()
    write_boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if (
        not _inside(config_path, write_boundary)
        or not _inside(output_path, write_boundary)
        or output_path == write_boundary
    ):
        raise ValueError("decode census artifact leaves remote write boundary")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config_sha256 = _sha256_file(config_path)
    if config_sha256 != args.expected_bound_config_sha256.lower():
        raise RuntimeError("decode-census bound-config hash mismatch")

    cfg = Config.fromfile(str(config_path))
    protocol = cfg.get("georoute_protocol", {})
    if (
        bool(protocol.get("official_test_open_allowed", True))
        or bool(protocol.get("gt_for_route_allowed", True))
        or str(cfg.dataset.test.get("subset_name", "")) != "training"
        or not bool(cfg.dataset.test.get("test_mode", False))
    ):
        raise RuntimeError(
            "decode census requires the no-GT development test population"
        )
    dataset = build_dataset(cfg.dataset.test)
    descriptors = [
        _window_descriptor(index, row)
        for index, row in enumerate(dataset.data_list)
    ]
    if not descriptors:
        raise RuntimeError("decode census population is empty")
    population_sha256 = _canonical_sha256({"windows": descriptors})
    custom = cfg.model.backbone.custom
    expected_frames = int(cfg.get("window_size", 768))
    expected_scout_size = int(custom.get("georoute_scout_size", 96))
    failures = []
    successful_items = 0
    first_sample_schema = None
    for pass_index in range(int(args.passes)):
        for dataset_index, descriptor in enumerate(descriptors):
            try:
                sample_schema = _validate_sample(
                    dataset[dataset_index],
                    expected_source_hw=(180, 320),
                    expected_scout_hw=(
                        expected_scout_size,
                        expected_scout_size,
                    ),
                    expected_frames=expected_frames,
                )
                if first_sample_schema is None:
                    first_sample_schema = sample_schema
                elif sample_schema != first_sample_schema:
                    raise ValueError(
                        "decoded GeoRoute sample shape/dtype changed inside census"
                    )
                successful_items += 1
            except Exception as error:  # preserve the complete census
                trace = traceback.format_exc()
                failures.append(
                    {
                        "pass_index": int(pass_index),
                        **descriptor,
                        "exception_type": type(error).__name__,
                        "exception_message": str(error)[:1000],
                        "traceback_sha256": hashlib.sha256(
                            trace.encode("utf-8", errors="replace")
                        ).hexdigest(),
                    }
                )

    expected_items = int(args.passes) * len(descriptors)
    passed = not failures and successful_items == expected_items
    receipt: dict[str, Any] = {
        "schema_version": CENSUS_SCHEMA,
        "status": "PASS_DECODE_CENSUS" if passed else "FAIL_DATA_DECODE",
        "runtime_commit": runtime_commit,
        "source_experiment_commit": args.source_experiment_commit.lower(),
        "slurm_job_id": str(os.environ["SLURM_JOB_ID"]),
        "bound_config_path": str(config_path),
        "bound_config_sha256": config_sha256,
        "passes": int(args.passes),
        "dataset_count": len(descriptors),
        "expected_item_retrievals": expected_items,
        "successful_item_retrievals": successful_items,
        "failure_count": len(failures),
        "population_sha256": population_sha256,
        "first_sample_schema": first_sample_schema,
        "failures": failures,
        "development_only": True,
        "official_test_opened": False,
        "gt_for_route_used": False,
        "teacher_for_route_used": False,
        "oracle_used": False,
        "raw_prediction_cache_used": False,
        "paper_claim_allowed": False,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    _atomic_write_json(output_path, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
