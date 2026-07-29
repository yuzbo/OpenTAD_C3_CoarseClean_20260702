#!/usr/bin/env python3
"""Real-CUDA gate for the official native-grid K384 ActionFormer head."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


EXPECTED_BASE_COMMIT = "61ea7eb9308a568b0cf45e3804830836e30061de"
EXPECTED_BASE_TREE = "7b06c5261ba244788c942a0d73e304581bc35154"
EXPECTED_CHECKPOINT_SHA256 = (
    "003b90506d549c01dd249344d782d45ef5379d977fd54eb0712f4d40640ff8d8"
)
EXPECTED_CHECKPOINT_EPOCH = 34
EXPECTED_CONFIG_RELATIVE = "configs/thumos_i3d_sparsehead_k384_uniform.yaml"
OFFICIAL_CONFIG_RELATIVE = "configs/thumos_i3d.yaml"
OFFICIAL_CONFIG_SHA256 = (
    "c0ac0df560cd564941b56cd9391ad0bd5cea386d2e4b6cf9fc8ffcab821955cd"
)
OFFICIAL_CONFIG_LOADER_SHA256 = (
    "014f1000ac09eb1687d2e6b59bdf9f0afa1dc0a2daed909ee988808929723bc8"
)
OFFICIAL_EFFECTIVE_CONFIG_SHA256 = (
    "835cf30fbcfd27bd6af8885fff002813c8596e2948fce3adf29e3716f316dde4"
)
EXPECTED_QUERY_BUDGET = 384
MINIMUM_ISOLATED_HEAD_SPEEDUP = 1.05
SCHEMA_VERSION = "actionformer_native_grid_sparse_head_cuda_gate_v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_text(repository, *arguments):
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.tmp.{}".format(path.name, os.getpid()))
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def extract_submodule_state(state_dict, prefix):
    alternatives = ("module.{}.".format(prefix), "{}.".format(prefix))
    for candidate in alternatives:
        extracted = {
            key[len(candidate) :]: value
            for key, value in state_dict.items()
            if key.startswith(candidate)
        }
        if extracted:
            return extracted, candidate
    raise ValueError("checkpoint has no state for {}".format(prefix))


def summarize_samples(samples):
    return {
        "samples_ms": samples,
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def measure_cuda_suite(torch, functions, warmup, repeats, rounds):
    names = list(functions)
    aggregate = {name: [] for name in names}
    round_summaries = []
    with torch.inference_mode():
        for round_idx in range(rounds):
            base_order = names[round_idx % len(names) :] + names[
                : round_idx % len(names)
            ]
            if round_idx % 2:
                base_order = list(reversed(base_order))
            for _ in range(warmup):
                for name in base_order:
                    functions[name]()
            torch.cuda.synchronize()
            round_samples = {name: [] for name in names}
            for repeat_idx in range(repeats):
                order = (
                    base_order
                    if repeat_idx % 2 == 0
                    else list(reversed(base_order))
                )
                for name in order:
                    start = time.perf_counter()
                    functions[name]()
                    torch.cuda.synchronize()
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    aggregate[name].append(elapsed_ms)
                    round_samples[name].append(elapsed_ms)
            round_summaries.append(
                {
                    "round_index": round_idx,
                    "execution_order": base_order,
                    "measurements": {
                        name: summarize_samples(round_samples[name])
                        for name in names
                    },
                }
            )
    return {
        "interleaved": True,
        "round_count": rounds,
        "repeats_per_round": repeats,
        "measurements": {
            name: summarize_samples(aggregate[name]) for name in names
        },
        "rounds": round_summaries,
    }


def parse_time_sizes(value):
    sizes = [int(item) for item in value.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("time sizes must be positive integers")
    return sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--budget", type=int, default=384)
    parser.add_argument(
        "--time-sizes",
        type=parse_time_sizes,
        default=parse_time_sizes("2304,1152,576,288,144,72"),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--timing-rounds", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    try:
        config_relative = config_path.relative_to(source_root).as_posix()
    except ValueError as error:
        raise ValueError("candidate config must live inside candidate source") from error
    if config_relative != EXPECTED_CONFIG_RELATIVE:
        raise ValueError("candidate config path is not the sealed primary control")
    if args.budget != EXPECTED_QUERY_BUDGET:
        raise ValueError("native-grid gate is sealed to query budget K=384")
    if args.expected_commit == EXPECTED_BASE_COMMIT:
        raise ValueError("candidate commit must differ from the official base")
    observed_commit = git_text(source_root, "rev-parse", "HEAD")
    observed_tree = git_text(source_root, "rev-parse", "HEAD^{tree}")
    if observed_commit != args.expected_commit:
        raise ValueError("candidate source commit mismatch")
    if observed_tree != args.expected_tree:
        raise ValueError("candidate source tree mismatch")
    if git_text(source_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("candidate source worktree is not clean")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("official ActionFormer checkpoint SHA-256 mismatch")

    sys.path.insert(0, str(source_root))
    import torch

    from libs.core import load_config
    from libs.modeling.meta_archs import (
        PtTransformerClsHead,
        PtTransformerRegHead,
    )
    from libs.modeling.sparse_heads import (
        NativeGridSparseQuerySelector,
        build_sparse_head_execution_receipt,
        run_sparse_cls_head,
        run_sparse_reg_head,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("real-CUDA gate requires an allocated CUDA device")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("real-CUDA gate requires exactly one visible GPU")
    if (
        args.warmup <= 0
        or args.repeats < 5
        or args.timing_rounds < 3
    ):
        raise ValueError("invalid timing repetition counts")
    if args.atol <= 0.0:
        raise ValueError("atol must be positive")

    torch.manual_seed(1234567891)
    torch.cuda.manual_seed_all(1234567891)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")

    official_config_path = source_root / OFFICIAL_CONFIG_RELATIVE
    config_loader_path = source_root / "libs" / "core" / "config.py"
    if sha256_file(official_config_path) != OFFICIAL_CONFIG_SHA256:
        raise ValueError("candidate snapshot changes the pinned official config blob")
    if sha256_file(config_loader_path) != OFFICIAL_CONFIG_LOADER_SHA256:
        raise ValueError("candidate snapshot changes the pinned config loader blob")
    cfg = load_config(str(config_path))
    official_cfg = load_config(str(official_config_path))
    if canonical_sha256(official_cfg) != OFFICIAL_EFFECTIVE_CONFIG_SHA256:
        raise ValueError("pinned official effective-config identity mismatch")
    sparse_cfg = cfg["model"].get("sparse_head")
    expected_sparse_cfg = {
        "enabled": True,
        "budget": args.budget,
        "policy": "stratified_uniform",
        "hash_seed": 1234567891,
        "training_loss_support": "selected_native_grid_queries",
    }
    if sparse_cfg != expected_sparse_cfg:
        raise ValueError("effective sparse-head config mismatch")
    protected_cfg = copy.deepcopy(cfg)
    protected_cfg["model"].pop("sparse_head")
    if protected_cfg != official_cfg:
        raise ValueError(
            "candidate changes protected official effective-config fields"
        )
    if len(args.time_sizes) != len(cfg["model"]["regression_range"]):
        raise ValueError("time-size count differs from official FPN levels")
    expected_time_sizes = [
        cfg["model"]["max_seq_len"] // (2 ** level_idx)
        for level_idx in range(len(cfg["model"]["regression_range"]))
    ]
    if args.time_sizes != expected_time_sizes:
        raise ValueError("synthetic FPN sizes differ from the official training grid")

    cls_head = PtTransformerClsHead(
        cfg["model"]["fpn_dim"],
        cfg["model"]["head_dim"],
        cfg["model"]["num_classes"],
        prior_prob=cfg["train_cfg"]["cls_prior_prob"],
        num_layers=cfg["model"]["head_num_layers"],
        kernel_size=cfg["model"]["head_kernel_size"],
        with_ln=cfg["model"]["head_with_ln"],
        empty_cls=cfg["train_cfg"]["head_empty_cls"],
    )
    reg_head = PtTransformerRegHead(
        cfg["model"]["fpn_dim"],
        cfg["model"]["head_dim"],
        len(cfg["model"]["regression_range"]),
        num_layers=cfg["model"]["head_num_layers"],
        kernel_size=cfg["model"]["head_kernel_size"],
        with_ln=cfg["model"]["head_with_ln"],
    )
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    checkpoint_epoch = int(checkpoint["epoch"])
    if checkpoint_epoch != EXPECTED_CHECKPOINT_EPOCH:
        raise ValueError("official ActionFormer checkpoint epoch mismatch")
    state_dict = checkpoint["state_dict_ema"]
    cls_state, cls_prefix = extract_submodule_state(state_dict, "cls_head")
    reg_state, reg_prefix = extract_submodule_state(state_dict, "reg_head")
    cls_head.load_state_dict(cls_state, strict=True)
    reg_head.load_state_dict(reg_state, strict=True)
    del checkpoint, state_dict
    cls_head.to(device).eval()
    reg_head.to(device).eval()

    fpn_feats = tuple(
        torch.randn(
            1,
            cfg["model"]["fpn_dim"],
            time_size,
            device=device,
            dtype=torch.float32,
        )
        for time_size in args.time_sizes
    )
    fpn_masks = tuple(
        torch.ones(
            1, 1, time_size, device=device, dtype=torch.bool
        )
        for time_size in args.time_sizes
    )
    immutable_feats = tuple(feat.clone() for feat in fpn_feats)
    immutable_masks = tuple(mask.clone() for mask in fpn_masks)
    selector = NativeGridSparseQuerySelector(
        args.budget,
        policy="stratified_uniform",
        hash_seed=1234567891,
    ).to(device)
    selected_masks = selector(fpn_masks, None)
    immutable_selected_masks = tuple(mask.clone() for mask in selected_masks)
    selected_masks_repeat = selector(fpn_masks, None)
    selector_deterministic = all(
        torch.equal(first, second)
        for first, second in zip(selected_masks, selected_masks_repeat)
    )

    with torch.inference_mode():
        dense_cls = cls_head(fpn_feats, fpn_masks)
        dense_reg = reg_head(fpn_feats, fpn_masks)
        sparse_cls = run_sparse_cls_head(
            cls_head, fpn_feats, fpn_masks, selected_masks
        )
        sparse_reg = run_sparse_reg_head(
            reg_head, fpn_feats, fpn_masks, selected_masks
        )
    maximum_abs_error = 0.0
    unselected_nonzero = 0
    for dense, sparse, selected in zip(
        dense_cls + dense_reg,
        sparse_cls + sparse_reg,
        selected_masks + selected_masks,
    ):
        expanded = selected.expand_as(dense)
        if expanded.any():
            maximum_abs_error = max(
                maximum_abs_error,
                float((dense[expanded] - sparse[expanded]).abs().max().item()),
            )
        unselected_nonzero += int(
            torch.count_nonzero(sparse[~expanded]).item()
        )
    raw_tensors_immutable = all(
        torch.equal(before, after)
        for before, after in zip(immutable_feats, fpn_feats)
    ) and all(
        torch.equal(before, after)
        for before, after in zip(immutable_masks, fpn_masks)
    )
    selected_masks_immutable = all(
        torch.equal(before, after)
        for before, after in zip(immutable_selected_masks, selected_masks)
    )
    all_valid_maximum_abs_error = maximum_abs_error

    boundary_masks = tuple(
        (
            torch.arange(time_size, device=device)[None, None, :]
            < max(1, time_size - 3 * (level_idx + 1))
        )
        for level_idx, time_size in enumerate(args.time_sizes)
    )
    boundary_immutable_feats = tuple(feat.clone() for feat in fpn_feats)
    boundary_immutable_masks = tuple(mask.clone() for mask in boundary_masks)
    boundary_selected_masks = selector(boundary_masks, None)
    with torch.inference_mode():
        boundary_dense_cls = cls_head(fpn_feats, boundary_masks)
        boundary_dense_reg = reg_head(fpn_feats, boundary_masks)
        boundary_sparse_cls = run_sparse_cls_head(
            cls_head, fpn_feats, boundary_masks, boundary_selected_masks
        )
        boundary_sparse_reg = run_sparse_reg_head(
            reg_head, fpn_feats, boundary_masks, boundary_selected_masks
        )
    boundary_maximum_abs_error = 0.0
    boundary_unselected_nonzero = 0
    for dense, sparse, selected in zip(
        boundary_dense_cls + boundary_dense_reg,
        boundary_sparse_cls + boundary_sparse_reg,
        boundary_selected_masks + boundary_selected_masks,
    ):
        expanded = selected.expand_as(dense)
        if expanded.any():
            boundary_maximum_abs_error = max(
                boundary_maximum_abs_error,
                float((dense[expanded] - sparse[expanded]).abs().max().item()),
            )
        boundary_unselected_nonzero += int(
            torch.count_nonzero(sparse[~expanded]).item()
        )
    boundary_raw_tensors_immutable = all(
        torch.equal(before, after)
        for before, after in zip(boundary_immutable_feats, fpn_feats)
    ) and all(
        torch.equal(before, after)
        for before, after in zip(boundary_immutable_masks, boundary_masks)
    )
    maximum_abs_error = max(maximum_abs_error, boundary_maximum_abs_error)
    unselected_nonzero += boundary_unselected_nonzero
    raw_tensors_immutable = (
        raw_tensors_immutable and boundary_raw_tensors_immutable
    )

    execution = build_sparse_head_execution_receipt(
        cls_head,
        reg_head,
        fpn_masks,
        selected_masks,
        budget=args.budget,
        policy="stratified_uniform",
        training_loss_support="selected_native_grid_queries",
    )

    def dense_function():
        cls_head(fpn_feats, fpn_masks)
        reg_head(fpn_feats, fpn_masks)

    def sparse_preselected_function():
        run_sparse_cls_head(cls_head, fpn_feats, fpn_masks, selected_masks)
        run_sparse_reg_head(reg_head, fpn_feats, fpn_masks, selected_masks)

    def sparse_with_selector_function():
        live_selected = selector(fpn_masks, None)
        run_sparse_cls_head(cls_head, fpn_feats, fpn_masks, live_selected)
        run_sparse_reg_head(reg_head, fpn_feats, fpn_masks, live_selected)

    timings = measure_cuda_suite(
        torch,
        {
            "dense_heads": dense_function,
            "sparse_heads_preselected": sparse_preselected_function,
            "sparse_heads_with_selector": sparse_with_selector_function,
        },
        args.warmup,
        args.repeats,
        args.timing_rounds,
    )
    measurements = timings["measurements"]
    dense_mean = measurements["dense_heads"]["mean_ms"]
    sparse_mean = measurements["sparse_heads_with_selector"]["mean_ms"]
    dense_median = measurements["dense_heads"]["median_ms"]
    sparse_median = measurements["sparse_heads_with_selector"]["median_ms"]
    wall_clock_speedup = dense_median / sparse_median
    numerical_equivalence = maximum_abs_error <= args.atol
    mac_reduction = (
        execution["combined_sparse_macs"]
        < execution["combined_dense_macs"]
    )
    per_round_speedups = [
        (
            item["measurements"]["dense_heads"]["median_ms"]
            / item["measurements"]["sparse_heads_with_selector"]["median_ms"]
        )
        for item in timings["rounds"]
    ]
    wall_clock_reduction = (
        sparse_mean < dense_mean
        and wall_clock_speedup >= MINIMUM_ISOLATED_HEAD_SPEEDUP
        and all(
            speedup >= MINIMUM_ISOLATED_HEAD_SPEEDUP
            for speedup in per_round_speedups
        )
    )
    selected_count_pass = execution["selected_count_contract_pass"]
    gate_pass = all(
        (
            numerical_equivalence,
            unselected_nonzero == 0,
            raw_tensors_immutable,
            selected_masks_immutable,
            selector_deterministic,
            mac_reduction,
            wall_clock_reduction,
            selected_count_pass,
        )
    )
    execution["wall_clock_claim_allowed"] = False
    execution["isolated_head_path_wall_clock_claim_allowed"] = (
        wall_clock_reduction
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "validation_pass": gate_pass,
        "gate_pass": gate_pass,
        "candidate": {
            "repository_root": str(source_root),
            "commit": observed_commit,
            "tree": observed_tree,
            "config_path": str(config_path),
            "config_relative_path": config_relative,
            "config_sha256": sha256_file(config_path),
            "effective_config_sha256": canonical_sha256(cfg),
            "protected_effective_config_sha256": canonical_sha256(
                protected_cfg
            ),
            "official_effective_config_sha256": canonical_sha256(
                official_cfg
            ),
            "pinned_official_config_sha256": OFFICIAL_CONFIG_SHA256,
            "pinned_official_config_loader_sha256": (
                OFFICIAL_CONFIG_LOADER_SHA256
            ),
            "protected_effective_config_match": True,
            "clean": True,
        },
        "official_base": {
            "commit": EXPECTED_BASE_COMMIT,
            "tree": EXPECTED_BASE_TREE,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_epoch": checkpoint_epoch,
            "cls_state_prefix": cls_prefix,
            "reg_state_prefix": reg_prefix,
        },
        "cuda": {
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "tf32_enabled": False,
            "deterministic_cudnn": True,
        },
        "synthetic_fpn_contract": {
            "seed": 1234567891,
            "batch_size": 1,
            "feature_dim": cfg["model"]["fpn_dim"],
            "time_sizes": args.time_sizes,
            "timing_masks_all_valid": True,
            "equivalence_includes_prefix_invalid_masks": True,
        },
        "equivalence": {
            "atol": args.atol,
            "maximum_abs_error": maximum_abs_error,
            "selected_outputs_equivalent": numerical_equivalence,
            "unselected_nonzero_count": unselected_nonzero,
            "raw_tensors_immutable": raw_tensors_immutable,
            "selected_masks_immutable": selected_masks_immutable,
            "selector_deterministic": selector_deterministic,
            "all_valid_maximum_abs_error": all_valid_maximum_abs_error,
            "prefix_invalid_maximum_abs_error": boundary_maximum_abs_error,
            "prefix_invalid_unselected_nonzero_count": (
                boundary_unselected_nonzero
            ),
            "prefix_invalid_raw_tensors_immutable": (
                boundary_raw_tensors_immutable
            ),
        },
        "execution": execution,
        "timings": timings,
        "wall_clock_speedup": wall_clock_speedup,
        "per_round_wall_clock_speedups": per_round_speedups,
        "minimum_isolated_head_speedup": MINIMUM_ISOLATED_HEAD_SPEEDUP,
        "theoretical_head_mac_reduction": mac_reduction,
        "isolated_head_path_wall_clock_reduction": wall_clock_reduction,
        "claim_boundary": (
            "real_cuda_equivalence_and_cost_gate"
            if gate_pass
            else "engineering_failure_no_model_metric"
        ),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
