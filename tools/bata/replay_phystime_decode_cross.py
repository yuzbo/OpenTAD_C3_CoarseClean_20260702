#!/usr/bin/env python3
"""Replay one frozen PhysTime checkpoint on uniform and physical time axes."""

import argparse
import copy
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config

from opentad.cores.test_engine import apply_sliding_window_nms
from opentad.evaluations import build_evaluator
from opentad.models.detectors.single_stage import SingleStageDetector


SCHEMA_VERSION = "phystime_decode_replay_inputs_v1"
ARTIFACT_KIND = "frozen_pre_decode_actionformer_tensors"
REQUIRED_ARRAYS = {
    "cls_logits",
    "cls_scores",
    "reg_distances",
    "base_mask",
    "native_mask",
    "native_points",
    "native_proposals",
    "base_points",
    "level_offsets",
    "uniform_axis_sec",
    "physical_axis_sec",
    "native_valid_count",
    "domain_sec",
}
AXIS_SPECS = {
    "uniform_rank_seconds": "uniform_axis_sec",
    "physical_time_seconds": "physical_axis_sec",
}
ARM_NATIVE_AXIS = {
    "selected_axis": "uniform_rank_seconds",
    "physical_metric": "physical_time_seconds",
}
P0_RUNTIME_COMMIT = "c2cfcfa2470f9f1e0b9d10e397480f6c66aeaf2c"
P0_RUNTIME_TREE = "0b78dd402e8997239ef9d1b4b4cd8bfa4f7a6338"
P0_GATE_SHA256 = (
    "1ca0efcdeb9f6343da076a00660675759358ac467074919a34d01c0d7c7250d9"
)
DATASET_MANIFEST_SHA256 = (
    "1da0bca28f14ca2f1e4b2baf0f199dce18f4dd925e0f097a70a3fcc1c13eb1b2"
)
VIDEOMAE_SHA256 = (
    "4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay frozen PhysTime tensors with U/P decode axes."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--direct-pre-cross", required=True)
    parser.add_argument("--direct-result", required=True)
    parser.add_argument("--direct-metrics", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-completion", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--p0-completion", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--arm",
        required=True,
        choices=("selected_axis", "physical_metric"),
    )
    parser.add_argument(
        "--weights-source",
        required=True,
        choices=("online", "ema"),
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--expected-runtime-commit", required=True)
    parser.add_argument("--expected-runtime-tree", required=True)
    parser.add_argument("--evaluation-epoch", type=int, default=59)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_json_gzip(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as out:
        json.dump(
            payload,
            out,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        )
        out.write("\n")
    os.replace(temporary, path)


def atomic_write_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_gzip_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def current_git_identity():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"],
        text=True,
    ).strip()
    return commit, tree


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_and_validate_capture(args):
    manifest = load_json(args.manifest)
    require(
        manifest.get("schema_version") == SCHEMA_VERSION,
        "unsupported decode replay manifest schema",
    )
    require(
        manifest.get("artifact_kind") == ARTIFACT_KIND,
        "decode replay manifest has the wrong artifact kind",
    )
    require(
        manifest.get("runtime", {}).get("commit")
        == args.expected_runtime_commit,
        "capture runtime commit mismatch",
    )
    require(
        manifest.get("runtime", {}).get("git_tree")
        == args.expected_runtime_tree,
        "capture runtime tree mismatch",
    )
    require(
        manifest.get("source", {}).get("commit") == args.source_commit,
        "capture source commit mismatch",
    )
    require(
        manifest.get("source", {}).get("git_tree") == args.source_tree,
        "capture source tree mismatch",
    )
    native_axis = ARM_NATIVE_AXIS[args.arm]
    require(
        manifest.get("train_axis") == native_axis,
        "capture train axis differs from the arm",
    )
    require(
        manifest.get("expected_native_coordinate_mode") == native_axis,
        "capture native axis differs from the arm",
    )
    require(
        manifest.get("weights_source") == args.weights_source,
        "capture weights source mismatch",
    )
    require(
        int(manifest.get("evaluation_epoch", -1))
        == args.evaluation_epoch,
        "capture evaluation epoch mismatch",
    )
    require(
        manifest.get("checkpoint", {}).get("sha256")
        == sha256_file(args.checkpoint),
        "capture checkpoint hash mismatch",
    )
    require(
        Path(manifest["artifact"]["path"]).resolve()
        == Path(args.artifact).resolve(),
        "capture artifact path mismatch",
    )
    require(
        manifest["artifact"]["sha256"] == sha256_file(args.artifact),
        "capture artifact file hash mismatch",
    )

    with np.load(args.artifact, allow_pickle=False) as archive:
        require(
            set(archive.files) == REQUIRED_ARRAYS,
            "decode replay NPZ array set is incomplete or contains extras",
        )
        arrays = {name: archive[name] for name in archive.files}
    require(
        set(manifest.get("array_contract", {})) == REQUIRED_ARRAYS,
        "manifest array contract differs from the NPZ schema",
    )
    for name, array in arrays.items():
        contract = manifest["array_contract"][name]
        require(
            contract.get("dtype") == str(array.dtype),
            f"{name} dtype differs from the manifest",
        )
        require(
            contract.get("shape") == list(array.shape),
            f"{name} shape differs from the manifest",
        )
        require(
            contract.get("canonical_sha256") == array_sha256(array),
            f"{name} canonical hash differs from the manifest",
        )
    validate_array_semantics(arrays, manifest)
    return arrays, manifest


def validate_array_semantics(arrays, manifest):
    logits = arrays["cls_logits"]
    scores = arrays["cls_scores"]
    reg = arrays["reg_distances"]
    base_mask = arrays["base_mask"]
    native_mask = arrays["native_mask"]
    native_points = arrays["native_points"]
    native_proposals = arrays["native_proposals"]
    base_points = arrays["base_points"]
    offsets = arrays["level_offsets"]
    counts = arrays["native_valid_count"]
    domains = arrays["domain_sec"]

    require(logits.dtype == np.float32, "cls_logits must be float32")
    require(scores.dtype == np.float32, "cls_scores must be float32")
    require(reg.dtype == np.float32, "reg_distances must be float32")
    require(base_mask.dtype == np.bool_, "base_mask must be bool")
    require(native_mask.dtype == np.bool_, "native_mask must be bool")
    require(native_points.dtype == np.float32, "native_points must be float32")
    require(
        native_proposals.dtype == np.float32,
        "native_proposals must be float32",
    )
    require(base_points.dtype == np.float32, "base_points must be float32")
    require(offsets.dtype == np.int32, "level_offsets must be int32")
    require(counts.dtype == np.int32, "native_valid_count must be int32")
    require(domains.dtype == np.float64, "domain_sec must be float64")

    require(logits.ndim == 3, "cls_logits must have shape [N,Q,C]")
    n_windows, q_count, class_count = logits.shape
    require(scores.shape == logits.shape, "cls_scores shape mismatch")
    require(reg.shape == (n_windows, q_count, 2), "reg_distances shape mismatch")
    require(base_mask.shape == (n_windows, q_count), "base_mask shape mismatch")
    require(native_mask.shape == base_mask.shape, "native_mask shape mismatch")
    require(
        native_points.shape == (n_windows, q_count, 4),
        "native_points shape mismatch",
    )
    require(
        native_proposals.shape == (n_windows, q_count, 2),
        "native_proposals shape mismatch",
    )
    require(base_points.shape == (q_count, 4), "base_points shape mismatch")
    require(
        offsets.ndim == 1
        and offsets.size >= 2
        and int(offsets[0]) == 0
        and int(offsets[-1]) == q_count
        and np.all(np.diff(offsets) > 0),
        "level_offsets are invalid",
    )
    require(counts.shape == (n_windows,), "native_valid_count shape mismatch")
    require(domains.shape == (n_windows, 2), "domain_sec shape mismatch")
    require(
        manifest.get("window_count") == n_windows
        and manifest.get("candidate_count") == q_count
        and manifest.get("class_count") == class_count,
        "manifest N/Q/C binding mismatch",
    )
    require(
        len(manifest.get("class_map", [])) == class_count,
        "class map count differs from C",
    )
    require(
        len(manifest.get("windows", [])) == n_windows,
        "window manifest count differs from N",
    )
    require(
        np.isfinite(logits).all()
        and np.isfinite(scores).all()
        and np.isfinite(reg).all()
        and np.isfinite(native_points).all()
        and np.isfinite(native_proposals).all()
        and np.isfinite(base_points).all()
        and np.isfinite(domains).all(),
        "captured dense tensors contain non-finite values",
    )
    require(np.all(reg >= 0.0), "regression distances must be non-negative")
    require(np.all(scores >= 0.0) and np.all(scores <= 1.0), "scores outside [0,1]")
    source_dtypes = manifest.get("source_tensor_dtypes")
    require(
        isinstance(source_dtypes, dict)
        and set(source_dtypes)
        >= {"cls_logits", "cls_scores", "reg_distances"},
        "source tensor dtype provenance is missing",
    )
    capture_memory = manifest.get("capture_memory")
    require(
        isinstance(capture_memory, dict)
        and capture_memory.get("within_budget") is True
        and int(capture_memory.get("estimated_peak_tensor_bytes", -1))
        <= int(capture_memory.get("max_in_memory_bytes", -2)),
        "capture memory budget provenance is invalid",
    )
    require(
        np.all(domains[:, 1] > domains[:, 0]),
        "physical domains must have positive duration",
    )
    base_centers = base_points[:, 0]
    for idx in range(n_windows):
        count = int(counts[idx])
        require(count > 0, f"window {idx} valid count is not positive")
        expected_native_mask = base_mask[idx] & (base_centers < float(count))
        require(
            np.array_equal(expected_native_mask, native_mask[idx]),
            f"window {idx} native mask differs from the prefix contract",
        )
        for axis_name in AXIS_SPECS.values():
            axis = arrays[axis_name][idx]
            require(
                axis.ndim == 1 and axis.size >= count,
                f"window {idx} {axis_name} shape is invalid",
            )
            valid = axis[:count]
            require(
                np.isfinite(valid).all() and np.all(np.diff(valid) > 0),
                f"window {idx} {axis_name} valid prefix is invalid",
            )
            require(
                np.isnan(axis[count:]).all(),
                f"window {idx} {axis_name} padding must be NaN",
            )
            require(
                float(valid[0]) >= float(domains[idx, 0]) - 1.0e-6
                and float(valid[-1]) <= float(domains[idx, 1]) + 1.0e-6,
                f"window {idx} {axis_name} lies outside its domain",
            )


def map_selected_axis(coords, positions, domain_start, domain_end):
    count = int(positions.numel())
    rank_centers = torch.arange(
        count,
        dtype=coords.dtype,
        device=coords.device,
    )
    xp = torch.cat(
        (
            rank_centers.new_tensor([-0.5]),
            rank_centers,
            rank_centers.new_tensor([float(count) - 0.5]),
        )
    )
    fp = torch.cat(
        (
            positions.new_tensor([float(domain_start)]),
            positions,
            positions.new_tensor([float(domain_end)]),
        )
    )
    flat = coords.reshape(-1).clamp(-0.5, float(count) - 0.5)
    right = torch.searchsorted(xp, flat, right=True).clamp(1, xp.numel() - 1)
    left = right - 1
    x0, x1 = xp[left], xp[right]
    y0, y1 = fp[left], fp[right]
    weight = (flat - x0) / (x1 - x0).clamp(min=1.0e-6)
    return (y0 + weight * (y1 - y0)).reshape(coords.shape)


def build_axis_points(base_points, positions, domain_start, domain_end):
    points = base_points.clone()
    center = base_points[:, 0]
    nominal_stride = base_points[:, 3].clamp(min=1.0e-6)
    mapped_center = map_selected_axis(
        center,
        positions,
        domain_start,
        domain_end,
    )
    mapped_left = map_selected_axis(
        center - 0.5 * nominal_stride,
        positions,
        domain_start,
        domain_end,
    )
    mapped_right = map_selected_axis(
        center + 0.5 * nominal_stride,
        positions,
        domain_start,
        domain_end,
    )
    physical_stride = (mapped_right - mapped_left).clamp(min=1.0e-6)
    range_scale = physical_stride / nominal_stride
    points[:, 0] = mapped_center
    points[:, 1] = points[:, 1] * range_scale
    points[:, 2] = points[:, 2] * range_scale
    points[:, 3] = physical_stride
    return points


def decode_axis(arrays, axis_name, native_axis_name):
    base_points = torch.from_numpy(arrays["base_points"])
    reg = torch.from_numpy(arrays["reg_distances"])
    base_mask = torch.from_numpy(arrays["base_mask"])
    native_mask = torch.from_numpy(arrays["native_mask"])
    native_points = torch.from_numpy(arrays["native_points"])
    native_proposals = torch.from_numpy(arrays["native_proposals"])
    scores = torch.from_numpy(arrays["cls_scores"])
    counts = arrays["native_valid_count"]
    domains = arrays["domain_sec"]
    axis_values = arrays[AXIS_SPECS[axis_name]]

    proposal_list = []
    score_list = []
    valid_indices = []
    dense_proposals = []
    point_error_max = 0.0
    proposal_error_max = 0.0
    valid_candidate_count = 0
    for idx in range(reg.shape[0]):
        count = int(counts[idx])
        positions = torch.from_numpy(axis_values[idx, :count].copy())
        start, end = (float(value) for value in domains[idx])
        points = build_axis_points(base_points, positions, start, end)
        mask = base_mask[idx] & (base_points[:, 0] < float(count))
        require(
            torch.equal(mask, native_mask[idx]),
            f"window {idx} candidate traversal differs between decode axes",
        )
        decoded = torch.stack(
            (
                points[:, 0] - reg[idx, :, 0] * points[:, 3],
                points[:, 0] + reg[idx, :, 1] * points[:, 3],
            ),
            dim=-1,
        )
        if axis_name == native_axis_name:
            point_error_max = max(
                point_error_max,
                float(torch.max(torch.abs(points - native_points[idx])).item()),
            )
            proposal_error_max = max(
                proposal_error_max,
                float(
                    torch.max(
                        torch.abs(decoded - native_proposals[idx])
                    ).item()
                ),
            )
        decoded[:, 0].clamp_(min=start, max=end)
        decoded[:, 1].clamp_(min=start, max=end)
        dense_proposals.append(decoded.numpy())
        indices = torch.nonzero(mask, as_tuple=True)[0]
        proposal_list.append(decoded[indices].contiguous())
        score_list.append(scores[idx, indices].contiguous())
        valid_indices.append(indices.numpy())
        valid_candidate_count += int(indices.numel())

    if axis_name == native_axis_name:
        require(
            point_error_max <= 5.0e-5,
            f"native point reconstruction error is too large: {point_error_max}",
        )
        require(
            proposal_error_max <= 1.0e-4,
            "native proposal reconstruction error is too large: "
            f"{proposal_error_max}",
        )
    return {
        "proposals": proposal_list,
        "scores": score_list,
        "valid_indices": valid_indices,
        "dense_proposals": np.stack(dense_proposals, axis=0),
        "valid_candidate_count": valid_candidate_count,
        "native_point_reconstruction_max_abs_error": point_error_max,
        "native_proposal_reconstruction_max_abs_error": proposal_error_max,
        "uses_captured_production_proposals": False,
    }


def build_metas(manifest):
    metas = []
    for window in manifest["windows"]:
        require(
            window.get("prediction_time_unit") == "seconds",
            "decode replay requires direct physical-second predictions",
        )
        metas.append(
            {
                "video_name": window["video_name"],
                "duration": float(window["duration"]),
                "prediction_time_unit": "seconds",
            }
        )
    return metas


def evaluate_predictions(cfg, prediction_payload):
    evaluation_cfg = copy.deepcopy(cfg.evaluation)
    evaluator = build_evaluator(
        dict(prediction_filename=prediction_payload, **evaluation_cfg)
    )
    return evaluator.evaluate()


def run_axis_mode(
    *,
    cfg,
    arrays,
    manifest,
    axis_name,
    native_axis_name,
    output_dir,
    evaluation_epoch,
):
    mode_dir = Path(output_dir) / "modes" / axis_name
    mode_dir.mkdir(parents=True, exist_ok=False)
    decoded = decode_axis(arrays, axis_name, native_axis_name)
    candidate_path = mode_dir / "decoded_candidates.npz"
    candidate_arrays = {
        "proposals": decoded["dense_proposals"],
        "valid_mask": arrays["native_mask"],
        "scores": arrays["cls_scores"],
    }
    atomic_write_npz(candidate_path, candidate_arrays)
    detector = SingleStageDetector()
    post_cfg = copy.deepcopy(cfg.post_processing)
    post_cfg.sliding_window = True
    pre_cross = detector.post_processing(
        (decoded["proposals"], decoded["scores"]),
        build_metas(manifest),
        post_cfg,
        manifest["class_map"],
    )
    pre_cross_payload = {
        "schema_version": "phystime_decode_cross_pre_cross_v1",
        "artifact_kind": "same_frozen_raw_tensors_axis_redecoded",
        "decode_axis": axis_name,
        "evaluation_epoch": evaluation_epoch,
        "results": pre_cross,
    }
    pre_cross_path = mode_dir / "pre_cross_window_detections.json.gz"
    atomic_write_json_gzip(pre_cross_path, pre_cross_payload)

    merged, nms_audit = apply_sliding_window_nms(
        pre_cross,
        post_cfg,
        return_audit=True,
    )
    result_payload = {
        "results": merged,
        "evaluation_epoch": evaluation_epoch,
    }
    metrics = evaluate_predictions(cfg, result_payload)
    metrics_payload = dict(metrics, evaluation_epoch=evaluation_epoch)
    result_path = mode_dir / "result_detection.json"
    metrics_path = mode_dir / "evaluation_metrics.json"
    audit_path = mode_dir / "post_processing_audit.json"
    atomic_write_json(result_path, result_payload)
    atomic_write_json(metrics_path, metrics_payload)
    atomic_write_json(audit_path, nms_audit)

    valid_sequence_sha = canonical_sha256(
        [indices.tolist() for indices in decoded["valid_indices"]]
    )
    report = {
        "schema_version": "phystime_decode_cross_mode_v1",
        "status": "completed",
        "decode_axis": axis_name,
        "native_axis": axis_name == native_axis_name,
        "shared_raw_tensor_sha256": canonical_sha256(
            {
                name: manifest["array_contract"][name][
                    "canonical_sha256"
                ]
                for name in (
                    "cls_logits",
                    "cls_scores",
                    "reg_distances",
                    "base_mask",
                    "base_points",
                )
            }
        ),
        "valid_candidate_sequence_sha256": valid_sequence_sha,
        "valid_candidate_count": decoded["valid_candidate_count"],
        "candidate_array_contract": {
            name: {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "canonical_sha256": array_sha256(array),
            }
            for name, array in candidate_arrays.items()
        },
        "uses_captured_production_proposals": decoded[
            "uses_captured_production_proposals"
        ],
        "native_point_reconstruction_max_abs_error": decoded[
            "native_point_reconstruction_max_abs_error"
        ],
        "native_proposal_reconstruction_max_abs_error": decoded[
            "native_proposal_reconstruction_max_abs_error"
        ],
        "pre_cross_canonical_results_sha256": canonical_sha256(pre_cross),
        "result_canonical_results_sha256": canonical_sha256(merged),
        "metrics": metrics_payload,
        "prediction_count": sum(len(items) for items in merged.values()),
        "artifacts": {
            "pre_cross": {
                "path": str(pre_cross_path.resolve()),
                "sha256": sha256_file(pre_cross_path),
            },
            "decoded_candidates": {
                "path": str(candidate_path.resolve()),
                "sha256": sha256_file(candidate_path),
            },
            "result": {
                "path": str(result_path.resolve()),
                "sha256": sha256_file(result_path),
            },
            "metrics": {
                "path": str(metrics_path.resolve()),
                "sha256": sha256_file(metrics_path),
            },
            "post_processing_audit": {
                "path": str(audit_path.resolve()),
                "sha256": sha256_file(audit_path),
            },
        },
        "post_processing_aggregate": nms_audit["aggregate"],
    }
    atomic_write_json(mode_dir / "mode_report.json", report)
    return report


def compare_native_direct(args, native_mode_report):
    direct_pre = load_gzip_json(args.direct_pre_cross)
    require(
        direct_pre.get("schema_version")
        == "opentad_pre_cross_window_detections_v1",
        "direct pre-cross artifact schema mismatch",
    )
    direct_result = load_json(args.direct_result)
    direct_metrics = load_json(args.direct_metrics)
    direct_pre_hash = canonical_sha256(direct_pre["results"])
    direct_result_hash = canonical_sha256(direct_result["results"])
    replay_metrics = {
        key: float(value)
        for key, value in native_mode_report["metrics"].items()
        if key != "evaluation_epoch"
    }
    direct_metric_values = {
        key: float(value)
        for key, value in direct_metrics.items()
        if key != "evaluation_epoch"
    }
    metric_deltas = {
        key: replay_metrics.get(key, float("nan"))
        - direct_metric_values.get(key, float("nan"))
        for key in sorted(set(replay_metrics) | set(direct_metric_values))
    }
    metrics_match = (
        replay_metrics.keys() == direct_metric_values.keys()
        and all(abs(value) <= 1.0e-12 for value in metric_deltas.values())
    )
    report = {
        "match": (
            direct_pre_hash
            == native_mode_report["pre_cross_canonical_results_sha256"]
            and direct_result_hash
            == native_mode_report["result_canonical_results_sha256"]
            and metrics_match
        ),
        "pre_cross_match": (
            direct_pre_hash
            == native_mode_report["pre_cross_canonical_results_sha256"]
        ),
        "result_match": (
            direct_result_hash
            == native_mode_report["result_canonical_results_sha256"]
        ),
        "metrics_match": metrics_match,
        "direct_pre_cross_canonical_sha256": direct_pre_hash,
        "replay_pre_cross_canonical_sha256": native_mode_report[
            "pre_cross_canonical_results_sha256"
        ],
        "direct_result_canonical_sha256": direct_result_hash,
        "replay_result_canonical_sha256": native_mode_report[
            "result_canonical_results_sha256"
        ],
        "metric_deltas": metric_deltas,
        "direct_artifacts": {
            "pre_cross_sha256": sha256_file(args.direct_pre_cross),
            "result_sha256": sha256_file(args.direct_result),
            "metrics_sha256": sha256_file(args.direct_metrics),
        },
    }
    return report


def validate_provenance(args, capture_manifest):
    run_manifest = load_json(args.run_manifest)
    source_completion = load_json(args.source_completion)
    source_manifest = load_json(args.source_manifest)
    p0_completion = load_json(args.p0_completion)
    require(
        run_manifest.get("runtime_commit") == args.expected_runtime_commit
        and run_manifest.get("runtime_tree") == args.expected_runtime_tree,
        "run manifest runtime binding mismatch",
    )
    require(
        run_manifest.get("arm") == args.arm
        and run_manifest.get("weights_source") == args.weights_source,
        "run manifest condition mismatch",
    )
    require(
        run_manifest.get("solver_ema")
        is (args.weights_source == "ema"),
        "run manifest solver EMA binding mismatch",
    )
    require(
        run_manifest.get("checkpoint_sha256")
        == sha256_file(args.checkpoint),
        "run manifest checkpoint hash mismatch",
    )
    require(
        run_manifest.get("effective_config_sha256")
        == capture_manifest["runtime"]["effective_config_sha256"],
        "capture/run effective config hash mismatch",
    )
    require(
        source_completion.get("validation_pass") is True,
        "source full60 completion did not pass",
    )
    require(
        source_manifest.get("commit") == args.source_commit
        and source_manifest.get("git_tree") == args.source_tree,
        "source full60 manifest snapshot mismatch",
    )
    require(
        source_manifest.get("variant") == args.arm,
        "source full60 arm mismatch",
    )
    require(
        source_manifest.get("dataset_manifest_sha256")
        == DATASET_MANIFEST_SHA256
        and source_manifest.get("pretrained_checkpoint_sha256")
        == VIDEOMAE_SHA256
        and run_manifest.get("videomae_checkpoint_sha256")
        == VIDEOMAE_SHA256,
        "source/run data or VideoMAE content binding mismatch",
    )
    source_checkpoint_sha = source_completion.get("artifacts", {}).get(
        "checkpoint",
        {},
    ).get("sha256")
    require(
        source_checkpoint_sha == sha256_file(args.checkpoint),
        "source completion checkpoint hash mismatch",
    )
    require(
        p0_completion.get("schema_version")
        == "phystime_p0_fullprecision_completion_v2"
        and p0_completion.get("runtime_commit") == P0_RUNTIME_COMMIT
        and p0_completion.get("runtime_tree") == P0_RUNTIME_TREE
        and p0_completion.get("validation_pass") is True
        and p0_completion.get("arm") == args.arm
        and p0_completion.get("weights_source") == args.weights_source,
        "matching P0 condition did not pass validation",
    )
    require(
        p0_completion["artifacts"]["checkpoint"]["sha256"]
        == sha256_file(args.checkpoint)
        and p0_completion["artifacts"]["gate"]["sha256"]
        == P0_GATE_SHA256,
        "P0 checkpoint binding mismatch",
    )
    return {
        "run_manifest_sha256": sha256_file(args.run_manifest),
        "source_completion_sha256": sha256_file(args.source_completion),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "p0_completion_sha256": sha256_file(args.p0_completion),
    }


def main():
    args = parse_args()
    runtime_commit, runtime_tree = current_git_identity()
    require(
        runtime_commit == args.expected_runtime_commit,
        "runtime commit differs from the submitted snapshot",
    )
    require(
        runtime_tree == args.expected_runtime_tree,
        "runtime tree differs from the submitted snapshot",
    )
    require(
        args.evaluation_epoch == 59,
        "decode cross replay is fixed to epoch 59",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    cfg = Config.fromfile(args.config, lazy_import=False)
    arrays, capture_manifest = load_and_validate_capture(args)
    provenance = validate_provenance(args, capture_manifest)
    native_axis_name = ARM_NATIVE_AXIS[args.arm]

    mode_reports = {}
    for axis_name in AXIS_SPECS:
        mode_reports[axis_name] = run_axis_mode(
            cfg=cfg,
            arrays=arrays,
            manifest=capture_manifest,
            axis_name=axis_name,
            native_axis_name=native_axis_name,
            output_dir=output_dir,
            evaluation_epoch=args.evaluation_epoch,
        )
    require(
        mode_reports["uniform_rank_seconds"][
            "shared_raw_tensor_sha256"
        ]
        == mode_reports["physical_time_seconds"][
            "shared_raw_tensor_sha256"
        ],
        "U/P replay did not share one raw tensor artifact",
    )
    require(
        mode_reports["uniform_rank_seconds"][
            "valid_candidate_sequence_sha256"
        ]
        == mode_reports["physical_time_seconds"][
            "valid_candidate_sequence_sha256"
        ],
        "U/P replay candidate traversal differs",
    )
    native_equivalence = compare_native_direct(
        args,
        mode_reports[native_axis_name],
    )
    validation_pass = native_equivalence["match"]
    metric_keys = sorted(
        set(mode_reports["uniform_rank_seconds"]["metrics"])
        & set(mode_reports["physical_time_seconds"]["metrics"])
        - {"evaluation_epoch"}
    )
    decode_delta = {
        "physical_minus_uniform_fraction": {
            key: float(
                mode_reports["physical_time_seconds"]["metrics"][key]
            )
            - float(mode_reports["uniform_rank_seconds"]["metrics"][key])
            for key in metric_keys
        },
        "physical_minus_uniform_percentage_points": {
            key: 100.0
            * (
                float(mode_reports["physical_time_seconds"]["metrics"][key])
                - float(
                    mode_reports["uniform_rank_seconds"]["metrics"][key]
                )
            )
            for key in metric_keys
        },
    }
    completion = {
        "schema_version": "phystime_frozen_decode_cross_replay_v1",
        "validation_pass": validation_pass,
        "status": "completed" if validation_pass else "failed_closed",
        "arm": args.arm,
        "weights_source": args.weights_source,
        "native_axis": native_axis_name,
        "evaluation_epoch": args.evaluation_epoch,
        "new_training": False,
        "same_frozen_raw_tensors_for_both_axes": True,
        "runtime": {
            "commit": runtime_commit,
            "git_tree": runtime_tree,
            "config": str(Path(args.config).resolve()),
        },
        "capture": {
            "artifact": str(Path(args.artifact).resolve()),
            "artifact_sha256": sha256_file(args.artifact),
            "manifest": str(Path(args.manifest).resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "window_sequence_sha256": capture_manifest[
                "window_sequence_sha256"
            ],
        },
        "numeric_precision": {
            "source_amp_enabled": capture_manifest[
                "source_amp_enabled"
            ],
            "source_tensor_dtypes": capture_manifest[
                "source_tensor_dtypes"
            ],
            "stored_tensor_dtypes": {
                name: str(array.dtype)
                for name, array in arrays.items()
                if name
                in {
                    "cls_logits",
                    "cls_scores",
                    "reg_distances",
                    "base_points",
                    "uniform_axis_sec",
                    "physical_axis_sec",
                }
            },
            "decode_compute_dtype": "float32",
            "decode_compute_device": "cpu",
        },
        "provenance": provenance,
        "mode_reports": mode_reports,
        "native_direct_equivalence": native_equivalence,
        "decode_delta": decode_delta,
    }
    completion_path = output_dir / "DECODE_CROSS_REPLAY_COMPLETE.json"
    atomic_write_json(completion_path, completion)
    if not validation_pass:
        raise SystemExit(
            "native-axis replay did not exactly reproduce direct inference"
        )
    print(json.dumps(completion, indent=2, sort_keys=True))


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


if __name__ == "__main__":
    main()
