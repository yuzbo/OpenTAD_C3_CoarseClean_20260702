#!/usr/bin/env python3
"""Validate one PhysTime frozen U/P replay with duplicate semantic recompute."""

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config

from opentad.cores.test_engine import apply_sliding_window_nms
from opentad.evaluations import build_evaluator


EXPECTED_EPOCH = 59
P0_RUNTIME_COMMIT = "c2cfcfa2470f9f1e0b9d10e397480f6c66aeaf2c"
P0_RUNTIME_TREE = "0b78dd402e8997239ef9d1b4b4cd8bfa4f7a6338"
P0_GATE_SHA256 = (
    "1ca0efcdeb9f6343da076a00660675759358ac467074919a34d01c0d7c7250d9"
)
P0_DATASET_MANIFEST_SHA256 = (
    "1da0bca28f14ca2f1e4b2baf0f199dce18f4dd925e0f097a70a3fcc1c13eb1b2"
)
P0_VIDEOMAE_SHA256 = (
    "4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251"
)
AXIS_ARRAYS = {
    "uniform_rank_seconds": "uniform_axis_sec",
    "physical_time_seconds": "physical_axis_sec",
}
ARM_NATIVE_AXIS = {
    "selected_axis": "uniform_rank_seconds",
    "physical_metric": "physical_time_seconds",
}
REQUIRED_CAPTURE_ARRAYS = {
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate one frozen PhysTime decode-cross run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path, description):
    path = Path(path)
    require(path.is_file(), f"missing {description}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_gzip_json(path, description):
    path = Path(path)
    require(path.is_file(), f"missing {description}: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


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
        default=lambda value: value.item(),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def artifact(path, started_at):
    path = Path(path).resolve()
    require(path.is_file(), f"missing artifact: {path}")
    require(
        path.stat().st_mtime >= started_at - 1.0,
        f"artifact predates the run: {path}",
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def recompute_map(coords, positions, start, end):
    count = int(positions.numel())
    ranks = torch.arange(count, dtype=torch.float32)
    xp = torch.cat(
        (
            torch.tensor([-0.5], dtype=torch.float32),
            ranks,
            torch.tensor([float(count) - 0.5], dtype=torch.float32),
        )
    )
    fp = torch.cat(
        (
            torch.tensor([float(start)], dtype=torch.float32),
            positions.to(dtype=torch.float32),
            torch.tensor([float(end)], dtype=torch.float32),
        )
    )
    values = coords.to(dtype=torch.float32).reshape(-1)
    values = torch.minimum(
        torch.maximum(values, torch.tensor(-0.5)),
        torch.tensor(float(count) - 0.5),
    )
    upper = torch.searchsorted(xp, values, right=True)
    upper = torch.minimum(
        torch.maximum(upper, torch.tensor(1)),
        torch.tensor(xp.numel() - 1),
    )
    lower = upper - 1
    denominator = torch.maximum(
        xp[upper] - xp[lower],
        torch.tensor(1.0e-6, dtype=torch.float32),
    )
    fraction = (values - xp[lower]) / denominator
    return (
        fp[lower] + fraction * (fp[upper] - fp[lower])
    ).reshape(coords.shape)


def recompute_dense_decode(arrays, axis_name):
    base = torch.from_numpy(arrays["base_points"])
    reg = torch.from_numpy(arrays["reg_distances"])
    base_mask = torch.from_numpy(arrays["base_mask"])
    counts = arrays["native_valid_count"]
    domains = arrays["domain_sec"]
    axes = arrays[AXIS_ARRAYS[axis_name]]
    dense = []
    masks = []
    points_all = []
    for window_idx in range(reg.shape[0]):
        count = int(counts[window_idx])
        positions = torch.from_numpy(axes[window_idx, :count].copy())
        start = float(domains[window_idx, 0])
        end = float(domains[window_idx, 1])
        center = base[:, 0]
        nominal_stride = torch.maximum(
            base[:, 3],
            torch.tensor(1.0e-6, dtype=torch.float32),
        )
        mapped_center = recompute_map(
            center,
            positions,
            start,
            end,
        )
        mapped_left = recompute_map(
            center - 0.5 * nominal_stride,
            positions,
            start,
            end,
        )
        mapped_right = recompute_map(
            center + 0.5 * nominal_stride,
            positions,
            start,
            end,
        )
        stride = torch.maximum(
            mapped_right - mapped_left,
            torch.tensor(1.0e-6, dtype=torch.float32),
        )
        scale = stride / nominal_stride
        points = base.clone()
        points[:, 0] = mapped_center
        points[:, 1] = base[:, 1] * scale
        points[:, 2] = base[:, 2] * scale
        points[:, 3] = stride
        proposals = torch.stack(
            (
                mapped_center - reg[window_idx, :, 0] * stride,
                mapped_center + reg[window_idx, :, 1] * stride,
            ),
            dim=-1,
        )
        proposals[:, 0].clamp_(start, end)
        proposals[:, 1].clamp_(start, end)
        mask = base_mask[window_idx] & (center < float(count))
        dense.append(proposals.numpy())
        masks.append(mask.numpy())
        points_all.append(points.numpy())
    return (
        np.stack(dense, axis=0),
        np.stack(masks, axis=0),
        np.stack(points_all, axis=0),
    )


def validate_capture(run_dir, run_manifest):
    direct_dir = run_dir / "direct_work" / "gpu1_id0"
    manifest_path = direct_dir / "decode_replay_manifest.json"
    artifact_path = direct_dir / "decode_replay_inputs.npz"
    capture = read_json(manifest_path, "capture manifest")
    require(
        capture.get("schema_version")
        == "phystime_decode_replay_inputs_v1",
        "capture schema mismatch",
    )
    require(
        capture.get("artifact_kind")
        == "frozen_pre_decode_actionformer_tensors",
        "capture artifact kind mismatch",
    )
    require(
        capture.get("runtime", {}).get("commit")
        == run_manifest["runtime_commit"],
        "capture/run commit mismatch",
    )
    require(
        capture.get("runtime", {}).get("git_tree")
        == run_manifest["runtime_tree"],
        "capture/run tree mismatch",
    )
    require(
        capture.get("source", {}).get("commit")
        == run_manifest["source_commit"]
        and capture.get("source", {}).get("git_tree")
        == run_manifest["source_tree"],
        "capture/run source snapshot mismatch",
    )
    require(
        capture.get("runtime", {}).get("effective_config_sha256")
        == run_manifest["effective_config_sha256"],
        "capture/run config hash mismatch",
    )
    require(
        capture.get("checkpoint", {}).get("sha256")
        == run_manifest["checkpoint_sha256"],
        "capture/run checkpoint hash mismatch",
    )
    require(
        Path(capture["checkpoint"]["path"]).resolve()
        == Path(run_manifest["checkpoint"]).resolve(),
        "capture/run checkpoint path mismatch",
    )
    require(
        capture.get("train_axis")
        == ARM_NATIVE_AXIS[run_manifest["arm"]],
        "capture train axis mismatch",
    )
    require(
        capture.get("weights_source") == run_manifest["weights_source"],
        "capture weights source mismatch",
    )
    require(
        int(capture.get("evaluation_epoch", -1)) == EXPECTED_EPOCH,
        "capture epoch mismatch",
    )
    require(
        capture.get("artifact", {}).get("sha256")
        == sha256_file(artifact_path),
        "capture NPZ file hash mismatch",
    )
    require(
        Path(capture["artifact"]["path"]).resolve()
        == artifact_path.resolve(),
        "capture NPZ path mismatch",
    )
    with np.load(artifact_path, allow_pickle=False) as archive:
        require(
            set(archive.files) == REQUIRED_CAPTURE_ARRAYS,
            "capture array set mismatch",
        )
        arrays = {name: archive[name] for name in archive.files}
    require(
        set(capture.get("array_contract", {}))
        == REQUIRED_CAPTURE_ARRAYS,
        "capture array manifest set mismatch",
    )
    for name, array in arrays.items():
        contract = capture["array_contract"][name]
        require(
            contract["dtype"] == str(array.dtype)
            and contract["shape"] == list(array.shape)
            and contract["canonical_sha256"] == array_sha256(array),
            f"capture array contract mismatch: {name}",
        )

    logits = arrays["cls_logits"]
    scores = arrays["cls_scores"]
    reg = arrays["reg_distances"]
    require(
        logits.dtype == scores.dtype == reg.dtype == np.float32,
        "capture logits/scores/reg must be float32",
    )
    require(
        logits.ndim == 3
        and scores.shape == logits.shape
        and reg.shape == logits.shape[:2] + (2,),
        "capture dense prediction shapes mismatch",
    )
    require(
        arrays["base_mask"].dtype == np.bool_
        and arrays["native_mask"].dtype == np.bool_,
        "capture masks must be bool",
    )
    require(
        arrays["base_points"].shape == (logits.shape[1], 4),
        "capture base point shape mismatch",
    )
    require(
        arrays["native_points"].shape
        == (logits.shape[0], logits.shape[1], 4),
        "capture native point shape mismatch",
    )
    require(
        arrays["native_proposals"].shape
        == (logits.shape[0], logits.shape[1], 2),
        "capture native proposal shape mismatch",
    )
    require(
        np.isfinite(logits).all()
        and np.isfinite(scores).all()
        and np.isfinite(reg).all()
        and np.isfinite(arrays["base_points"]).all()
        and np.isfinite(arrays["native_points"]).all()
        and np.isfinite(arrays["native_proposals"]).all(),
        "capture has non-finite dense tensors",
    )
    require(np.all(reg >= 0.0), "capture has negative regression distances")
    require(
        len(capture["windows"]) == logits.shape[0]
        and len(capture["class_map"]) == logits.shape[2],
        "capture windows/class map mismatch",
    )
    observation_hashes = []
    window_hashes = []
    for window in capture["windows"]:
        observation_record = {
            key: window[key]
            for key in (
                "index",
                "video_name",
                "duration",
                "prediction_time_unit",
                "window_start_frame",
                "native_valid_count",
                "native_token_count",
                "raw_observation_count",
                "selected_raw_frame_count",
                "selected_raw_frame_sha256",
                "selected_dense_sha256",
                "domain_start_sec",
                "domain_end_sec",
            )
        }
        observation_hash = canonical_sha256(observation_record)
        require(
            observation_hash
            == window["observation_binding_sha256"],
            "capture observation binding hash mismatch",
        )
        observation_hashes.append(observation_hash)
        window_record = dict(window)
        stored_window_hash = window_record.pop("window_binding_sha256")
        window_hash = canonical_sha256(window_record)
        require(
            window_hash == stored_window_hash,
            "capture window binding hash mismatch",
        )
        window_hashes.append(window_hash)
    require(
        canonical_sha256(window_hashes)
        == capture["window_sequence_sha256"],
        "capture window sequence hash mismatch",
    )
    require(
        canonical_sha256(observation_hashes)
        == capture["observation_sequence_sha256"],
        "capture observation sequence hash mismatch",
    )
    for idx, count_value in enumerate(arrays["native_valid_count"]):
        count = int(count_value)
        require(0 < count <= arrays["uniform_axis_sec"].shape[1], "bad J")
        expected_mask = arrays["base_mask"][idx] & (
            arrays["base_points"][:, 0] < float(count)
        )
        require(
            np.array_equal(expected_mask, arrays["native_mask"][idx]),
            f"window {idx} native mask mismatch",
        )
        for axis_array_name in AXIS_ARRAYS.values():
            values = arrays[axis_array_name][idx]
            require(
                np.isfinite(values[:count]).all()
                and np.all(np.diff(values[:count]) > 0.0)
                and np.isnan(values[count:]).all(),
                f"window {idx} axis contract mismatch",
            )
    return arrays, capture, manifest_path, artifact_path


def load_candidate_artifact(mode_dir, mode_report, capture_arrays):
    candidate_path = mode_dir / "decoded_candidates.npz"
    require(
        mode_report["artifacts"]["decoded_candidates"]["sha256"]
        == sha256_file(candidate_path),
        "decoded candidate file hash mismatch",
    )
    with np.load(candidate_path, allow_pickle=False) as archive:
        require(
            set(archive.files) == {"proposals", "valid_mask", "scores"},
            "decoded candidate array set mismatch",
        )
        arrays = {name: archive[name] for name in archive.files}
    for name, array in arrays.items():
        contract = mode_report["candidate_array_contract"][name]
        require(
            contract["dtype"] == str(array.dtype)
            and contract["shape"] == list(array.shape)
            and contract["canonical_sha256"] == array_sha256(array),
            f"decoded candidate contract mismatch: {name}",
        )
    require(
        np.array_equal(arrays["valid_mask"], capture_arrays["native_mask"]),
        "decoded candidate mask differs from capture",
    )
    require(
        np.array_equal(arrays["scores"], capture_arrays["cls_scores"]),
        "decoded candidate scores differ from capture",
    )
    return arrays, candidate_path


def production_semantic_recompute_post_process(
    candidate_arrays, capture, post_cfg
):
    proposals = torch.from_numpy(candidate_arrays["proposals"])
    valid_mask = torch.from_numpy(candidate_arrays["valid_mask"])
    scores_all = torch.from_numpy(candidate_arrays["scores"])
    class_map = capture["class_map"]
    threshold = float(getattr(post_cfg, "pre_nms_thresh", 0.001))
    topk = int(getattr(post_cfg, "pre_nms_topk", 2000))
    round_before = bool(
        getattr(post_cfg, "round_before_cross_window_nms", True)
    )
    segment_digits = int(getattr(post_cfg, "segment_round_digits", 2))
    score_digits = int(getattr(post_cfg, "score_round_digits", 4))
    results = {}
    for idx, window in enumerate(capture["windows"]):
        mask = valid_mask[idx]
        segments = proposals[idx, mask].clone()
        scores = scores_all[idx, mask]
        flat = scores.flatten()
        keep = flat > threshold
        flat = flat[keep]
        flat_indices = torch.nonzero(keep, as_tuple=True)[0]
        num_topk = min(topk, int(flat_indices.numel()))
        flat, order = flat.sort(descending=True)
        flat = flat[:num_topk].clone()
        flat_indices = flat_indices[order[:num_topk]].clone()
        point_indices = torch.div(
            flat_indices,
            len(class_map),
            rounding_mode="floor",
        )
        class_indices = torch.fmod(flat_indices, len(class_map))
        segments = segments[point_indices]
        duration = float(window["duration"])
        segments.clamp_(0.0, duration)
        video_name = window["video_name"]
        outputs = []
        for segment, label_idx, score in zip(
            segments,
            class_indices,
            flat,
        ):
            segment_value = [
                float(segment[0].item()),
                float(segment[1].item()),
            ]
            score_value = float(score.item())
            if round_before:
                segment_value = [
                    round(value, segment_digits) for value in segment_value
                ]
                score_value = round(score_value, score_digits)
            outputs.append(
                {
                    "segment": segment_value,
                    "label": class_map[int(label_idx.item())],
                    "score": score_value,
                }
            )
        results.setdefault(video_name, []).extend(outputs)
    return results


def production_semantic_recompute_evaluate(cfg, prediction_payload):
    evaluation_cfg = copy.deepcopy(cfg.evaluation)
    evaluator = build_evaluator(
        dict(prediction_filename=prediction_payload, **evaluation_cfg)
    )
    return evaluator.evaluate()


def require_metrics_close(observed, expected, description):
    observed_values = {
        key: float(value)
        for key, value in observed.items()
        if key != "evaluation_epoch"
    }
    expected_values = {
        key: float(value)
        for key, value in expected.items()
        if key != "evaluation_epoch"
    }
    require(
        observed_values.keys() == expected_values.keys(),
        f"{description}: metric key mismatch",
    )
    deltas = {
        key: observed_values[key] - expected_values[key]
        for key in observed_values
    }
    require(
        all(math.isfinite(value) and abs(value) <= 1.0e-12 for value in deltas.values()),
        f"{description}: metric values differ: {deltas}",
    )
    return deltas


def scan_logs(run_dir, run_manifest):
    patterns = (
        r"\btraceback\b",
        r"\bcuda out of memory\b",
        r"\boutofmemoryerror\b",
        r"\boom(?:[- ]|_)?kill(?:ed)?\b",
        r"\bkilled\b",
        r"\bsegmentation fault\b|\bsegfault\b",
        r"\bbus error\b",
        r"\bnccl\b[^\n]{0,120}\berror\b",
        r"\bloss\s*=\s*(?:nan|[-+]?inf)\b",
        r"\bnan gradient\b",
        r"\bnon[- ]finite\b",
        r"\bpytorchstreamwriter\b",
        r"\bdependencyneversatisfied\b",
    )
    findings = {}
    for path in (
        run_dir / "inference.out",
        run_dir / "replay.out",
        run_dir / "validator.out",
        Path(run_manifest["slurm"]["stdout"]),
        Path(run_manifest["slurm"]["stderr"]),
    ):
        require(path.is_file(), f"required log is missing: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = [
            pattern
            for pattern in patterns
            if re.search(pattern, text, flags=re.IGNORECASE)
        ]
        if hits:
            findings[str(path)] = hits
    require(not findings, f"fatal log patterns found: {findings}")
    return findings


def validate_run(run_dir):
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    run_manifest = read_json(manifest_path, "run manifest")
    require(
        run_manifest.get("schema_version")
        == "phystime_decode_cross_run_manifest_v1",
        "run manifest schema mismatch",
    )
    require(run_manifest.get("new_training") is False, "run trained a model")
    require(
        run_manifest.get("frozen_checkpoint_replay") is True,
        "run is not marked as frozen replay",
    )
    require(
        int(run_manifest.get("evaluation_epoch", -1)) == EXPECTED_EPOCH,
        "run manifest epoch mismatch",
    )
    arm = run_manifest["arm"]
    weights_source = run_manifest["weights_source"]
    require(arm in ARM_NATIVE_AXIS, "unknown replay arm")
    require(weights_source in {"online", "ema"}, "unknown weights source")
    require(
        run_manifest.get("solver_ema") is (weights_source == "ema"),
        "run manifest solver EMA binding mismatch",
    )
    slurm = run_manifest.get("slurm", {})
    require(
        slurm.get("job_id") == os.environ.get("SLURM_JOB_ID")
        and slurm.get("job_name") == os.environ.get("SLURM_JOB_NAME")
        and slurm.get("comment")
        == os.environ.get("PHYSTIME_EXPECTED_JOB_COMMENT")
        and slurm.get("dag_token") == os.environ.get("PHYSTIME_DAG_TOKEN"),
        "live Slurm identity differs from the run manifest",
    )
    started_at = float(run_manifest["started_at_unix"])
    gate_path = Path(run_manifest["gate"]).resolve()
    require(
        sha256_file(gate_path) == run_manifest["gate_sha256"],
        "decode-cross gate hash changed",
    )
    gate = read_json(gate_path, "decode-cross gate")
    require(gate.get("gate_pass") is True, "decode-cross gate did not pass")
    require(
        gate["runtime"]["commit"] == run_manifest["runtime_commit"]
        and gate["runtime"]["git_tree"] == run_manifest["runtime_tree"],
        "decode-cross gate runtime snapshot mismatch",
    )
    require(
        gate["source"]["commit"] == run_manifest["source_commit"]
        and gate["source"]["git_tree"] == run_manifest["source_tree"],
        "decode-cross gate source snapshot mismatch",
    )
    preflight_path = Path(run_manifest["preflight_manifest"]).resolve()
    require(
        sha256_file(preflight_path)
        == run_manifest["preflight_manifest_sha256"]
        == gate["preflight"]["sha256"]
        and Path(gate["preflight"]["path"]).resolve() == preflight_path,
        "decode-cross preflight binding mismatch",
    )
    preflight = read_json(preflight_path, "decode-cross preflight")
    require(
        preflight.get("schema_version")
        == "phystime_decode_cross_preflight_v1"
        and preflight.get("validation_pass") is True,
        "decode-cross preflight did not pass",
    )
    runtime_preflight_path = Path(
        run_manifest["runtime_preflight_manifest"]
    ).resolve()
    require(
        runtime_preflight_path.is_file()
        and sha256_file(runtime_preflight_path)
        == run_manifest["runtime_preflight_manifest_sha256"]
        == run_manifest["preflight_manifest_sha256"],
        "replay-time full preflight differs from submission preflight",
    )
    runtime_preflight = read_json(
        runtime_preflight_path,
        "replay-time preflight",
    )
    require(
        runtime_preflight.get("schema_version")
        == "phystime_decode_cross_preflight_v1"
        and runtime_preflight.get("validation_pass") is True
        and canonical_sha256(runtime_preflight)
        == canonical_sha256(preflight),
        "replay-time preflight payload mismatch",
    )
    require(
        gate.get("dataset_manifest_sha256")
        == P0_DATASET_MANIFEST_SHA256
        and gate["videomae_checkpoint"]["sha256"]
        == P0_VIDEOMAE_SHA256,
        "decode-cross gate data/VideoMAE content binding mismatch",
    )
    checkpoint_key = (
        "selected_axis" if arm == "selected_axis" else "physical_metric"
    )
    require(
        gate["checkpoints"][checkpoint_key]["sha256"]
        == run_manifest["checkpoint_sha256"],
        "decode-cross gate checkpoint mismatch",
    )
    condition_prefix = (
        "selected" if arm == "selected_axis" else "physical"
    )
    condition_name = f"{condition_prefix}_{weights_source}"
    real_window = gate["real_windows"][condition_name]
    require(
        gate.get("all_native_direct_exact_equivalence") is True
        and real_window["native_direct_exact_equivalence"] is True
        and real_window["raw_tensors_immutable"] is True,
        "decode-cross four-condition real-window gate did not pass",
    )
    require(
        real_window["checkpoint_state_dict_sha256"]
        == run_manifest["checkpoint_state_dict_sha256"]
        and real_window["checkpoint_state_key"]
        == run_manifest["checkpoint_state_key"],
        "decode-cross gate checkpoint state binding mismatch",
    )

    cfg = Config.fromfile(run_manifest["config"], lazy_import=False)
    capture_arrays, capture, capture_manifest_path, capture_artifact_path = (
        validate_capture(run_dir, run_manifest)
    )
    replay_dir = run_dir / "replay"
    producer_completion_path = (
        replay_dir / "DECODE_CROSS_REPLAY_COMPLETE.json"
    )
    producer = read_json(producer_completion_path, "producer completion")
    require(
        producer.get("validation_pass") is True
        and producer.get("status") == "completed",
        "producer did not pass",
    )
    require(
        producer.get("arm") == arm
        and producer.get("weights_source") == weights_source,
        "producer condition mismatch",
    )
    require(
        producer.get("same_frozen_raw_tensors_for_both_axes") is True,
        "producer did not bind one shared raw tensor artifact",
    )
    numeric_precision = producer.get("numeric_precision", {})
    require(
        numeric_precision.get("source_amp_enabled")
        == capture.get("source_amp_enabled")
        and numeric_precision.get("source_tensor_dtypes")
        == capture.get("source_tensor_dtypes")
        and numeric_precision.get("decode_compute_dtype") == "float32"
        and numeric_precision.get("decode_compute_device") == "cpu"
        and set(numeric_precision.get("stored_tensor_dtypes", {}).values())
        == {"float32"},
        "producer numeric precision provenance mismatch",
    )

    mode_metrics = {}
    mode_artifacts = {}
    recompute_audits = {}
    candidate_errors = {}
    for axis_name in AXIS_ARRAYS:
        mode_dir = replay_dir / "modes" / axis_name
        mode_report_path = mode_dir / "mode_report.json"
        mode_report = read_json(mode_report_path, f"{axis_name} mode report")
        require(
            mode_report.get("status") == "completed"
            and mode_report.get("decode_axis") == axis_name,
            f"{axis_name} mode did not complete",
        )
        candidate_arrays, candidate_path = load_candidate_artifact(
            mode_dir,
            mode_report,
            capture_arrays,
        )
        recomputed_decoded, recomputed_mask, recomputed_points = (
            recompute_dense_decode(capture_arrays, axis_name)
        )
        require(
            np.array_equal(recomputed_mask, candidate_arrays["valid_mask"]),
            f"{axis_name} recomputed mask differs",
        )
        proposal_error = float(
            np.max(
                np.abs(
                    recomputed_decoded - candidate_arrays["proposals"]
                )
            )
        )
        require(
            proposal_error <= 1.0e-4,
            f"{axis_name} recomputed proposal error is too large: {proposal_error}",
        )
        point_error = 0.0
        if axis_name == ARM_NATIVE_AXIS[arm]:
            point_error = float(
                np.max(
                    np.abs(
                        recomputed_points
                        - capture_arrays["native_points"]
                    )
                )
            )
            require(
                point_error <= 5.0e-5,
                f"native recomputed point error is too large: {point_error}",
            )
            require(
                mode_report.get("uses_captured_production_proposals")
                is False,
                "native replay substituted captured production proposals",
            )
        candidate_errors[axis_name] = {
            "proposal_max_abs_error": proposal_error,
            "native_point_max_abs_error": point_error,
        }

        manual_pre_cross = production_semantic_recompute_post_process(
            candidate_arrays,
            capture,
            cfg.post_processing,
        )
        pre_cross_path = mode_dir / "pre_cross_window_detections.json.gz"
        pre_cross = read_gzip_json(pre_cross_path, f"{axis_name} pre-cross")
        require(
            pre_cross.get("schema_version")
            == "phystime_decode_cross_pre_cross_v1"
            and pre_cross.get("decode_axis") == axis_name,
            f"{axis_name} pre-cross schema mismatch",
        )
        require(
            canonical_sha256(manual_pre_cross)
            == canonical_sha256(pre_cross["results"])
            == mode_report["pre_cross_canonical_results_sha256"],
            f"{axis_name} production-semantic pre-cross recompute differs",
        )
        post_cfg = copy.deepcopy(cfg.post_processing)
        post_cfg.sliding_window = True
        merged, audit = apply_sliding_window_nms(
            manual_pre_cross,
            post_cfg,
            return_audit=True,
        )
        result_path = mode_dir / "result_detection.json"
        metrics_path = mode_dir / "evaluation_metrics.json"
        audit_path = mode_dir / "post_processing_audit.json"
        result = read_json(result_path, f"{axis_name} result")
        metrics = read_json(metrics_path, f"{axis_name} metrics")
        stored_audit = read_json(audit_path, f"{axis_name} audit")
        require(
            canonical_sha256(merged)
            == canonical_sha256(result["results"])
            == mode_report["result_canonical_results_sha256"],
            f"{axis_name} production-semantic final recompute differs",
        )
        require(
            canonical_sha256(audit) == canonical_sha256(stored_audit),
            f"{axis_name} production-semantic NMS audit differs",
        )
        recomputed_metrics = production_semantic_recompute_evaluate(
            cfg,
            {
                "results": merged,
                "evaluation_epoch": EXPECTED_EPOCH,
            },
        )
        require_metrics_close(
            recomputed_metrics,
            metrics,
            f"{axis_name} production-semantic recomputed metrics",
        )
        require_metrics_close(
            metrics,
            mode_report["metrics"],
            f"{axis_name} producer mode metrics",
        )
        require(
            audit["aggregate"]["invalid_detections"] == 0
            and audit["aggregate"]["post_nms_invalid_detections"] == 0,
            f"{axis_name} contains invalid proposals",
        )
        mode_metrics[axis_name] = {
            key: float(value)
            for key, value in metrics.items()
            if key != "evaluation_epoch"
        }
        recompute_audits[axis_name] = audit["aggregate"]
        mode_artifacts[axis_name] = {
            "mode_report": artifact(mode_report_path, started_at),
            "decoded_candidates": artifact(candidate_path, started_at),
            "pre_cross": artifact(pre_cross_path, started_at),
            "result": artifact(result_path, started_at),
            "metrics": artifact(metrics_path, started_at),
            "post_processing_audit": artifact(audit_path, started_at),
        }

    uniform_report = producer["mode_reports"]["uniform_rank_seconds"]
    physical_report = producer["mode_reports"]["physical_time_seconds"]
    require(
        uniform_report["shared_raw_tensor_sha256"]
        == physical_report["shared_raw_tensor_sha256"],
        "producer U/P raw tensor hash differs",
    )
    require(
        uniform_report["valid_candidate_sequence_sha256"]
        == physical_report["valid_candidate_sequence_sha256"],
        "producer U/P candidate traversal hash differs",
    )

    native_axis = ARM_NATIVE_AXIS[arm]
    direct_pre_path = (
        run_dir
        / "direct_work"
        / "gpu1_id0"
        / "pre_cross_window_detections.json.gz"
    )
    direct_result_path = (
        run_dir / "direct_work" / "gpu1_id0" / "result_detection.json"
    )
    direct_metrics_path = (
        run_dir / "direct_work" / "gpu1_id0" / "evaluation_metrics.json"
    )
    direct_pre = read_gzip_json(direct_pre_path, "direct pre-cross")
    direct_result = read_json(direct_result_path, "direct result")
    direct_metrics = read_json(direct_metrics_path, "direct metrics")
    native_pre = read_gzip_json(
        replay_dir
        / "modes"
        / native_axis
        / "pre_cross_window_detections.json.gz",
        "native replay pre-cross",
    )
    native_result = read_json(
        replay_dir / "modes" / native_axis / "result_detection.json",
        "native replay result",
    )
    require(
        canonical_sha256(direct_pre["results"])
        == canonical_sha256(native_pre["results"]),
        "native replay does not exactly reproduce direct pre-cross predictions",
    )
    require(
        canonical_sha256(direct_result["results"])
        == canonical_sha256(native_result["results"]),
        "native replay does not exactly reproduce direct final predictions",
    )
    require_metrics_close(
        direct_metrics,
        mode_metrics[native_axis],
        "native replay/direct metrics",
    )
    require(
        producer["native_direct_equivalence"]["match"] is True,
        "producer native equivalence is false",
    )

    checkpoint_path = Path(run_manifest["checkpoint"])
    source_completion_path = Path(run_manifest["source_completion"])
    source_manifest_path = Path(run_manifest["source_manifest"])
    p0_completion_path = Path(run_manifest["p0_completion"])
    require(
        sha256_file(checkpoint_path) == run_manifest["checkpoint_sha256"],
        "checkpoint changed after run manifest creation",
    )
    require(
        sha256_file(source_completion_path)
        == run_manifest["source_completion_sha256"],
        "source completion changed",
    )
    require(
        sha256_file(source_manifest_path)
        == run_manifest["source_manifest_sha256"],
        "source manifest changed",
    )
    require(
        sha256_file(p0_completion_path)
        == run_manifest["p0_completion_sha256"],
        "P0 completion changed",
    )
    source_completion = read_json(
        source_completion_path,
        "source completion",
    )
    source_manifest = read_json(
        source_manifest_path,
        "source manifest",
    )
    p0_completion = read_json(p0_completion_path, "P0 completion")
    require(
        source_completion.get("validation_pass") is True
        and p0_completion.get("validation_pass") is True,
        "source or P0 provenance did not pass",
    )
    require(
        source_manifest.get("commit") == run_manifest["source_commit"]
        and source_manifest.get("git_tree")
        == run_manifest["source_tree"]
        and source_manifest.get("variant") == arm,
        "source full60 manifest identity mismatch",
    )
    require(
        source_manifest.get("dataset_manifest_sha256")
        == P0_DATASET_MANIFEST_SHA256
        and source_manifest.get("pretrained_checkpoint_sha256")
        == P0_VIDEOMAE_SHA256
        and run_manifest.get("videomae_checkpoint_sha256")
        == P0_VIDEOMAE_SHA256,
        "source/run data or VideoMAE content binding mismatch",
    )
    require(
        source_completion.get("artifacts", {})
        .get("checkpoint", {})
        .get("sha256")
        == run_manifest["checkpoint_sha256"],
        "source full60 checkpoint binding mismatch",
    )
    require(
        p0_completion.get("schema_version")
        == "phystime_p0_fullprecision_completion_v2"
        and p0_completion.get("runtime_commit") == P0_RUNTIME_COMMIT
        and p0_completion.get("runtime_tree") == P0_RUNTIME_TREE
        and p0_completion.get("arm") == arm
        and p0_completion.get("weights_source") == weights_source,
        "P0 provenance condition mismatch",
    )
    require(
        p0_completion.get("source_commit")
        == run_manifest["source_commit"]
        and p0_completion.get("source_tree")
        == run_manifest["source_tree"],
        "P0 source snapshot mismatch",
    )
    require(
        p0_completion["artifacts"]["gate"]["sha256"] == P0_GATE_SHA256,
        "P0 gate hash mismatch",
    )
    for key in ("direct_result", "direct_metrics", "pre_cross_window"):
        record = p0_completion["artifacts"][key]
        require(
            sha256_file(record["path"]) == record["sha256"],
            f"P0 {key} artifact hash mismatch",
        )
    p0_direct_result = read_json(
        p0_completion["artifacts"]["direct_result"]["path"],
        "P0 direct result",
    )
    p0_direct_metrics = read_json(
        p0_completion["artifacts"]["direct_metrics"]["path"],
        "P0 direct metrics",
    )
    p0_direct_pre = read_gzip_json(
        p0_completion["artifacts"]["pre_cross_window"]["path"],
        "P0 direct pre-cross",
    )
    require(
        canonical_sha256(direct_result["results"])
        == canonical_sha256(p0_direct_result["results"]),
        "current capture-enabled direct result differs from reviewed P0 direct",
    )
    require(
        canonical_sha256(direct_pre["results"])
        == canonical_sha256(p0_direct_pre["results"]),
        "current capture-enabled direct pre-cross differs from reviewed P0 direct",
    )
    require_metrics_close(
        direct_metrics,
        p0_direct_metrics,
        "current direct/P0 direct metrics artifact",
    )
    require_metrics_close(
        direct_metrics,
        p0_completion["direct_fullprecision_filtered_metrics"],
        "current direct/P0 reviewed metrics",
    )
    findings = scan_logs(run_dir, run_manifest)

    metric_keys = sorted(
        set(mode_metrics["uniform_rank_seconds"])
        & set(mode_metrics["physical_time_seconds"])
    )
    validated_delta = {
        key: mode_metrics["physical_time_seconds"][key]
        - mode_metrics["uniform_rank_seconds"][key]
        for key in metric_keys
    }
    producer_delta = producer["decode_delta"][
        "physical_minus_uniform_fraction"
    ]
    require_metrics_close(
        validated_delta,
        producer_delta,
        "producer decode delta",
    )

    return {
        "schema_version": "phystime_decode_cross_completion_v1",
        "validation_pass": True,
        "status": "tested",
        "completed_at_unix": time.time(),
        "run_dir": str(run_dir),
        "arm": arm,
        "weights_source": weights_source,
        "native_axis": native_axis,
        "evaluation_epoch": EXPECTED_EPOCH,
        "new_training": False,
        "same_frozen_raw_tensors_for_both_axes": True,
        "native_direct_exact_equivalence": True,
        "reviewed_p0_direct_exact_equivalence": True,
        "runtime_commit": run_manifest["runtime_commit"],
        "runtime_tree": run_manifest["runtime_tree"],
        "source_commit": run_manifest["source_commit"],
        "source_tree": run_manifest["source_tree"],
        "mode_metrics": mode_metrics,
        "physical_minus_uniform_fraction": validated_delta,
        "physical_minus_uniform_percentage_points": {
            key: 100.0 * value for key, value in validated_delta.items()
        },
        "candidate_reconstruction_errors": candidate_errors,
        "numeric_precision": numeric_precision,
        "post_processing_aggregates": recompute_audits,
        "fatal_log_findings": findings,
        "artifacts": {
            "run_manifest": artifact(manifest_path, started_at),
            "decode_cross_gate": artifact(gate_path, 0.0),
            "preflight_manifest": artifact(preflight_path, 0.0),
            "runtime_preflight_manifest": artifact(
                runtime_preflight_path,
                0.0,
            ),
            "capture_manifest": artifact(
                capture_manifest_path,
                started_at,
            ),
            "capture_npz": artifact(capture_artifact_path, started_at),
            "producer_completion": artifact(
                producer_completion_path,
                started_at,
            ),
            "direct_pre_cross": artifact(
                direct_pre_path,
                started_at,
            ),
            "direct_result": artifact(direct_result_path, started_at),
            "direct_metrics": artifact(direct_metrics_path, started_at),
            "checkpoint": artifact(checkpoint_path, 0.0),
            "source_completion": artifact(source_completion_path, 0.0),
            "source_manifest": artifact(source_manifest_path, 0.0),
            "p0_completion": artifact(p0_completion_path, 0.0),
        },
        "mode_artifacts": mode_artifacts,
    }


def main():
    args = parse_args()
    completion = validate_run(args.run_dir)
    atomic_write_json(args.output, completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
