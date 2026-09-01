#!/usr/bin/env python3
"""Run a real one-window gate for frozen PhysTime decode cross replay."""

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config

from opentad.cores.phystime_decode_replay_capture import (
    build_decode_replay_collector,
)
from opentad.datasets import build_dataset
from opentad.datasets.builder import collate
from opentad.models import build_detector
from opentad.models.detectors.single_stage import SingleStageDetector
from tools.bata.replay_phystime_decode_cross import (
    array_sha256,
    canonical_sha256,
    decode_axis,
    sha256_file,
)
from tools.bata.run_phystime_p0_fullprecision_gate import (
    build_dataset_manifest,
    coordinate_modes_from_config,
    inference_semantic_payload,
)


SOURCE_COMMIT = "0dc5851a8feb12b97d16bdb5ea8fc60e9273d132"
SOURCE_TREE = "bddc9b9386604d00d213275a47ce7997b35d3f4c"
P0_RUNTIME_COMMIT = "c2cfcfa2470f9f1e0b9d10e397480f6c66aeaf2c"
P0_RUNTIME_TREE = "0b78dd402e8997239ef9d1b4b4cd8bfa4f7a6338"
P0_SUITE_SHA256 = (
    "afb3e300424a57eb590a21129217e040677dc875fdede3be344352dc2bd268e7"
)
P0_GATE_SHA256 = (
    "1ca0efcdeb9f6343da076a00660675759358ac467074919a34d01c0d7c7250d9"
)
P0_DATASET_MANIFEST_SHA256 = (
    "1da0bca28f14ca2f1e4b2baf0f199dce18f4dd925e0f097a70a3fcc1c13eb1b2"
)
P0_VIDEOMAE_SHA256 = (
    "4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251"
)
EXPECTED_CONDITIONS = {
    "selected_online": ("selected_axis", "online"),
    "selected_ema": ("selected_axis", "ema"),
    "physical_online": ("physical_metric", "online"),
    "physical_ema": ("physical_metric", "ema"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real THUMOS one-window gate for decode cross replay."
    )
    parser.add_argument("--selected-config", required=True)
    parser.add_argument("--physical-config", required=True)
    parser.add_argument("--selected-checkpoint", required=True)
    parser.add_argument("--physical-checkpoint", required=True)
    parser.add_argument("--videomae-checkpoint", required=True)
    parser.add_argument("--selected-source-dir", required=True)
    parser.add_argument("--physical-source-dir", required=True)
    parser.add_argument("--p0-run-root", required=True)
    parser.add_argument("--preflight-manifest", required=True)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--focused-test-log", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-runtime-commit", required=True)
    parser.add_argument("--expected-runtime-tree", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path, description):
    path = Path(path)
    require(path.is_file(), f"missing {description}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def git_identity():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"],
        text=True,
    ).strip()
    clean = not subprocess.check_output(
        ["git", "status", "--porcelain"],
        text=True,
    ).strip()
    return commit, tree, clean


def normalize_state_dict(state):
    state = dict(state)
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    return state


def state_dict_sha256(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(tensor.shape), separators=(",", ":")).encode(
                "ascii"
            )
        )
        digest.update(b"\0")
        digest.update(
            tensor.reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes(order="C")
        )
    return digest.hexdigest()


def checkpoint_report(path):
    checkpoint = torch.load(path, map_location="cpu")
    require(
        isinstance(checkpoint, dict)
        and int(checkpoint.get("epoch", -1)) == 59,
        f"checkpoint is not epoch 59: {path}",
    )
    for key in ("state_dict", "state_dict_ema"):
        require(
            isinstance(checkpoint.get(key), dict) and checkpoint[key],
            f"checkpoint is missing {key}: {path}",
        )
        require(
            all(torch.isfinite(value).all() for value in checkpoint[key].values()),
            f"checkpoint {key} contains non-finite tensors",
        )
    require(
        "optimizer" not in checkpoint and "scheduler" not in checkpoint,
        "frozen checkpoint unexpectedly contains optimizer/scheduler",
    )
    return checkpoint, {
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "epoch": 59,
        "online_tensor_count": len(checkpoint["state_dict"]),
        "ema_tensor_count": len(checkpoint["state_dict_ema"]),
        "online_state_dict_sha256": state_dict_sha256(
            normalize_state_dict(checkpoint["state_dict"])
        ),
        "ema_state_dict_sha256": state_dict_sha256(
            normalize_state_dict(checkpoint["state_dict_ema"])
        ),
    }


def validate_source_dir(
    path,
    expected_variant,
    checkpoint_sha,
    dataset_manifest_sha256,
    videomae_checkpoint_sha256,
):
    path = Path(path).resolve()
    completion = read_json(path / "FULL_COMPLETE.json", "source completion")
    manifest = read_json(path / "run_manifest.json", "source manifest")
    require(completion.get("validation_pass") is True, "source completion failed")
    require(
        manifest.get("commit") == SOURCE_COMMIT
        and manifest.get("git_tree") == SOURCE_TREE
        and manifest.get("variant") == expected_variant,
        "source full60 identity mismatch",
    )
    require(
        completion.get("artifacts", {})
        .get("checkpoint", {})
        .get("sha256")
        == checkpoint_sha,
        "source checkpoint hash mismatch",
    )
    require(
        manifest.get("dataset_manifest_sha256")
        == dataset_manifest_sha256,
        "source dataset content hash mismatch",
    )
    require(
        manifest.get("pretrained_checkpoint_sha256")
        == videomae_checkpoint_sha256,
        "source VideoMAE content hash mismatch",
    )
    source_gate_path = Path(manifest["g1a_gate"]).resolve()
    source_gate = read_json(source_gate_path, "source G1a gate")
    require(
        source_gate.get("gate_pass") is True
        and source_gate.get("dataset_manifest_sha256")
        == dataset_manifest_sha256,
        "source G1a gate dataset binding mismatch",
    )
    result = {
        "path": str(path),
        "completion_sha256": sha256_file(path / "FULL_COMPLETE.json"),
        "manifest_sha256": sha256_file(path / "run_manifest.json"),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "videomae_checkpoint_sha256": videomae_checkpoint_sha256,
        "source_gate": {
            "path": str(source_gate_path),
            "sha256": sha256_file(source_gate_path),
        },
    }
    return result


def validate_p0(p0_root, checkpoint_reports):
    p0_root = Path(p0_root).resolve()
    suite_path = p0_root / "P0_SUITE_COMPLETE.json"
    suite = read_json(suite_path, "P0 suite")
    require(
        suite.get("schema_version")
        == "phystime_p0_fullprecision_suite_completion_v1"
        and suite.get("validation_pass") is True
        and suite.get("runtime_commit") == P0_RUNTIME_COMMIT
        and suite.get("runtime_tree") == P0_RUNTIME_TREE
        and suite.get("source_commit") == SOURCE_COMMIT
        and suite.get("source_tree") == SOURCE_TREE,
        "P0 suite identity/schema did not pass",
    )
    require(
        sha256_file(suite_path) == P0_SUITE_SHA256,
        "P0 suite content hash differs from the reviewed artifact",
    )
    p0_gate_path = Path(suite["gate"]["path"]).resolve()
    require(
        suite["gate"]["sha256"] == P0_GATE_SHA256
        and sha256_file(p0_gate_path) == P0_GATE_SHA256,
        "P0 gate content hash differs from the reviewed artifact",
    )
    p0_gate = read_json(p0_gate_path, "P0 gate")
    require(
        p0_gate.get("schema_version") == "phystime_p0_fullprecision_gate_v1"
        and p0_gate.get("gate_pass") is True
        and p0_gate["runtime"]["commit"] == P0_RUNTIME_COMMIT
        and p0_gate["runtime"]["git_tree"] == P0_RUNTIME_TREE
        and p0_gate["runtime"]["dataset_manifest_sha256"]
        == P0_DATASET_MANIFEST_SHA256
        and p0_gate["runtime"]["videomae_checkpoint_sha256"]
        == P0_VIDEOMAE_SHA256,
        "P0 gate provenance mismatch",
    )
    conditions = {}
    for variant, (arm, weights) in EXPECTED_CONDITIONS.items():
        completion_path = p0_root / variant / "P0_COMPLETE.json"
        completion = read_json(completion_path, f"{variant} P0 completion")
        suite_artifact = suite["completion_artifacts"][variant]
        require(
            Path(suite_artifact["path"]).resolve()
            == completion_path.resolve()
            and suite_artifact["sha256"] == sha256_file(completion_path),
            f"{variant} P0 completion differs from the reviewed suite",
        )
        require(
            completion.get("schema_version")
            == "phystime_p0_fullprecision_completion_v2"
            and completion.get("validation_pass") is True
            and completion.get("arm") == arm
            and completion.get("weights_source") == weights
            and completion.get("runtime_commit") == P0_RUNTIME_COMMIT
            and completion.get("runtime_tree") == P0_RUNTIME_TREE
            and completion.get("source_commit") == SOURCE_COMMIT
            and completion.get("source_tree") == SOURCE_TREE,
            f"{variant} P0 condition mismatch",
        )
        require(
            completion["artifacts"]["gate"]["sha256"] == P0_GATE_SHA256
            and Path(completion["artifacts"]["gate"]["path"]).resolve()
            == p0_gate_path,
            f"{variant} P0 gate binding mismatch",
        )
        checkpoint_key = (
            "selected_axis" if arm == "selected_axis" else "physical_metric"
        )
        require(
            completion["artifacts"]["checkpoint"]["sha256"]
            == checkpoint_reports[checkpoint_key]["sha256"],
            f"{variant} P0 checkpoint differs from full60",
        )
        conditions[variant] = {
            "path": str(completion_path.resolve()),
            "sha256": sha256_file(completion_path),
        }
    return {
        "run_root": str(p0_root),
        "suite": {
            "path": str(suite_path.resolve()),
            "sha256": sha256_file(suite_path),
        },
        "gate": {
            "path": str(p0_gate_path),
            "sha256": P0_GATE_SHA256,
        },
        "conditions": conditions,
    }


def validate_configs(selected_path, physical_path):
    selected = Config.fromfile(selected_path, lazy_import=False)
    physical = Config.fromfile(physical_path, lazy_import=False)
    reports = {}
    for name, cfg, expected_axis in (
        ("selected_axis", selected, "uniform_rank_seconds"),
        ("physical_metric", physical, "physical_time_seconds"),
    ):
        capture = cfg.inference.phystime_decode_replay_capture
        require(capture.enabled is True, f"{name} capture is disabled")
        require(
            capture.train_axis == expected_axis
            and capture.expected_native_coordinate_mode == expected_axis,
            f"{name} capture axis mismatch",
        )
        require(
            capture.weights_source == "must_be_overridden",
            f"{name} config hard-codes a weight source",
        )
        require(
            cfg.post_processing.round_before_cross_window_nms is False
            and cfg.post_processing.round_after_cross_window_nms is False
            and cfg.post_processing.filter_invalid_proposals is True,
            f"{name} post-processing differs from P0 full precision",
        )
        coordinate_modes = coordinate_modes_from_config(cfg)
        require(
            set(coordinate_modes["pipelines"].values()) == {expected_axis},
            f"{name} dataset coordinate mode mismatch",
        )
        semantic_payload = inference_semantic_payload(cfg)
        p0_semantic_payload = dict(semantic_payload)
        normalized_inference = dict(p0_semantic_payload["inference"])
        normalized_inference.pop("phystime_decode_replay_capture", None)
        p0_semantic_payload["inference"] = normalized_inference
        reports[name] = {
            "canonical_config_sha256": canonical_sha256(cfg.to_dict()),
            "coordinate_modes": coordinate_modes,
            "inference_semantic_sha256": canonical_sha256(
                semantic_payload
            ),
            "p0_base_inference_semantic_sha256": canonical_sha256(
                p0_semantic_payload
            ),
            "dataset_bindings": {
                "annotation": str(
                    Path(cfg.evaluation.ground_truth_filename).resolve()
                ),
                "class_map": str(Path(cfg.dataset.train.class_map).resolve()),
                "train_videos": str(
                    Path(cfg.dataset.train.data_path).resolve()
                ),
                "test_videos": str(
                    Path(cfg.dataset.test.data_path).resolve()
                ),
            },
        }
    require(
        reports["selected_axis"]["dataset_bindings"]
        == reports["physical_metric"]["dataset_bindings"],
        "selected and physical replay configs bind different datasets",
    )
    return selected, physical, reports


def validate_config_semantics_against_p0(config_reports, p0_report):
    p0_gate = read_json(
        p0_report["gate"]["path"],
        "reviewed P0 gate",
    )
    for name, report in config_reports.items():
        reviewed_sha256 = p0_gate["runtime_configs"][name][
            "inference_semantic_sha256"
        ]
        require(
            report["p0_base_inference_semantic_sha256"]
            == reviewed_sha256,
            f"{name} inference semantics differ from the reviewed P0 config",
        )
        report["reviewed_p0_inference_semantic_sha256"] = reviewed_sha256


def move_batch(batch, device):
    moved = dict(batch)
    moved["inputs"] = batch["inputs"].to(device)
    moved["masks"] = batch["masks"].to(device)
    moved["metas"] = [dict(meta) for meta in batch["metas"]]
    return moved


def run_real_window(
    cfg,
    checkpoint,
    checkpoint_path,
    videomae_checkpoint,
    work_dir,
    seed,
    arm,
    weights_source,
):
    cfg = Config(cfg.to_dict())
    cfg.work_dir = str(Path(work_dir).resolve())
    use_ema = weights_source == "ema"
    cfg.solver.ema = use_ema
    cfg.model.backbone.custom.pretrain = str(
        Path(videomae_checkpoint).resolve()
    )
    cfg.inference.phystime_decode_replay_capture.weights_source = (
        weights_source
    )
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = build_dataset(cfg.dataset.test)
    sample = None
    axis_delta = None
    sample_index = None
    for index in range(min(len(dataset), 8)):
        candidate = dataset[index]
        meta = candidate["metas"]
        uniform = np.asarray(
            meta["phystime_uniform_rank_timestamps_sec"],
            dtype=np.float32,
        )
        physical = np.asarray(
            meta["phystime_native_token_timestamps_sec"],
            dtype=np.float32,
        )
        delta = float(np.max(np.abs(uniform - physical)))
        if delta > 1.0e-6:
            sample = candidate
            axis_delta = delta
            sample_index = index
            break
    require(sample is not None, "real gate did not find an irregular window")
    batch = collate([sample])
    class_map = list(dataset.class_map)
    device = torch.device("cuda:0")
    model = build_detector(cfg.model).to(device)
    checkpoint_state_key = "state_dict_ema" if use_ema else "state_dict"
    state = normalize_state_dict(checkpoint[checkpoint_state_key])
    missing, unexpected = model.load_state_dict(state, strict=False)
    require(
        not missing and not unexpected,
        f"real gate checkpoint load mismatch: missing={missing[:5]}, "
        f"unexpected={unexpected[:5]}",
    )
    model.eval()

    collector = build_decode_replay_collector(
        model=model,
        cfg=cfg,
        external_cls=class_map,
        world_size=1,
        rank=0,
        evaluation_epoch=59,
    )
    batch = move_batch(batch, device)
    cfg.post_processing.sliding_window = True
    use_amp = bool(cfg.solver.get("amp", False))
    with torch.no_grad(), torch.cuda.amp.autocast(
        dtype=torch.float16,
        enabled=use_amp,
    ):
        direct = model(
            **batch,
            return_loss=False,
            infer_cfg=cfg.inference,
            post_cfg=cfg.post_processing,
            ext_cls=class_map,
        )
    collector.collect_latest_batch()
    artifact_record = collector.finalize()
    with np.load(artifact_record["artifact_path"], allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    raw_tensor_hashes_before = {
        name: array_sha256(array) for name, array in arrays.items()
    }
    capture_manifest = read_json(
        artifact_record["manifest_path"],
        "real gate capture manifest",
    )
    detector = SingleStageDetector()
    replay_results = {}
    mode_reports = {}
    native_axis = (
        "uniform_rank_seconds"
        if arm == "selected_axis"
        else "physical_time_seconds"
    )
    for axis_name in ("uniform_rank_seconds", "physical_time_seconds"):
        decoded = decode_axis(arrays, axis_name, native_axis)
        replay_results[axis_name] = detector.post_processing(
            (decoded["proposals"], decoded["scores"]),
            [
                {
                    "video_name": capture_manifest["windows"][0][
                        "video_name"
                    ],
                    "duration": capture_manifest["windows"][0]["duration"],
                    "prediction_time_unit": "seconds",
                }
            ],
            cfg.post_processing,
            class_map,
        )
        mode_reports[axis_name] = {
            "candidate_count": decoded["valid_candidate_count"],
            "native_point_reconstruction_max_abs_error": decoded[
                "native_point_reconstruction_max_abs_error"
            ],
            "native_proposal_reconstruction_max_abs_error": decoded[
                "native_proposal_reconstruction_max_abs_error"
            ],
            "canonical_result_sha256": canonical_sha256(
                replay_results[axis_name]
            ),
        }
    require(
        canonical_sha256(direct)
        == canonical_sha256(replay_results[native_axis]),
        "real gate native replay differs from direct inference",
    )
    require(
        mode_reports["uniform_rank_seconds"]["candidate_count"]
        == mode_reports["physical_time_seconds"]["candidate_count"]
        > 0,
        "real gate U/P candidate traversal mismatch",
    )
    raw_tensor_hashes_after = {
        name: array_sha256(array) for name, array in arrays.items()
    }
    require(
        raw_tensor_hashes_before == raw_tensor_hashes_after,
        "real gate raw tensors are not immutable",
    )
    observation_contract = {
        "observation_sequence_sha256": capture_manifest[
            "observation_sequence_sha256"
        ],
        "uniform_axis_sha256": capture_manifest["array_contract"][
            "uniform_axis_sec"
        ]["canonical_sha256"],
        "physical_axis_sha256": capture_manifest["array_contract"][
            "physical_axis_sec"
        ]["canonical_sha256"],
        "base_mask_sha256": capture_manifest["array_contract"]["base_mask"][
            "canonical_sha256"
        ],
        "native_mask_sha256": capture_manifest["array_contract"][
            "native_mask"
        ]["canonical_sha256"],
        "base_points_sha256": capture_manifest["array_contract"][
            "base_points"
        ]["canonical_sha256"],
        "class_map": capture_manifest["class_map"],
        "window_count": capture_manifest["window_count"],
        "candidate_count": capture_manifest["candidate_count"],
        "native_token_count": capture_manifest["native_token_count"],
    }
    axis_window_contract = {
        "native_axis": native_axis,
        "window_sequence_sha256": capture_manifest[
            "window_sequence_sha256"
        ],
    }
    result = {
        "sample_index": sample_index,
        "video_name": capture_manifest["windows"][0]["video_name"],
        "axis_max_abs_delta_sec": axis_delta,
        "arm": arm,
        "weights_source": weights_source,
        "checkpoint_state_key": checkpoint_state_key,
        "checkpoint_state_dict_sha256": state_dict_sha256(state),
        "native_axis": native_axis,
        "direct_canonical_result_sha256": canonical_sha256(direct),
        "mode_reports": mode_reports,
        "native_direct_exact_equivalence": True,
        "raw_tensor_hashes_before": raw_tensor_hashes_before,
        "raw_tensor_hashes_after": raw_tensor_hashes_after,
        "raw_tensors_immutable": True,
        "capture_artifact": {
            **artifact_record,
            "manifest_payload_sha256": canonical_sha256(capture_manifest),
        },
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "observation_contract": observation_contract,
        "axis_window_contract": axis_window_contract,
        "numeric_precision": {
            "numeric_semantics_version": capture_manifest[
                "numeric_semantics_version"
            ],
            "source_amp_enabled": use_amp,
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
            "score_sort_dtype": str(arrays["cls_scores"].dtype),
            "score_sort_device": "cpu",
            "geometry_compute_dtype": "float32",
            "geometry_compute_device": "cpu",
        },
    }
    del model, collector, batch
    torch.cuda.empty_cache()
    return result


def validate_real_window_contracts(real_windows, checkpoint_reports):
    require(
        set(real_windows) == set(EXPECTED_CONDITIONS),
        "four-condition real gate set mismatch",
    )
    expected_state_hashes = {
        "selected_online": checkpoint_reports["selected_axis"][
            "online_state_dict_sha256"
        ],
        "selected_ema": checkpoint_reports["selected_axis"][
            "ema_state_dict_sha256"
        ],
        "physical_online": checkpoint_reports["physical_metric"][
            "online_state_dict_sha256"
        ],
        "physical_ema": checkpoint_reports["physical_metric"][
            "ema_state_dict_sha256"
        ],
    }
    for variant, expected_state_sha256 in expected_state_hashes.items():
        require(
            real_windows[variant]["checkpoint_state_dict_sha256"]
            == expected_state_sha256,
            f"{variant} loaded the wrong checkpoint state",
        )
    shared_observation_contract = real_windows[
        "selected_online"
    ]["observation_contract"]
    require(
        all(
            record["observation_contract"] == shared_observation_contract
            for record in real_windows.values()
        ),
        "four-condition real gate sparse observation contract differs",
    )
    expected_native_axes = {
        "selected_axis": "uniform_rank_seconds",
        "physical_metric": "physical_time_seconds",
    }
    axis_window_contracts = {}
    for arm, online_variant, ema_variant in (
        ("selected_axis", "selected_online", "selected_ema"),
        ("physical_metric", "physical_online", "physical_ema"),
    ):
        online_contract = real_windows[online_variant][
            "axis_window_contract"
        ]
        ema_contract = real_windows[ema_variant]["axis_window_contract"]
        require(
            online_contract == ema_contract,
            f"{arm} online/EMA axis window contract differs",
        )
        require(
            online_contract.get("native_axis")
            == expected_native_axes[arm],
            f"{arm} native axis contract mismatch",
        )
        axis_window_contracts[arm] = online_contract
    require(
        axis_window_contracts["selected_axis"][
            "window_sequence_sha256"
        ]
        != axis_window_contracts["physical_metric"][
            "window_sequence_sha256"
        ],
        "selected and physical axis-specific window hashes unexpectedly match",
    )
    return shared_observation_contract


def main():
    args = parse_args()
    require(args.seed == 42, "gate seed must be 42")
    runtime_commit, runtime_tree, clean = git_identity()
    require(
        runtime_commit == args.expected_runtime_commit
        and runtime_tree == args.expected_runtime_tree
        and clean,
        "gate runtime snapshot mismatch or dirty tree",
    )
    preflight_path = Path(args.preflight_manifest).resolve()
    preflight = read_json(preflight_path, "decode-cross preflight")
    require(
        preflight.get("schema_version")
        == "phystime_decode_cross_preflight_v1"
        and preflight.get("validation_pass") is True
        and preflight["runtime"]["commit"] == runtime_commit
        and preflight["runtime"]["git_tree"] == runtime_tree
        and sha256_file(preflight_path)
        == args.expected_preflight_sha256,
        "decode-cross preflight identity/hash did not pass",
    )
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gate requires a Slurm-assigned GPU",
    )
    require(
        os.environ.get("SLURM_JOB_ID")
        and os.environ.get("SLURM_JOB_NAME"),
        "gate must run in a directly submitted Slurm allocation",
    )
    require(torch.cuda.is_available(), "CUDA is unavailable in the gate")
    selected_cfg, physical_cfg, config_reports = validate_configs(
        args.selected_config,
        args.physical_config,
    )
    selected_checkpoint, selected_checkpoint_report = checkpoint_report(
        args.selected_checkpoint
    )
    physical_checkpoint, physical_checkpoint_report = checkpoint_report(
        args.physical_checkpoint
    )
    checkpoint_reports = {
        "selected_axis": selected_checkpoint_report,
        "physical_metric": physical_checkpoint_report,
    }
    videomae_path = Path(args.videomae_checkpoint).resolve()
    require(videomae_path.is_file(), "VideoMAE checkpoint is missing")
    videomae_sha256 = sha256_file(videomae_path)
    require(
        videomae_sha256 == P0_VIDEOMAE_SHA256,
        "VideoMAE checkpoint differs from the reviewed P0 artifact",
    )
    dataset_manifest, dataset_manifest_sha256 = build_dataset_manifest(
        selected_cfg,
        selected_cfg.evaluation.ground_truth_filename,
    )
    require(
        dataset_manifest_sha256 == P0_DATASET_MANIFEST_SHA256,
        "current dataset content differs from the reviewed P0 dataset",
    )
    require(
        preflight["dataset"]["manifest_sha256"]
        == dataset_manifest_sha256
        and preflight["videomae_checkpoint"]["sha256"]
        == videomae_sha256
        and preflight["checkpoints"]["selected_axis"]["sha256"]
        == selected_checkpoint_report["sha256"]
        and preflight["checkpoints"]["physical_metric"]["sha256"]
        == physical_checkpoint_report["sha256"],
        "preflight inputs changed before the real gate",
    )
    for report in config_reports.values():
        report["dataset_manifest_sha256"] = dataset_manifest_sha256
    source_reports = {
        "selected_axis": validate_source_dir(
            args.selected_source_dir,
            "selected_axis",
            selected_checkpoint_report["sha256"],
            dataset_manifest_sha256,
            videomae_sha256,
        ),
        "physical_metric": validate_source_dir(
            args.physical_source_dir,
            "physical_metric",
            physical_checkpoint_report["sha256"],
            dataset_manifest_sha256,
            videomae_sha256,
        ),
    }
    require(
        source_reports["selected_axis"]["source_gate"]
        == source_reports["physical_metric"]["source_gate"],
        "source arms do not share the same full60 real gate",
    )
    p0_report = validate_p0(args.p0_run_root, checkpoint_reports)
    validate_config_semantics_against_p0(config_reports, p0_report)
    test_log_path = Path(args.focused_test_log)
    require(test_log_path.is_file(), "focused test log is missing")
    test_log = test_log_path.read_text(encoding="utf-8", errors="replace")
    require(
        "failed" not in test_log.lower()
        and "error" not in test_log.lower(),
        "focused tests contain a failure marker",
    )

    os.environ["PHYSTIME_SOURCE_COMMIT"] = SOURCE_COMMIT
    os.environ["PHYSTIME_SOURCE_TREE"] = SOURCE_TREE
    real_windows = {}
    gate_conditions = {
        "selected_online": (
            selected_cfg,
            selected_checkpoint,
            args.selected_checkpoint,
            "selected_axis",
            "online",
        ),
        "selected_ema": (
            selected_cfg,
            selected_checkpoint,
            args.selected_checkpoint,
            "selected_axis",
            "ema",
        ),
        "physical_online": (
            physical_cfg,
            physical_checkpoint,
            args.physical_checkpoint,
            "physical_metric",
            "online",
        ),
        "physical_ema": (
            physical_cfg,
            physical_checkpoint,
            args.physical_checkpoint,
            "physical_metric",
            "ema",
        ),
    }
    for variant, (
        cfg,
        checkpoint,
        checkpoint_path,
        arm,
        weights_source,
    ) in gate_conditions.items():
        os.environ["PHYSTIME_CHECKPOINT_PATH"] = str(
            Path(checkpoint_path).resolve()
        )
        real_windows[variant] = run_real_window(
            cfg,
            checkpoint,
            checkpoint_path,
            args.videomae_checkpoint,
            Path(args.work_dir) / variant,
            args.seed,
            arm,
            weights_source,
        )
    shared_observation_contract = validate_real_window_contracts(
        real_windows,
        checkpoint_reports,
    )
    report = {
        "schema_version": "phystime_decode_cross_real_gate_v1",
        "gate_pass": True,
        "completed_at_unix": time.time(),
        "runtime": {
            "commit": runtime_commit,
            "git_tree": runtime_tree,
            "tree_clean": clean,
            "cuda_device": torch.cuda.get_device_name(0),
            "execution": {
                "scheduler": "slurm",
                "job_id": os.environ["SLURM_JOB_ID"],
                "job_name": os.environ["SLURM_JOB_NAME"],
            },
        },
        "preflight": {
            "path": str(preflight_path),
            "sha256": sha256_file(preflight_path),
        },
        "source": {
            "commit": SOURCE_COMMIT,
            "git_tree": SOURCE_TREE,
            "arms": source_reports,
        },
        "configs": config_reports,
        "checkpoints": checkpoint_reports,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "dataset_manifest": dataset_manifest,
        "videomae_checkpoint": {
            "path": str(videomae_path),
            "sha256": videomae_sha256,
        },
        "p0": p0_report,
        "focused_tests": {
            "path": str(test_log_path.resolve()),
            "sha256": sha256_file(test_log_path),
        },
        "real_windows": real_windows,
        "shared_observation_contract": shared_observation_contract,
        "all_native_direct_exact_equivalence": all(
            record["native_direct_exact_equivalence"]
            for record in real_windows.values()
        ),
        "new_training": False,
        "frozen_epoch": 59,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
