"""Capture immutable pre-decode tensors for PhysTime frozen cross replay."""

import hashlib
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch


SCHEMA_VERSION = "phystime_decode_replay_inputs_v2"
ARTIFACT_KIND = "frozen_pre_decode_actionformer_tensors"
NUMERIC_SEMANTICS_VERSION = "source_score_dtype_legacy_order_v1"
_TORCH_TO_NUMPY_DTYPE = {
    "torch.float16": np.dtype("float16"),
    "torch.float32": np.dtype("float32"),
    "torch.float64": np.dtype("float64"),
}
SUPPORTED_AXES = {
    "uniform_rank_seconds",
    "physical_time_seconds",
}
SUPPORTED_WEIGHT_SOURCES = {"online", "ema"}


def _cfg_get(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _canonical_json_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def decode_replay_effective_config_sha256(cfg):
    payload = copy.deepcopy(cfg.to_dict())
    inference = payload.get("inference")
    if isinstance(inference, dict):
        inference.pop("folder", None)
    post_processing = payload.get("post_processing")
    if isinstance(post_processing, dict):
        post_processing.pop("sliding_window", None)
    return _canonical_json_sha256(payload)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _atomic_write_json(path, payload):
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


def _atomic_write_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _score_dtype_contract(source_dtype, stored_dtype):
    """Keep the captured ranking representation identical to production."""
    try:
        expected = _TORCH_TO_NUMPY_DTYPE[str(source_dtype)]
    except KeyError as exc:
        raise RuntimeError(
            f"unsupported source score dtype for decode replay: {source_dtype!r}"
        ) from exc
    if stored_dtype != expected:
        raise RuntimeError(
            "decode replay score capture widened or narrowed the production "
            f"dtype: source={source_dtype}, stored={stored_dtype}"
        )
    return {
        "semantic_role": "ranking_scores",
        "ordering_sensitive": True,
        "source_torch_dtype": str(source_dtype),
        "stored_numpy_dtype": str(stored_dtype),
        "replay_torch_dtype": str(source_dtype),
        "allowed_casts_before_topk": [],
    }


def _unwrap_model(model):
    return getattr(model, "module", model)


def _get_capture_head(model):
    detector = _unwrap_model(model)
    head = getattr(detector, "rpn_head", None)
    if head is None:
        raise RuntimeError("decode replay capture requires detector.rpn_head")
    for method_name in (
        "enable_decode_replay_capture",
        "consume_decode_replay_state",
    ):
        if not callable(getattr(head, method_name, None)):
            raise RuntimeError(
                f"decode replay capture head is missing {method_name}"
            )
    return head


def build_decode_replay_collector(
    *,
    model,
    cfg,
    external_cls,
    world_size,
    rank,
    evaluation_epoch,
):
    capture_cfg = _cfg_get(
        _cfg_get(cfg, "inference"),
        "phystime_decode_replay_capture",
    )
    if not bool(_cfg_get(capture_cfg, "enabled", False)):
        return None
    if int(world_size) != 1 or int(rank) != 0:
        raise RuntimeError(
            "PhysTime decode replay capture requires exactly one DDP rank"
        )
    if evaluation_epoch is None:
        raise RuntimeError(
            "PhysTime decode replay capture requires a checkpoint epoch"
        )
    if not isinstance(external_cls, list) or not external_cls:
        raise RuntimeError(
            "PhysTime decode replay capture requires the fixed dataset class map"
        )
    if not all(isinstance(label, str) and label for label in external_cls):
        raise RuntimeError("decode replay class map contains an invalid label")
    train_axis = str(_cfg_get(capture_cfg, "train_axis", ""))
    native_mode = str(
        _cfg_get(capture_cfg, "expected_native_coordinate_mode", "")
    )
    weights_source = str(_cfg_get(capture_cfg, "weights_source", ""))
    if train_axis not in SUPPORTED_AXES:
        raise RuntimeError(f"unsupported decode replay train axis: {train_axis!r}")
    if native_mode != train_axis:
        raise RuntimeError(
            "decode replay native coordinate mode must match the training axis"
        )
    if weights_source not in SUPPORTED_WEIGHT_SOURCES:
        raise RuntimeError(
            f"unsupported decode replay weights source: {weights_source!r}"
        )
    for env_key in ("PHYSTIME_EXPECTED_COMMIT", "PHYSTIME_EXPECTED_TREE"):
        value = os.environ.get(env_key, "")
        if len(value) != 40:
            raise RuntimeError(
                f"decode replay requires a 40-character {env_key} binding"
            )
    for env_key in ("PHYSTIME_SOURCE_COMMIT", "PHYSTIME_SOURCE_TREE"):
        value = os.environ.get(env_key, "")
        if len(value) != 40:
            raise RuntimeError(
                f"decode replay requires a 40-character {env_key} binding"
            )
    checkpoint_path = Path(os.environ.get("PHYSTIME_CHECKPOINT_PATH", ""))
    if not checkpoint_path.is_file():
        raise RuntimeError(
            "decode replay requires PHYSTIME_CHECKPOINT_PATH to bind the "
            "frozen checkpoint"
        )

    head = _get_capture_head(model)
    head.enable_decode_replay_capture(True)
    return DecodeReplayCollector(
        model=model,
        cfg=cfg,
        capture_cfg=capture_cfg,
        class_map=external_cls,
        evaluation_epoch=int(evaluation_epoch),
    )


class DecodeReplayCollector:
    def __init__(
        self,
        *,
        model,
        cfg,
        capture_cfg,
        class_map,
        evaluation_epoch,
    ):
        self.model = model
        self.cfg = cfg
        self.capture_cfg = capture_cfg
        self.class_map = list(class_map)
        self.evaluation_epoch = int(evaluation_epoch)
        self.states = []
        self.metadata = []
        self.base_points = None
        self.level_lengths = None
        self.source_tensor_dtypes = None
        self.captured_tensor_bytes = 0
        self.max_in_memory_bytes = int(
            _cfg_get(capture_cfg, "max_in_memory_bytes", 8 * 1024**3)
        )
        if self.max_in_memory_bytes <= 0:
            raise RuntimeError(
                "decode replay max_in_memory_bytes must be positive"
            )

    def collect_latest_batch(self):
        head = _get_capture_head(self.model)
        state = head.consume_decode_replay_state()
        base_points = state.pop("base_points")
        level_lengths = tuple(int(value) for value in state.pop("level_lengths"))
        metadata = state.pop("metadata")
        source_tensor_dtypes = state.pop("source_tensor_dtypes")

        if self.source_tensor_dtypes is None:
            self.source_tensor_dtypes = dict(source_tensor_dtypes)
        elif source_tensor_dtypes != self.source_tensor_dtypes:
            raise RuntimeError(
                "decode replay source tensor dtypes changed between batches"
            )

        batch_bytes = sum(
            int(tensor.numel() * tensor.element_size())
            for tensor in state.values()
            if torch.is_tensor(tensor)
        )
        if self.base_points is None:
            batch_bytes += int(
                base_points.numel() * base_points.element_size()
            )
        projected_bytes = self.captured_tensor_bytes + batch_bytes
        projected_peak_bytes = 2 * projected_bytes
        if projected_peak_bytes > self.max_in_memory_bytes:
            raise RuntimeError(
                "decode replay in-memory capture budget exceeded: "
                f"estimated peak {projected_peak_bytes} > "
                f"{self.max_in_memory_bytes} bytes"
            )
        self.captured_tensor_bytes = projected_bytes

        if self.base_points is None:
            self.base_points = base_points
            self.level_lengths = level_lengths
        else:
            if level_lengths != self.level_lengths:
                raise RuntimeError(
                    "decode replay FPN topology changed between batches"
                )
            if not torch.equal(base_points, self.base_points):
                raise RuntimeError(
                    "decode replay base point grid changed between batches"
                )

        batch_size = int(state["cls_logits"].shape[0])
        if len(metadata) != batch_size:
            raise RuntimeError(
                "decode replay metadata count differs from batch size"
            )
        self.states.append(state)
        self.metadata.extend(metadata)

    def finalize(self):
        if not self.states or self.base_points is None:
            raise RuntimeError("decode replay capture produced no batches")

        arrays = {
            key: torch.cat([state[key] for state in self.states], dim=0).numpy()
            for key in (
                "cls_logits",
                "cls_scores",
                "reg_distances",
                "base_mask",
                "native_mask",
                "native_points",
                "native_proposals",
            )
        }
        arrays["base_points"] = self.base_points.numpy()
        offsets = [0]
        for length in self.level_lengths:
            offsets.append(offsets[-1] + int(length))
        arrays["level_offsets"] = np.asarray(offsets, dtype=np.int32)

        window_count = int(arrays["cls_logits"].shape[0])
        if len(self.metadata) != window_count:
            raise RuntimeError(
                "decode replay metadata count differs from captured windows"
            )
        q_count = int(arrays["cls_logits"].shape[1])
        class_count = int(arrays["cls_logits"].shape[2])
        if class_count != len(self.class_map):
            raise RuntimeError(
                "decode replay class map count differs from classification logits"
            )
        if offsets[-1] != q_count:
            raise RuntimeError("decode replay level offsets do not sum to Q")

        max_tokens = max(
            int(meta["phystime_native_token_count"]) for meta in self.metadata
        )
        uniform_axis = np.full(
            (window_count, max_tokens), np.nan, dtype=np.float32
        )
        physical_axis = np.full(
            (window_count, max_tokens), np.nan, dtype=np.float32
        )
        native_valid_count = np.zeros(window_count, dtype=np.int32)
        domain_sec = np.zeros((window_count, 2), dtype=np.float64)
        window_records = []
        for sample_idx, meta in enumerate(self.metadata):
            valid_count = int(meta["phystime_native_valid_count"])
            token_count = int(meta["phystime_native_token_count"])
            if token_count != max_tokens:
                raise RuntimeError(
                    "decode replay native token count changed between windows"
                )
            if valid_count <= 0 or valid_count > token_count:
                raise RuntimeError(
                    f"decode replay window {sample_idx} has invalid native count"
                )
            uniform_values = np.asarray(
                meta["phystime_uniform_rank_timestamps_sec"],
                dtype=np.float32,
            ).reshape(-1)
            physical_values = np.asarray(
                meta["phystime_native_token_timestamps_sec"],
                dtype=np.float32,
            ).reshape(-1)
            if (
                uniform_values.shape[0] != valid_count
                or physical_values.shape[0] != valid_count
            ):
                raise RuntimeError(
                    f"decode replay window {sample_idx} axis length mismatch"
                )
            domain_start = float(meta["phystime_g1a_axis_start_sec"])
            domain_end = float(meta["phystime_g1a_axis_end_sec"])
            if (
                not np.isfinite(domain_start)
                or not np.isfinite(domain_end)
                or domain_end <= domain_start
            ):
                raise RuntimeError(
                    f"decode replay window {sample_idx} has invalid domain"
                )
            for axis_name, values in (
                ("uniform", uniform_values),
                ("physical", physical_values),
            ):
                if (
                    not np.isfinite(values).all()
                    or (values.shape[0] > 1 and not np.all(np.diff(values) > 0))
                    or float(values[0]) < domain_start - 1.0e-6
                    or float(values[-1]) > domain_end + 1.0e-6
                ):
                    raise RuntimeError(
                        f"decode replay window {sample_idx} {axis_name} axis is invalid"
                    )
            uniform_axis[sample_idx, :valid_count] = uniform_values
            physical_axis[sample_idx, :valid_count] = physical_values
            native_valid_count[sample_idx] = valid_count
            domain_sec[sample_idx] = [domain_start, domain_end]

            selected_raw = [
                int(value)
                for value in meta["phystime_selected_raw_frame_indices"]
            ]
            selected_dense = meta.get(
                "phystime_raw_selected_dense_indices",
                meta.get("selected_dense_indices", []),
            )
            if str(meta["phystime_native_coordinate_mode"]) != str(
                _cfg_get(
                    self.capture_cfg,
                    "expected_native_coordinate_mode",
                    "",
                )
            ):
                raise RuntimeError(
                    f"decode replay window {sample_idx} native mode differs "
                    "from the configured training axis"
                )
            record = {
                "index": sample_idx,
                "video_name": str(meta["video_name"]),
                "duration": float(meta["duration"]),
                "prediction_time_unit": str(meta["prediction_time_unit"]),
                "window_start_frame": (
                    None
                    if "window_start_frame" not in meta
                    else float(meta["window_start_frame"])
                ),
                "native_coordinate_mode": str(
                    meta["phystime_native_coordinate_mode"]
                ),
                "native_valid_count": valid_count,
                "native_token_count": token_count,
                "raw_observation_count": int(
                    meta["phystime_raw_observation_count"]
                ),
                "selected_raw_frame_count": len(selected_raw),
                "selected_raw_frame_sha256": _canonical_json_sha256(
                    selected_raw
                ),
                "selected_dense_sha256": _canonical_json_sha256(
                    [float(value) for value in selected_dense]
                ),
                "domain_start_sec": domain_start,
                "domain_end_sec": domain_end,
            }
            observation_record = {
                key: record[key]
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
            record["observation_binding_sha256"] = (
                _canonical_json_sha256(observation_record)
            )
            record["window_binding_sha256"] = _canonical_json_sha256(record)
            window_records.append(record)

        arrays["uniform_axis_sec"] = uniform_axis
        arrays["physical_axis_sec"] = physical_axis
        arrays["native_valid_count"] = native_valid_count
        arrays["domain_sec"] = domain_sec

        for name, array in arrays.items():
            if array.dtype.kind in {"f", "c"} and name not in (
                "uniform_axis_sec",
                "physical_axis_sec",
            ):
                if not np.isfinite(array).all():
                    raise RuntimeError(
                        f"decode replay tensor {name} contains non-finite values"
                    )
        for name in ("uniform_axis_sec", "physical_axis_sec"):
            valid_values = np.concatenate(
                [
                    arrays[name][idx, : int(native_valid_count[idx])]
                    for idx in range(window_count)
                ]
            )
            if not np.isfinite(valid_values).all():
                raise RuntimeError(
                    f"decode replay tensor {name} contains non-finite valid values"
                )

        work_dir = Path(str(self.cfg.work_dir))
        artifact_name = str(
            _cfg_get(
                self.capture_cfg,
                "artifact_filename",
                "decode_replay_inputs.npz",
            )
        )
        manifest_name = str(
            _cfg_get(
                self.capture_cfg,
                "manifest_filename",
                "decode_replay_manifest.json",
            )
        )
        artifact_path = work_dir / artifact_name
        manifest_path = work_dir / manifest_name
        _atomic_write_npz(artifact_path, arrays)

        array_contract = {}
        for name, array in arrays.items():
            contract = {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "canonical_sha256": _array_sha256(array),
            }
            if name == "cls_scores":
                contract.update(
                    _score_dtype_contract(
                        self.source_tensor_dtypes["cls_scores"], array.dtype
                    )
                )
            array_contract[name] = contract
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "numeric_semantics_version": NUMERIC_SEMANTICS_VERSION,
            "evaluation_epoch": self.evaluation_epoch,
            "runtime": {
                "commit": os.environ.get("PHYSTIME_EXPECTED_COMMIT"),
                "git_tree": os.environ.get("PHYSTIME_EXPECTED_TREE"),
                "effective_config_sha256": (
                    decode_replay_effective_config_sha256(self.cfg)
                ),
            },
            "source": {
                "commit": os.environ.get("PHYSTIME_SOURCE_COMMIT"),
                "git_tree": os.environ.get("PHYSTIME_SOURCE_TREE"),
            },
            "checkpoint": {
                "path": str(
                    Path(
                        os.environ["PHYSTIME_CHECKPOINT_PATH"]
                    ).resolve()
                ),
                "sha256": _sha256_file(
                    os.environ["PHYSTIME_CHECKPOINT_PATH"]
                ),
            },
            "train_axis": str(
                _cfg_get(self.capture_cfg, "train_axis", "")
            ),
            "weights_source": str(
                _cfg_get(self.capture_cfg, "weights_source", "")
            ),
            "expected_native_coordinate_mode": str(
                _cfg_get(
                    self.capture_cfg,
                    "expected_native_coordinate_mode",
                    "",
                )
            ),
            "new_training": False,
            "artifact": {
                "path": str(artifact_path.resolve()),
                "sha256": _sha256_file(artifact_path),
            },
            "class_map": self.class_map,
            "window_count": window_count,
            "candidate_count": q_count,
            "class_count": class_count,
            "native_token_count": max_tokens,
            "level_lengths": list(self.level_lengths),
            "source_tensor_dtypes": self.source_tensor_dtypes,
            "source_amp_enabled": bool(
                _cfg_get(
                    _cfg_get(self.cfg, "solver", None),
                    "amp",
                    False,
                )
            ),
            "capture_memory": {
                "captured_tensor_bytes": self.captured_tensor_bytes,
                "estimated_peak_tensor_bytes": (
                    2 * self.captured_tensor_bytes
                ),
                "max_in_memory_bytes": self.max_in_memory_bytes,
                "within_budget": True,
            },
            "array_contract": array_contract,
            "windows": window_records,
        }
        manifest["window_sequence_sha256"] = _canonical_json_sha256(
            [record["window_binding_sha256"] for record in window_records]
        )
        manifest["observation_sequence_sha256"] = _canonical_json_sha256(
            [
                record["observation_binding_sha256"]
                for record in window_records
            ]
        )
        _atomic_write_json(manifest_path, manifest)
        head = _get_capture_head(self.model)
        head.enable_decode_replay_capture(False)
        return {
            "artifact_path": str(artifact_path.resolve()),
            "artifact_sha256": manifest["artifact"]["sha256"],
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": _sha256_file(manifest_path),
            "window_count": window_count,
            "candidate_count": q_count,
        }


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")
