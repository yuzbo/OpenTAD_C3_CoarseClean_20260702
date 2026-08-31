from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "docs" / "methods" / "continuous_roi_s2_v2_2_protocol.json"
SCHEMA = "continuous_roi_s2_reference_protocol_v2_2"
AUDIT_SCHEMA = "continuous_roi_s2_v2_2_known_answer_closure_v1"
READY = "CONTINUOUS_ROI_S2_V2_2_PROTOCOL_READY_FOR_FRESH_PRO"
STOP = "STOP_CONTINUOUS_ROI_S2_REFERENCE_ROUTE_BEFORE_INFERENCE"
HEX = set("0123456789abcdef")


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def protocol_core_sha256(payload: Mapping[str, Any]) -> str:
    core = copy.deepcopy(dict(payload))
    core.pop("declared_protocol_sha256", None)
    return canonical_sha256(core)


def atomic_write_json(path: Path, payload: Any) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not (set(value) - HEX)
    )


def _float_text(value: float) -> str:
    value = float(value)
    if value == 0.0:
        value = 0.0
    return format(value, ".17g")


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("logit input must be strictly between zero and one")
    return math.log(probability / (1.0 - probability))


def inverse_center_logit(center: float, size: float, *, atol: float = 1e-12) -> float:
    """Invert the bounded-center decoder, including the size==1 special case."""

    center = float(center)
    size = float(size)
    if not (math.isfinite(center) and math.isfinite(size)):
        raise ValueError("center and size must be finite")
    if size <= 0.0 or size > 1.0 + atol:
        raise ValueError("size must lie in (0, 1]")
    if abs(size - 1.0) <= atol:
        if abs(center - 0.5) > atol:
            raise ValueError("size-one support has the unique center 0.5")
        return 0.0
    probability = (center - 0.5 * size) / (1.0 - size)
    if probability <= 0.0 or probability >= 1.0:
        raise ValueError("center is not strictly inside the decoded support")
    return _logit(probability)


def _decode_size(sa: float, sr: float, geometry: Mapping[str, Any]) -> tuple[float, float]:
    area_min = float(geometry["area_min"])
    area_max = float(geometry["area_max"])
    ratio_min = float(geometry["ratio_min"])
    ratio_max = float(geometry["ratio_max"])
    source_aspect = float(geometry["source_aspect"])
    area = area_min + (area_max - area_min) * _sigmoid(sa)
    ratio = math.exp(
        math.log(ratio_min)
        + math.log(ratio_max / ratio_min) * _sigmoid(sr)
    )
    return (
        math.sqrt(area * ratio / source_aspect),
        math.sqrt(area * source_aspect / ratio),
    )


def _decode_center(logit: float, size: float) -> float:
    return 0.5 * size + (1.0 - size) * _sigmoid(logit)


def _filter_channel(values: Sequence[float], passes: int) -> list[float]:
    output = [float(value) for value in values]
    for _ in range(passes):
        previous = [output[0], *output[:-1]]
        following = [*output[1:], output[-1]]
        output = [
            0.25 * left + 0.50 * center + 0.25 * right
            for left, center, right in zip(previous, output, following)
        ]
    return output


def _interpolate_align_corners(values: Sequence[float], output_size: int) -> list[float]:
    values = [float(value) for value in values]
    if len(values) == 1:
        return values * output_size
    result = []
    for output_index in range(output_size):
        position = output_index * (len(values) - 1) / (output_size - 1)
        lower = int(math.floor(position))
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        result.append(values[lower] * (1.0 - weight) + values[upper] * weight)
    return result


def _candidate_tubelets(
    transformed_knots: Sequence[Sequence[float]],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    generator = protocol["candidate_generator"]
    geometry = protocol["geometry"]
    passes = int(generator["temporal_filter"]["passes"])
    clips = int(generator["tubelets"])
    filtered = [
        _filter_channel([row[channel] for row in transformed_knots], passes)
        for channel in range(4)
    ]
    center_x_intent = _interpolate_align_corners(filtered[0], clips)
    center_y_intent = _interpolate_align_corners(filtered[1], clips)
    variable_sa = _interpolate_align_corners(filtered[2], clips)
    variable_sr = _interpolate_align_corners(filtered[3], clips)
    anchor_logits = [float(value) for value in geometry["anchor_logits"]]
    fixed_sa = [anchor_logits[2]] * clips
    fixed_sr = [anchor_logits[3]] * clips
    tubelets: list[dict[str, Any]] = []
    for index in range(clips):
        fw, fh = _decode_size(fixed_sa[index], fixed_sr[index], geometry)
        vw, vh = _decode_size(variable_sa[index], variable_sr[index], geometry)
        guard_w = max(fw, vw)
        guard_h = max(fh, vh)
        common_x = 0.5 * guard_w + (1.0 - guard_w) * _sigmoid(center_x_intent[index])
        common_y = 0.5 * guard_h + (1.0 - guard_h) * _sigmoid(center_y_intent[index])
        fsx = inverse_center_logit(common_x, fw)
        fsy = inverse_center_logit(common_y, fh)
        vsx = inverse_center_logit(common_x, vw)
        vsy = inverse_center_logit(common_y, vh)
        fixed_center = (_decode_center(fsx, fw), _decode_center(fsy, fh))
        variable_center = (_decode_center(vsx, vw), _decode_center(vsy, vh))
        _require(
            max(abs(left - right) for left, right in zip(fixed_center, variable_center))
            <= 2e-15,
            "shared physical center construction is not exact",
        )
        tubelets.append(
            {
                "tubelet_ordinal": index,
                "common_center": [_float_text(common_x), _float_text(common_y)],
                "fixed_size": {
                    "logits": [_float_text(fsx), _float_text(fsy), _float_text(fixed_sa[index]), _float_text(fixed_sr[index])],
                    "box": [_float_text(common_x), _float_text(common_y), _float_text(fw), _float_text(fh)],
                },
                "variable_size": {
                    "logits": [_float_text(vsx), _float_text(vsy), _float_text(variable_sa[index]), _float_text(variable_sr[index])],
                    "box": [_float_text(common_x), _float_text(common_y), _float_text(vw), _float_text(vh)],
                },
            }
        )
    return tubelets


def build_candidate_manifest(protocol: Mapping[str, Any]) -> dict[str, Any]:
    generator = protocol["candidate_generator"]
    geometry = protocol["geometry"]
    expected_version = generator["implementation"]["version"]
    try:
        import torch
    except Exception as error:  # pragma: no cover - depends on host runtime
        raise RuntimeError("the frozen torch Sobol implementation is unavailable") from error
    _require(torch.__version__ == expected_version, "torch Sobol implementation version changed")
    dimension = int(generator["dimension"])
    engine = torch.quasirandom.SobolEngine(
        dimension=dimension,
        scramble=bool(generator["scramble"]),
        seed=int(generator["seed"]),
    )
    skip = int(generator["skip"])
    if skip:
        engine.fast_forward(skip)
    draws = engine.draw(
        int(generator["non_anchor_candidates"]), dtype=torch.float64
    ).reshape(
        int(generator["non_anchor_candidates"]),
        int(generator["knots"]),
        4,
    )
    anchor = [float(value) for value in geometry["anchor_logits"]]
    anchor_knots = [anchor[:] for _ in range(int(generator["knots"]))]
    candidates = [
        {
            "candidate_id": "candidate-000",
            "sobol_draw_ordinal": None,
            "tubelets": _candidate_tubelets(anchor_knots, protocol),
        }
    ]
    for draw_index, draw in enumerate(draws.tolist()):
        transformed = []
        for row in draw:
            transformed.append(
                [
                    3.0 * (2.0 * row[0] - 1.0),
                    3.0 * (2.0 * row[1] - 1.0),
                    anchor[2] + 1.5 * (2.0 * row[2] - 1.0),
                    anchor[3] + 1.5 * (2.0 * row[3] - 1.0),
                ]
            )
        candidates.append(
            {
                "candidate_id": f"candidate-{draw_index + 1:03d}",
                "sobol_draw_ordinal": draw_index,
                "tubelets": _candidate_tubelets(transformed, protocol),
            }
        )
    return {
        "schema_version": "continuous_roi_s2_v2_2_candidate_manifest_v1",
        "serialization": generator["serialization"],
        "generator_identity": {
            "implementation": generator["implementation"],
            "dimension": dimension,
            "scramble": generator["scramble"],
            "seed": generator["seed"],
            "dtype": generator["dtype"],
            "skip": skip,
            "draw_count": generator["non_anchor_candidates"],
            "candidate_order": generator["candidate_order"],
        },
        "candidates": candidates,
    }


def build_raw_population_manifest(
    protocol: Mapping[str, Any], development_database: Mapping[str, Any]
) -> dict[str, Any]:
    population = protocol["raw_reference_population"]
    window = population["windowing"]
    database = development_database.get("database")
    _require(isinstance(database, Mapping), "development database is missing")
    entries: list[dict[str, Any]] = []
    window_size = int(window["window_size"])
    feature_stride = int(window["feature_stride"])
    sample_stride = int(window["sample_stride"])
    snippet_stride = feature_stride * sample_stride
    window_stride = int(window_size * (1.0 - float(window["overlap_ratio"])))
    ioa_threshold = float(window["ioa_threshold"])
    for video_id in population["gate_video_ids"]:
        _require(video_id in database, f"gate video is missing: {video_id}")
        video = database[video_id]
        _require(video.get("subset") == "training", f"gate video subset changed: {video_id}")
        frame_count = int(video["frame"])
        duration = float(video["duration"])
        segments = []
        for item in video.get("annotations", []):
            if item.get("label") == "Ambiguous":
                continue
            start = int(float(item["segment"][0]) / duration * frame_count)
            end = int(float(item["segment"][1]) / duration * frame_count)
            segments.append((start, end))
        _require(segments, f"gate video has no usable development segments: {video_id}")
        snippet_count = len(range(0, frame_count, snippet_stride))
        last_window = False
        for dataset_window_index in range(max(1, snippet_count // window_stride)):
            start_index = dataset_window_index * window_stride
            end_index = start_index + window_size
            if end_index > snippet_count:
                end_index = snippet_count
                start_index = max(0, end_index - window_size)
                last_window = True
            start_frame = start_index * snippet_stride
            end_frame = (end_index - 1) * snippet_stride
            keep = False
            for segment_start, segment_end in segments:
                if segment_start < end_frame and segment_end > start_frame:
                    overlap = min(segment_end, end_frame) - max(segment_start, start_frame)
                    completeness = overlap / max(segment_end - segment_start, 1e-6)
                    if completeness > ioa_threshold:
                        keep = True
                        break
            if keep:
                entries.append(
                    {
                        "ordinal": len(entries),
                        "video_id": video_id,
                        "dataset_window_index": dataset_window_index,
                        "feature_start_index": start_index,
                        "feature_end_index_inclusive": end_index - 1,
                        "window_start_frame": start_frame,
                        "window_end_frame_inclusive": end_frame,
                        "valid_snippet_count": end_index - start_index,
                    }
                )
            if last_window:
                break
    return {
        "schema_version": "continuous_roi_s2_v2_2_raw_population_manifest_v1",
        "gate_split_sha256": population["gate_split_sha256"],
        "windowing": window,
        "entries": entries,
    }


def _walk_strings(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value


def assert_raw_object_graph_clean(payload: Any, protocol: Mapping[str, Any]) -> None:
    forbidden = tuple(token.lower() for token in protocol["privilege_boundary"]["raw_forbidden_tokens"])
    for text in _walk_strings(payload):
        lowered = text.lower()
        for token in forbidden:
            if token in lowered:
                raise ValueError(f"raw object graph contains forbidden token: {token}")


def seal_raw_receipt(raw_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "continuous_roi_s2_v2_2_raw_seal_v1",
        "raw_payload_sha256": canonical_sha256(raw_payload),
        "immutable_before_privileged_join": True,
    }


def privileged_join(
    raw_payload: Mapping[str, Any],
    raw_seal: Mapping[str, Any],
    preferred_ids: Mapping[str, str],
) -> dict[str, Any]:
    _require(raw_seal.get("immutable_before_privileged_join") is True, "raw receipt is not sealed")
    _require(
        raw_seal.get("raw_payload_sha256") == canonical_sha256(raw_payload),
        "raw payload changed after its immutable seal",
    )
    return {
        "schema_version": "continuous_roi_s2_v2_2_privileged_join_v1",
        "bound_raw_payload_sha256": raw_seal["raw_payload_sha256"],
        "preferred_candidate_ids": dict(preferred_ids),
    }


def validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    protocol = copy.deepcopy(dict(protocol))
    _require(protocol.get("schema_version") == SCHEMA, "schema changed")
    _require(protocol_core_sha256(protocol) == protocol.get("declared_protocol_sha256"), "protocol SHA-256 mismatch")
    scope = protocol["scope"]
    _require(scope["training_allowed"] is False, "v2.2 must not train")
    _require(scope["raw_inference_allowed"] is False, "v2.2 must not run raw inference")
    _require(scope["official_test_open_allowed"] is False, "official test must remain sealed")
    _require(scope["performance_access_allowed"] is False, "v2.2 is result blind")
    generator = protocol["candidate_generator"]
    _require(generator["implementation"] == {"name": "torch.quasirandom.SobolEngine", "version": "2.0.1"}, "Sobol implementation changed")
    _require(generator["dimension"] == 48 and generator["knots"] == 12 and generator["tubelets"] == 48, "candidate dimensions changed")
    _require(generator["scramble"] is True and generator["seed"] == 20260720, "Sobol scramble identity changed")
    _require(generator["dtype"] == "torch.float64", "Sobol dtype changed")
    _require(generator["skip"] == 0 and generator["non_anchor_candidates"] == 16, "Sobol draw sequence changed")
    _require(generator["candidate_order"] == "anchor_then_single_draw_order", "candidate order changed")
    _require(generator["serialization"] == "canonical-json-utf8-sort-keys-compact-float17-v1", "candidate serialization changed")
    _require(_is_sha256(generator["known_answer_manifest_sha256"]), "candidate known-answer SHA is missing")
    population = protocol["raw_reference_population"]
    _require(len(population["gate_video_ids"]) == 40, "gate population changed")
    _require(len(set(population["gate_video_ids"])) == 40, "gate IDs are not unique")
    _require(population["expected_window_count"] == 129, "raw window count changed")
    _require(_is_sha256(population["known_answer_manifest_sha256"]), "population known-answer SHA is missing")
    _require(population["windowing"] == {
        "feature_stride": 4,
        "sample_stride": 1,
        "window_size": 768,
        "overlap_ratio": "0.5",
        "ioa_threshold": "0.75",
        "last_window_realign": True,
        "keep_rule": "any_non_ambiguous_development_segment_completeness_strictly_gt_ioa_threshold",
    }, "raw windowing contract changed")
    identities = protocol["frozen_training_identities"]
    _require(len(identities["cells"]) == 9, "exact-nine identity count changed")
    expected_pairs = {(family, seed) for family in ("D160", "G96", "U128") for seed in (3407, 3408, 3409)}
    observed_pairs = {(item["family"], item["seed"]) for item in identities["cells"]}
    _require(observed_pairs == expected_pairs, "exact-nine identity matrix changed")
    expected_job_identities = {
        (family, seed): (
            str(1177668 + ordinal),
            f"crs2_77c2149a_{family.lower()}_{seed}",
        )
        for ordinal, (family, seed) in enumerate(
            (family, seed)
            for family in ("D160", "G96", "U128")
            for seed in (3407, 3408, 3409)
        )
    }
    for item in identities["cells"]:
        pair = (item["family"], item["seed"])
        _require(
            (item.get("job_id"), item.get("job_name")) == expected_job_identities[pair],
            f"{item['family']} seed {item['seed']} job identity changed",
        )
        for key in (
            "config_sha256",
            "completion_file_sha256",
            "completion_sha256",
            "checkpoint_sha256",
            "checkpoint_sidecar_sha256",
            "checkpoint_metadata_sha256",
        ):
            _require(_is_sha256(item[key]), f"{item['family']} seed {item['seed']} lacks {key}")
        _require(item["epochs"] == 60 and item["successful_updates"] == 4800, "training schedule identity changed")
        _require(item["checkpoint_consumer"] == "state_dict_ema", "checkpoint consumer changed")
    privilege = protocol["privilege_boundary"]
    _require("candidate_id" in privilege["raw_output_fields"], "enumerated candidate ID must remain result blind")
    _require("preferred_candidate_id" not in privilege["raw_output_fields"], "preferred ID leaked into raw output")
    _require(privilege["preferred_id_stage"] == "separate_cpu_join_after_immutable_raw_seal", "preferred ID stage changed")
    statistics = protocol["statistics"]
    _require(statistics["d0"]["candidate_count"] == 21, "D0 order changed")
    _require(statistics["short_q1"]["duration_seconds_lte"] == "1.5", "Short-Q1 cutoff changed")
    _require(statistics["bootstrap"]["replicates"] == 20000 and statistics["bootstrap"]["seed"] == 20260720, "bootstrap identity changed")
    _require(statistics["max_t"]["zero_standard_error_outcome"] == "NO_DECISION_INVALID_EVIDENCE", "max-T invalid outcome changed")
    return {
        "schema_version": "continuous_roi_s2_v2_2_static_audit_v1",
        "protocol_sha256": protocol_core_sha256(protocol),
        "static_protocol_valid": True,
        "training_authorized": False,
        "raw_inference_authorized": False,
        "official_test_open_allowed": False,
    }


def _check_file(path: Path, expected_sha256: str, label: str, blockers: list[str]) -> bool:
    if not path.is_file():
        blockers.append(f"MISSING::{label}::{path}")
        return False
    actual = sha256_file(path)
    if actual != expected_sha256:
        blockers.append(f"HASH_MISMATCH::{label}::{path}::{actual}")
        return False
    return True


def validate_external_identities(
    protocol: Mapping[str, Any],
    *,
    identity_root: Path,
    source_manifest: Path,
    development_database: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    identities = protocol["frozen_training_identities"]
    blockers: list[str] = []
    expected_root = identities["canonical_root"]
    if str(identity_root.resolve()) != expected_root:
        blockers.append(f"IDENTITY_ROOT_MISMATCH::{identity_root.resolve()}::{expected_root}")
    matrix = identity_root / identities["matrix_receipt_relative_path"]
    _check_file(matrix, identities["matrix_receipt_file_sha256"], "training_matrix_receipt", blockers)
    if matrix.is_file():
        matrix_payload = load_json(matrix)
        if matrix_payload.get("matrix_completion_sha256") != identities["matrix_receipt_internal_sha256"]:
            blockers.append("MATRIX_INTERNAL_SHA_MISMATCH")
        if matrix_payload.get("status") != "PASS_TRAINING_ONLY":
            blockers.append("MATRIX_STATUS_CHANGED")
    for cell in identities["cells"]:
        prefix = f"{cell['family']}:{cell['seed']}"
        config_path = identity_root / cell["config_relative_path"]
        completion_path = identity_root / cell["completion_relative_path"]
        _check_file(config_path, cell["config_sha256"], f"{prefix}:config", blockers)
        completion_ok = _check_file(
            completion_path,
            cell["completion_file_sha256"],
            f"{prefix}:completion",
            blockers,
        )
        if completion_ok:
            completion = load_json(completion_path)
            expected_embedded = {
                "completion_sha256": cell["completion_sha256"],
                "checkpoint_sha256": cell["checkpoint_sha256"],
                "checkpoint_sidecar_sha256": cell["checkpoint_sidecar_sha256"],
                "checkpoint_metadata_sha256": cell["checkpoint_metadata_sha256"],
                "successful_updates": 4800,
            }
            for key, value in expected_embedded.items():
                if completion.get(key) != value:
                    blockers.append(f"COMPLETION_FIELD_MISMATCH::{prefix}::{key}")
        _check_file(identity_root / cell["checkpoint_relative_path"], cell["checkpoint_sha256"], f"{prefix}:checkpoint", blockers)
        _check_file(identity_root / cell["checkpoint_sidecar_relative_path"], cell["checkpoint_sidecar_sha256"], f"{prefix}:checkpoint_sidecar", blockers)
    data = protocol["data_identities"]
    source_ok = _check_file(source_manifest, data["source_manifest_file_sha256"], "source_manifest", blockers)
    development_ok = _check_file(development_database, data["development_database_sha256"], "development_database", blockers)
    population_manifest = None
    if source_ok:
        source_payload = load_json(source_manifest)
        if source_payload.get("manifest_sha256") != data["source_manifest_semantic_sha256"]:
            blockers.append("SOURCE_MANIFEST_INTERNAL_SHA_MISMATCH")
        if source_payload.get("splits", {}).get("gate") != protocol["raw_reference_population"]["gate_video_ids"]:
            blockers.append("GATE_ID_ORDER_MISMATCH")
    if development_ok:
        population_manifest = build_raw_population_manifest(protocol, load_json(development_database))
        expected_population = protocol["raw_reference_population"]
        if len(population_manifest["entries"]) != expected_population["expected_window_count"]:
            blockers.append(f"RAW_POPULATION_COUNT_MISMATCH::{len(population_manifest['entries'])}")
        actual_population_sha = canonical_sha256(population_manifest)
        if actual_population_sha != expected_population["known_answer_manifest_sha256"]:
            blockers.append(f"RAW_POPULATION_SHA_MISMATCH::{actual_population_sha}")
        try:
            assert_raw_object_graph_clean(population_manifest, protocol)
        except ValueError as error:
            blockers.append(f"RAW_POPULATION_PRIVILEGE_VIOLATION::{error}")
    return {
        "identity_root": str(identity_root.resolve()),
        "source_manifest": str(source_manifest.resolve()),
        "development_database": str(development_database.resolve()),
        "exact_nine_checked": 9,
        "blockers": blockers,
        "identity_valid": not blockers,
    }, population_manifest


def run_closure(
    protocol: Mapping[str, Any],
    *,
    identity_root: Path | None = None,
    source_manifest: Path | None = None,
    development_database: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    static_audit = validate_protocol(protocol)
    candidate_manifest = build_candidate_manifest(protocol)
    candidate_sha = canonical_sha256(candidate_manifest)
    _require(candidate_sha == protocol["candidate_generator"]["known_answer_manifest_sha256"], f"candidate known-answer SHA mismatch: {candidate_sha}")
    assert_raw_object_graph_clean(candidate_manifest, protocol)
    identity_receipt = None
    population_manifest = None
    full_identity_requested = any(item is not None for item in (identity_root, source_manifest, development_database))
    if full_identity_requested:
        _require(all(item is not None for item in (identity_root, source_manifest, development_database)), "all external identity paths are required together")
        identity_receipt, population_manifest = validate_external_identities(
            protocol,
            identity_root=identity_root,
            source_manifest=source_manifest,
            development_database=development_database,
        )
    blockers = [] if identity_receipt is None else list(identity_receipt["blockers"])
    terminal = READY if identity_receipt is not None and not blockers else STOP if blockers else "STATIC_KNOWN_ANSWER_ONLY"
    receipt = {
        "schema_version": AUDIT_SCHEMA,
        "terminal_classification": terminal,
        "protocol_sha256": static_audit["protocol_sha256"],
        "candidate_manifest_sha256": candidate_sha,
        "candidate_count": len(candidate_manifest["candidates"]),
        "tubelets_per_candidate": len(candidate_manifest["candidates"][0]["tubelets"]),
        "identity_evidence_checked": identity_receipt is not None,
        "identity_receipt": identity_receipt,
        "raw_population_manifest_sha256": canonical_sha256(population_manifest) if population_manifest else None,
        "raw_population_count": len(population_manifest["entries"]) if population_manifest else None,
        "training_run": False,
        "raw_inference_run": False,
        "performance_accessed": False,
        "official_test_opened": False,
        "fresh_pro_required": terminal in (READY, STOP),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt, candidate_manifest, population_manifest


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Validate Continuous-RoI S2 v2.2 without inference or performance access")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--identity-root", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--development-database", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidate-manifest-output", type=Path)
    parser.add_argument("--raw-population-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = load_json(args.protocol)
    receipt, candidate_manifest, population_manifest = run_closure(
        protocol,
        identity_root=args.identity_root,
        source_manifest=args.source_manifest,
        development_database=args.development_database,
    )
    if args.output:
        atomic_write_json(args.output, receipt)
    if args.candidate_manifest_output:
        atomic_write_json(args.candidate_manifest_output, candidate_manifest)
    if args.raw_population_output:
        _require(population_manifest is not None, "raw population output requires external identity paths")
        atomic_write_json(args.raw_population_output, population_manifest)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
