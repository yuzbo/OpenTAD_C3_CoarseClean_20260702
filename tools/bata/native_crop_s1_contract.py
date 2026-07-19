from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from tools.bata.spatial_zoom_s1_contract import canonical_sha256, stable_id_hash


NATIVE_CROP_GEOMETRY_SCHEMA = "native_crop_s1_geometry_census_v1"
NATIVE_CROP_PRECHECK_SCHEMA = "native_crop_s1_vertical_slice_precheck_v1"
NATIVE_CROP_COST_SCHEMA = "native_crop_s1_full_stack_cost_schema_v1"
NATIVE_CROP_PRETRAINED_FILENAME = (
    "vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
)
NATIVE_CROP_PRETRAINED_SHA256 = (
    "4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251"
)
NATIVE_CROP_MANIFEST_SCHEMA = "spatial_zoom_s1_manifest_v4"
NATIVE_CROP_MANIFEST_SHA256 = (
    "10b14faac57d4631dfae93c9a7d14eb81b8dc308f0e80232469e5b7c974589ca"
)
NATIVE_CROP_MANIFEST_FILE_SHA256 = (
    "8e5a8901cb24b735750d5766405996dcac022b37f5a79fdbbdaa1f5479bf141d"
)
NATIVE_CROP_SOURCE_ANNOTATION_SHA256 = (
    "ee526d55aa4315a8adc68c501d0331f96a56ce16fa960f1d2ea182b9381ab9ad"
)
NATIVE_CROP_DEVELOPMENT_ANNOTATION_SHA256 = (
    "0985d3711ab31f404ff0be5a1ba75420796a6807d486410337078b38090bf749"
)
NATIVE_CROP_CLASS_MAP_SHA256 = (
    "a158b7c4c130ce74375a9b114160e2faae7a0221e605a0464a556fe082644f31"
)
NATIVE_CROP_SPLIT_COUNTS = {
    "fit": 160,
    "gate": 40,
    "test": 211,
}
NATIVE_CROP_SPLIT_HASHES = {
    "fit": "fa8f3a005f97c82a8e3dbb470eec7bce27b813e9d6a8908cbdcc6d5a2d1bb3af",
    "gate": "4d4e956aae298720529a2949c8aeb9fc90a1fc4730348bf5b5ba165d7df1f576",
    "test": "5f9adf639fbcff869075ac78f6aa26d9da14986199a7d5b2181127769600746e",
}
NATIVE_CROP_DEVELOPMENT_VIDEO_COUNT = 200
NATIVE_CROP_DEVELOPMENT_WINDOW_COUNT = 664
NATIVE_CROP_GATE_WINDOW_COUNT = 129
NATIVE_CROP_CHECKPOINT_STATE_TENSORS = 163
NATIVE_CROP_CHECKPOINT_CORE_TENSORS = 161
NATIVE_CROP_CHECKPOINT_CORE_NUMEL = 22_482_048

NATIVE_CROP_COST_STAGES = (
    "decode",
    "source_uint8_crop",
    "format_and_h2d",
    "global_backbone",
    "local_backbone",
    "fusion",
    "detector_projection_head",
    "postprocess_nms",
)


def quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty population")
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "min": ordered[0],
        "q25": percentile(0.25),
        "median": percentile(0.5),
        "q75": percentile(0.75),
        "max": ordered[-1],
    }


def build_cost_schema(*, global_size: int, local_size: int) -> dict:
    payload = {
        "schema_version": NATIVE_CROP_COST_SCHEMA,
        "measurement_status": "not_measured",
        "required_stages": list(NATIVE_CROP_COST_STAGES),
        "view_budget": {
            "global_pixels_per_frame": int(global_size) ** 2,
            "local_pixels_per_frame": int(local_size) ** 2,
            "total_view_pixels_per_frame": int(global_size) ** 2
            + int(local_size) ** 2,
            "shared_weights_do_not_imply_shared_compute": True,
        },
        "required_metrics": [
            "warm_serial_latency_p50_ms",
            "warm_serial_latency_p95_ms",
            "peak_gpu_memory_bytes",
            "gross_gpu_energy_joules",
        ],
        "teacher_search_cost_separate": True,
        "full_stack_claim_allowed": False,
    }
    payload["schema_sha256"] = canonical_sha256(payload)
    return payload


def validate_development_only_manifest(manifest: Mapping) -> dict:
    payload = json.loads(json.dumps(dict(manifest)))
    manifest_sha256 = payload.pop("manifest_sha256", None)
    if (
        not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or canonical_sha256(payload) != manifest_sha256
    ):
        raise ValueError("manifest_sha256 is missing, malformed, or inconsistent")
    if (
        manifest_sha256 != NATIVE_CROP_MANIFEST_SHA256
        or payload.get("schema_version") != NATIVE_CROP_MANIFEST_SCHEMA
        or payload.get("annotation_sha256")
        != NATIVE_CROP_SOURCE_ANNOTATION_SHA256
        or payload.get("official_test_sealed_until_protocol_freeze") is not True
    ):
        raise ValueError("Native-Crop manifest identity is not the frozen S1 manifest")
    payload["manifest_sha256"] = manifest_sha256
    splits = payload.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("geometry census manifest requires split mapping")
    if set(splits) != {"fit", "gate", "test"}:
        raise ValueError("geometry census expects the frozen fit/gate/test split identity")
    fit = sorted(set(map(str, splits["fit"])))
    gate = sorted(set(map(str, splits["gate"])))
    sealed_test = sorted(set(map(str, splits["test"])))
    if not fit or not gate or not sealed_test:
        raise ValueError("fit/gate/test identities must be non-empty")
    if set(fit) & set(gate):
        raise ValueError("fit and gate video identities overlap")
    if (set(fit) | set(gate)) & set(sealed_test):
        raise ValueError("development and sealed-test identities overlap")
    checked_splits = {
        "fit": fit,
        "gate": gate,
        "test": sealed_test,
    }
    for split_name, values in checked_splits.items():
        if len(values) != NATIVE_CROP_SPLIT_COUNTS[split_name]:
            raise ValueError(
                f"Native-Crop {split_name} split count changed: {len(values)}"
            )
        split_hash = stable_id_hash(values)
        if (
            split_hash != NATIVE_CROP_SPLIT_HASHES[split_name]
            or payload.get("split_hashes", {}).get(split_name) != split_hash
        ):
            raise ValueError(f"Native-Crop {split_name} split identity changed")
    return {
        "manifest_sha256": manifest_sha256,
        "fit": fit,
        "gate": gate,
        "sealed_test": sealed_test,
    }


def finalize_self_hash(payload: Mapping, hash_key: str) -> dict:
    checked = json.loads(json.dumps(dict(payload)))
    checked.pop(hash_key, None)
    checked[hash_key] = canonical_sha256(checked)
    return checked
