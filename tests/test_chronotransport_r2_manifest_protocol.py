import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from opentad.models.chronotransport.actions import LayerGroup
from opentad.models.chronotransport.controls import (
    motion_topk_actions,
    r2_control_algorithm_identity,
    random_exact_count_actions,
    validate_r2_control_algorithm_identity,
)
from opentad.models.chronotransport.protocol import (
    build_r2_manifest,
    build_stage_b_exposure_artifact,
    canonical_sha256,
    canonical_json_bytes,
    manifest_exact_bytes,
    stage_b_exposure_matrix,
    stage_c_exposure_matrix,
    validate_r2_manifest,
    validate_stage_b_exposures,
    validate_stage_b_exposure_artifact,
    validate_stage_c_exposures,
)
from opentad.models.chronotransport.scheduler import (
    ScheduleLibrary,
    validate_r2_library_payload,
)
from tools.bata.run_chronotransport_stage_b_formal import (
    LegacyFormalRouteDisabledError,
    run as run_legacy_formal_stage_b,
)
from tools.bata.build_chronotransport_r2_manifest import (
    build_manifest_file,
    load_manifest_file,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _registry(*, annotation_sha256: str | None = None):
    return {
        "schema": "chronotransport-r2-label-free-media-registry-v1",
        "data_sha256": _sha("data-root"),
        "annotation_sha256": annotation_sha256 or _sha("annotations"),
        "records": [
            {
                "video_id": f"video_{index:03d}",
                "media_registry_id": f"thumos14/train/video_{index:03d}",
                "media_path": f"train/video_{index:03d}.mp4",
                "media_sha256": _sha(f"media-{index}"),
                "source_total_frames": 1000 + index,
                "fps": 25.0,
                "sampled_frame_indices": list(range(0, 1000 + index, 2)),
            }
            for index in range(200)
        ],
    }


def _config_identity():
    return {
        "schema": "chronotransport-r2-config-identity-v1",
        "config_sha256": _sha("resolved-config"),
        "snippet_stride": 2,
        "scale_factor": 1.0,
        "rounding": "floor",
        "clipping": "source_bounds",
    }


def test_manifest_is_exact_200_label_free_one_window_per_video_and_deeply_validated():
    registry = _registry()
    manifest = build_r2_manifest(registry, _config_identity())
    validate_r2_manifest(manifest, registry=registry, config_identity=_config_identity())

    assert manifest["protocol"] == "CT-P3R-3S-r2"
    assert manifest["population_size"] == 200
    assert [len(manifest["splits"][name]) for name in ("fit", "calibration", "evaluation")] == [140, 30, 30]
    assert len(manifest["windows"]) == 200
    assert len({row["video_id"] for row in manifest["windows"]}) == 200
    assert len({row["window_id"] for row in manifest["windows"]}) == 200
    assert all(len(row["sampled_frame_indices"]) == 768 for row in manifest["windows"])
    assert all(len(row["valid_mask"]) == 768 for row in manifest["windows"])
    assert len(manifest["manifest_sha256"]) == 64
    assert manifest_exact_bytes(manifest) == canonical_json_bytes(manifest) + b"\n"

    tampered = copy.deepcopy(manifest)
    tampered["windows"][0]["sampled_frame_indices"][0] += 1
    with pytest.raises(ValueError, match="fixed config window|window hash"):
        validate_r2_manifest(tampered)


def test_manifest_annotation_identity_never_changes_split_or_window_digest():
    first = build_r2_manifest(_registry(annotation_sha256=_sha("anno-a")), _config_identity())
    second = build_r2_manifest(_registry(annotation_sha256=_sha("anno-b")), _config_identity())
    assert first["splits"] == second["splits"]
    assert first["split_hashes"] == second["split_hashes"]
    assert [row["window_id"] for row in first["windows"]] == [row["window_id"] for row in second["windows"]]
    assert [row["window_sha256"] for row in first["windows"]] == [row["window_sha256"] for row in second["windows"]]
    assert first["manifest_sha256"] != second["manifest_sha256"]


def test_manifest_validator_rejects_noncanonical_nfc_tree_even_when_hash_normalizes():
    registry = _registry()
    registry["records"][0]["media_path"] = "train/vidéo.mp4"
    manifest = build_r2_manifest(registry, _config_identity())
    tampered = copy.deepcopy(manifest)
    target = next(row for row in tampered["windows"] if row["media_path"] == "train/vidéo.mp4")
    target["media_path"] = "train/vide\u0301o.mp4"
    with pytest.raises(ValueError, match="NFC-canonical"):
        validate_r2_manifest(tampered)


def test_manifest_validator_rejects_bool_and_float_integer_impersonators():
    manifest = build_r2_manifest(_registry(), _config_identity())
    bad_mask = copy.deepcopy(manifest)
    bad_mask["windows"][0]["valid_mask"][0] = 1
    with pytest.raises(ValueError, match="valid_mask.*boolean"):
        validate_r2_manifest(bad_mask)

    bad_start = copy.deepcopy(manifest)
    bad_start["windows"][0]["window_start"] = float(
        bad_start["windows"][0]["window_start"]
    )
    with pytest.raises(ValueError, match="window_start.*integer"):
        validate_r2_manifest(bad_start)


@pytest.mark.parametrize("forbidden", ["annotations", "gt_segments", "label", "detector_output"])
def test_manifest_builder_rejects_result_or_annotation_fields(forbidden):
    registry = _registry()
    registry["records"][0][forbidden] = []
    with pytest.raises(ValueError, match="forbidden"):
        build_r2_manifest(registry, _config_identity())


@pytest.mark.parametrize(
    "media_path",
    ["C:\\data\\video.mp4", "C:/data/video.mp4", "/abs/video.mp4", "train/../video.mp4", ""],
)
def test_manifest_registry_requires_canonical_relative_posix_media_paths(media_path):
    registry = _registry()
    registry["records"][0]["media_path"] = media_path
    with pytest.raises(ValueError, match="relative POSIX"):
        build_r2_manifest(registry, _config_identity())


@pytest.mark.parametrize("video_id", ["", "bad\x00video"])
def test_manifest_registry_rejects_empty_or_nul_video_ids(video_id):
    registry = _registry()
    registry["records"][0]["video_id"] = video_id
    with pytest.raises(ValueError, match="video_id"):
        build_r2_manifest(registry, _config_identity())


def test_standalone_manifest_validator_rejects_rehashed_noncanonical_media_path():
    manifest = build_r2_manifest(_registry(), _config_identity())
    tampered = copy.deepcopy(manifest)
    window = tampered["windows"][0]
    window["media_path"] = "/absolute/video.mp4"
    window_hash_payload = {
        key: value
        for key, value in window.items()
        if key not in {"annotation_sha256", "split", "window_id", "window_sha256"}
    }
    window["window_sha256"] = canonical_sha256(window_hash_payload)
    window["window_id"] = f"ct-r2-window-{window['window_sha256']}"
    old_window_id = tampered["splits"][window["split"]][0]
    assert old_window_id != window["window_id"]
    tampered["splits"][window["split"]][0] = window["window_id"]
    tampered["split_hashes"][window["split"]] = canonical_sha256(
        tampered["splits"][window["split"]]
    )
    unsigned = dict(tampered)
    unsigned.pop("manifest_sha256")
    tampered["manifest_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="relative POSIX"):
        validate_r2_manifest(tampered)


def test_standalone_manifest_rejects_rehashed_window_config_type_drift():
    manifest = build_r2_manifest(_registry(), _config_identity())
    tampered = copy.deepcopy(manifest)
    window = tampered["windows"][0]
    window["snippet_stride"] = float(window["snippet_stride"])
    window_hash_payload = {
        key: value
        for key, value in window.items()
        if key not in {"annotation_sha256", "split", "window_id", "window_sha256"}
    }
    old_window_id = window["window_id"]
    window["window_sha256"] = canonical_sha256(window_hash_payload)
    window["window_id"] = f"ct-r2-window-{window['window_sha256']}"
    split_rows = tampered["splits"][window["split"]]
    split_rows[split_rows.index(old_window_id)] = window["window_id"]
    tampered["split_hashes"][window["split"]] = canonical_sha256(split_rows)
    unsigned = dict(tampered)
    unsigned.pop("manifest_sha256")
    tampered["manifest_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="snippet_stride.*config identity"):
        validate_r2_manifest(tampered)


def test_manifest_builder_and_validator_fail_closed_on_cardinality_duplicate_and_missing_identity():
    registry = _registry()
    registry["records"] = registry["records"][:-1]
    with pytest.raises(ValueError, match="exactly 200"):
        build_r2_manifest(registry, _config_identity())

    registry = _registry()
    registry["records"][1]["video_id"] = registry["records"][0]["video_id"]
    with pytest.raises(ValueError, match="unique video"):
        build_r2_manifest(registry, _config_identity())

    config = _config_identity()
    del config["rounding"]
    with pytest.raises(ValueError, match="rounding"):
        build_r2_manifest(_registry(), config)


def test_manifest_rebuilds_source_sampling_vector_from_fixed_config():
    registry = _registry()
    registry["records"][0]["sampled_frame_indices"] = list(range(0, 1000, 3))
    with pytest.raises(ValueError, match="sampled_frame_indices.*fixed config"):
        build_r2_manifest(registry, _config_identity())

    bad_scale = _config_identity()
    bad_scale["scale_factor"] = 3
    with pytest.raises(ValueError, match="divisible"):
        build_r2_manifest(_registry(), bad_scale)


def test_stage_b_exposure_artifact_binds_exact_fit_order_and_all_hashes():
    manifest = build_r2_manifest(_registry(), _config_identity())
    fit_window_ids = manifest["splits"]["fit"]
    artifact = build_stage_b_exposure_artifact(fit_window_ids)
    validate_stage_b_exposure_artifact(artifact, fit_window_ids=fit_window_ids)
    assert tuple(artifact["matrices"]) == ("3407", "3408", "3409")
    assert all(len(rows) == 140 for rows in artifact["matrices"].values())
    assert artifact["matrices"]["3407"][0]["window_id"] == fit_window_ids[0]
    assert len(artifact["combined_matrix_sha256"]) == 64

    tampered = copy.deepcopy(artifact)
    tampered["matrices"]["3408"][3]["window_id"] = fit_window_ids[4]
    with pytest.raises(ValueError, match="window binding"):
        validate_stage_b_exposure_artifact(tampered, fit_window_ids=fit_window_ids)

    float_candidate = copy.deepcopy(artifact)
    float_candidate["matrices"]["3407"][0]["candidate"] = 0.0
    with pytest.raises(ValueError, match="candidate.*integer"):
        validate_stage_b_exposure_artifact(
            float_candidate, fit_window_ids=fit_window_ids
        )


def test_stage_b_exposure_validators_reject_float_integer_impersonators():
    primitive = stage_b_exposure_matrix()
    float_seed_keys = {
        float(seed): rows for seed, rows in primitive.items()
    }
    with pytest.raises(ValueError, match="seed order"):
        validate_stage_b_exposures(float_seed_keys)

    manifest = build_r2_manifest(_registry(), _config_identity())
    fit_window_ids = manifest["splits"]["fit"]
    for field, value in (
        ("seeds", [3407.0, 3408, 3409]),
        ("seed_offsets", {"3407": 0.0, "3408": 4, "3409": 8}),
        ("candidate_count", 16.0),
    ):
        artifact = build_stage_b_exposure_artifact(fit_window_ids)
        artifact[field] = value
        unsigned = dict(artifact)
        unsigned.pop("artifact_sha256")
        artifact["artifact_sha256"] = canonical_sha256(unsigned)
        with pytest.raises(ValueError, match="integer|seeds|offset|candidate count"):
            validate_stage_b_exposure_artifact(
                artifact, fit_window_ids=fit_window_ids
            )


def test_stage_c_validator_locks_shape_candidate_balance_and_cursor():
    matrix = stage_c_exposure_matrix()
    validate_stage_c_exposures(matrix, next_cursor={3407: 8400, 3408: 8400, 3409: 8400})
    tampered = {seed: list(rows) for seed, rows in matrix.items()}
    tampered[3407][1] = dict(tampered[3407][1], batch_position=0)
    with pytest.raises(ValueError, match="batch position"):
        validate_stage_c_exposures(tampered)
    with pytest.raises(ValueError, match="cursor"):
        validate_stage_c_exposures(matrix, next_cursor={3407: 8399, 3408: 8400, 3409: 8400})
    with pytest.raises(ValueError, match="cursor.*integer"):
        validate_stage_c_exposures(
            matrix, next_cursor={3407: 8400.0, 3408: 8400, 3409: 8400}
        )
    bool_row = {seed: list(rows) for seed, rows in matrix.items()}
    bool_row[3407][1] = dict(bool_row[3407][1], batch_position=True)
    with pytest.raises(ValueError, match="batch position.*integer"):
        validate_stage_c_exposures(bool_row)


def test_library_and_control_algorithm_identities_are_frozen_and_deeply_validated():
    groups = (LayerGroup(0, 4), LayerGroup(4, 8), LayerGroup(8, 12))
    payload = ScheduleLibrary.r2(layer_groups=groups).canonical_payload()
    validate_r2_library_payload(payload)
    assert payload["layer_groups"] == [[0, 4], [4, 8], [8, 12]]
    assert [candidate["name"] for candidate in payload["candidates"]][-1] == "dense"

    tampered = copy.deepcopy(payload)
    tampered["candidates"][0]["actions"][1][0] = 99
    with pytest.raises(ValueError, match="action"):
        validate_r2_library_payload(tampered)

    identity = r2_control_algorithm_identity()
    validate_r2_control_algorithm_identity(identity)
    assert identity["motion_topk"]["sha256"] == r2_control_algorithm_identity()["motion_topk"]["sha256"]
    assert identity["random"]["sha256"] == r2_control_algorithm_identity()["random"]["sha256"]

    tampered_identity = copy.deepcopy(identity)
    tampered_identity["random"]["periods"] = [2, 4]
    with pytest.raises(ValueError, match="control algorithm"):
        validate_r2_control_algorithm_identity(tampered_identity)

    float_identity = copy.deepcopy(identity)
    float_identity["motion_topk"]["periods"] = [2.0, 4.0, 8.0]
    float_identity["motion_topk"]["num_chunks"] = 48.0
    motion_unsigned = dict(float_identity["motion_topk"])
    motion_unsigned.pop("sha256")
    float_identity["motion_topk"]["sha256"] = canonical_sha256(motion_unsigned)
    root_unsigned = dict(float_identity)
    root_unsigned.pop("control_algorithms_sha256")
    float_identity["control_algorithms_sha256"] = canonical_sha256(root_unsigned)
    with pytest.raises(ValueError, match="control algorithm"):
        validate_r2_control_algorithm_identity(float_identity)

    float_library = copy.deepcopy(payload)
    float_library["num_chunks"] = 48.0
    float_library["num_groups"] = 3.0
    float_library["layer_groups"] = [[0.0, 4.0], [4.0, 8.0], [8.0, 12.0]]
    library_unsigned = dict(float_library)
    library_unsigned.pop("library_sha256")
    float_library["library_sha256"] = canonical_sha256(library_unsigned)
    with pytest.raises(ValueError, match="integer|layer-group|action shape"):
        validate_r2_library_payload(float_library)

    with pytest.raises(ValueError, match="exact layer groups"):
        ScheduleLibrary.r2(
            layer_groups=(LayerGroup(0, 1), LayerGroup(1, 2), LayerGroup(2, 12))
        )

    with pytest.raises(ValueError, match="48 clips"):
        ScheduleLibrary.r2(num_chunks=48.5, layer_groups=groups)
    with pytest.raises(ValueError, match="exact layer groups"):
        ScheduleLibrary.r2(
            layer_groups=(LayerGroup(0.0, 4.0), LayerGroup(4, 8), LayerGroup(8, 12))
        )


def test_formal_controls_reject_type_coercion_and_noncanonical_identity():
    with pytest.raises((TypeError, ValueError), match="period"):
        motion_topk_actions(torch.zeros(1, 48, 3), period=2.9)
    with pytest.raises((TypeError, ValueError), match="seed"):
        random_exact_count_actions("window", seed=3407.9, num_groups=3, period=2)
    with pytest.raises((TypeError, ValueError), match="num_groups"):
        random_exact_count_actions("window", seed=3407, num_groups=3.9, period=2)
    with pytest.raises(ValueError, match="window_id"):
        random_exact_count_actions("bad\x00id", seed=3407, num_groups=3, period=2)


def test_legacy_six_schedule_formal_route_is_technically_unreachable():
    with pytest.raises(LegacyFormalRouteDisabledError, match="superseded by CT-P3R-3S-r2"):
        run_legacy_formal_stage_b(SimpleNamespace())


def test_manifest_cli_writer_is_atomic_canonical_and_emits_exact_byte_hash(tmp_path):
    registry_path = tmp_path / "registry.json"
    config_path = tmp_path / "config_identity.json"
    output_path = tmp_path / "manifest.json"
    registry_path.write_text(json.dumps(_registry()), encoding="utf-8")
    config_path.write_text(json.dumps(_config_identity()), encoding="utf-8")

    report = build_manifest_file(registry_path, config_path, output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path.read_bytes() == manifest_exact_bytes(payload)
    assert report["exact_bytes_sha256"] == hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert output_path.with_suffix(".json.sha256").read_text(encoding="ascii").strip() == report["exact_bytes_sha256"]
    assert not list(tmp_path.glob("*.tmp"))
    assert load_manifest_file(
        output_path, registry_path=registry_path, config_identity_path=config_path
    ) == payload

    sidecar = output_path.with_suffix(".json.sha256")
    sidecar.write_text("0" * 64 + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="sidecar"):
        load_manifest_file(
            output_path, registry_path=registry_path, config_identity_path=config_path
        )
    sidecar.write_text(report["exact_bytes_sha256"] + "\n", encoding="ascii")

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="exact canonical bytes"):
        load_manifest_file(
            output_path, registry_path=registry_path, config_identity_path=config_path
        )


def test_manifest_cli_rejects_duplicate_json_keys(tmp_path):
    registry_path = tmp_path / "registry.json"
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "manifest.json"
    registry = _registry()
    good = json.dumps(registry)
    duplicate = good[:-1] + ',"schema":"chronotransport-r2-label-free-media-registry-v1"}'
    registry_path.write_text(duplicate, encoding="utf-8")
    config_path.write_text(json.dumps(_config_identity()), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        build_manifest_file(registry_path, config_path, output_path)
