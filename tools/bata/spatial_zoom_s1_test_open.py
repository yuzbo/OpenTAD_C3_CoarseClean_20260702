from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mmengine.config import Config

from tools.bata.spatial_zoom_s1_contract import (
    S1_RESOLUTIONS,
    S1_TRAINING_SEEDS,
    atomic_publish_json,
    canonical_sha256,
    sha256_file,
    validate_s1_manifest,
)
from tools.bata.spatial_zoom_s1_training import (
    S1_CANONICAL_STUDY_ROOT,
    validate_bound_s1_training_config,
)

S1_TEST_OPEN_SCHEMA = "spatial_zoom_s1_test_open_certificate_v6"
S1_GLOBAL_TEST_OPEN_MARKER_SCHEMA = "spatial_zoom_s1_global_test_open_marker_v3"
_PRECHECK_IDENTITY_KEYS = (
    "precheck_file_sha256",
    "precheck_sha256",
    "pretrained_checkpoint_sha256",
)
_EXPERIMENT_IDENTITY_KEYS = (
    "experiment_namespace",
    "canonical_experiment_root",
)


def _shared_precheck_identity(bindings: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    identities = [
        {key: binding.get(key) for key in _PRECHECK_IDENTITY_KEYS}
        for binding in bindings
    ]
    if not identities or any(
        not value for identity in identities for value in identity.values()
    ):
        raise ValueError("S1 test opening requires complete precheck identities")
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError(
            "S1 test opening selections do not share one precheck identity"
        )
    return {key: str(value) for key, value in identities[0].items()}


def _shared_experiment_identity(
    bindings: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    identities = [
        {key: binding.get(key) for key in _EXPERIMENT_IDENTITY_KEYS}
        for binding in bindings
    ]
    if not identities or any(
        not value for identity in identities for value in identity.values()
    ):
        raise ValueError("S1 test opening requires a canonical experiment namespace")
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("S1 test opening selections span experiment namespaces")
    return {key: str(value) for key, value in identities[0].items()}


def canonical_s1_study_root() -> Path:
    """Return the preregistered study root shared by every S1 rerun."""
    return Path(S1_CANONICAL_STUDY_ROOT).resolve()


def create_global_test_open_marker(
    certificate: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    certificate = json.loads(json.dumps(dict(certificate)))
    certificate_hash = certificate.get("certificate_sha256")
    unsigned_certificate = dict(certificate)
    unsigned_certificate.pop("certificate_sha256", None)
    if not certificate_hash or canonical_sha256(unsigned_certificate) != certificate_hash:
        raise ValueError("S1 test-open certificate self-hash mismatch")
    study_root = canonical_s1_study_root()
    if Path(certificate["canonical_study_root"]).resolve() != study_root:
        raise ValueError("S1 test-open certificate changed the canonical study root")
    marker_path = Path(certificate["global_test_open_marker_path"]).resolve()
    expected_path = (study_root / "test_open" / "test_open_issued.json").resolve()
    if marker_path != expected_path:
        raise ValueError("S1 global test-open marker path must be canonical")
    marker = {
        "schema_version": S1_GLOBAL_TEST_OPEN_MARKER_SCHEMA,
        "experiment_namespace": certificate["experiment_namespace"],
        "canonical_experiment_root": certificate["canonical_experiment_root"],
        "canonical_study_root": str(study_root),
        "manifest_sha256": certificate["manifest_sha256"],
        "code_commit": certificate["code_commit"],
        "precheck_file_sha256": certificate["precheck_file_sha256"],
        "precheck_sha256": certificate["precheck_sha256"],
        "pretrained_checkpoint_sha256": certificate["pretrained_checkpoint_sha256"],
        "certificate_sha256": certificate["certificate_sha256"],
        "certificate": certificate,
    }
    marker["marker_sha256"] = canonical_sha256(marker)
    atomic_publish_json(marker_path, marker)
    return marker_path, marker


def validate_global_test_open_marker(
    certificate: Mapping[str, Any],
) -> dict[str, Any]:
    study_root = canonical_s1_study_root()
    if Path(certificate["canonical_study_root"]).resolve() != study_root:
        raise ValueError("S1 test-open certificate changed the canonical study root")
    expected_path = (study_root / "test_open" / "test_open_issued.json").resolve()
    marker_path = Path(certificate["global_test_open_marker_path"]).resolve()
    if marker_path != expected_path or not marker_path.is_file():
        raise ValueError("S1 global test-open marker path is not canonical")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker_hash = marker.pop("marker_sha256", None)
    if not marker_hash or canonical_sha256(marker) != marker_hash:
        raise ValueError("S1 global test-open marker self-hash mismatch")
    marker["marker_sha256"] = marker_hash
    expected = {
        "schema_version": S1_GLOBAL_TEST_OPEN_MARKER_SCHEMA,
        "experiment_namespace": certificate["experiment_namespace"],
        "canonical_experiment_root": certificate["canonical_experiment_root"],
        "canonical_study_root": str(study_root),
        "manifest_sha256": certificate["manifest_sha256"],
        "code_commit": certificate["code_commit"],
        "precheck_file_sha256": certificate["precheck_file_sha256"],
        "precheck_sha256": certificate["precheck_sha256"],
        "pretrained_checkpoint_sha256": certificate["pretrained_checkpoint_sha256"],
        "certificate_sha256": certificate["certificate_sha256"],
        "certificate": json.loads(json.dumps(dict(certificate))),
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ValueError(f"S1 global test-open marker {key} mismatch")
    return marker


def recover_global_test_open_certificate(
    *,
    output_path: str | Path,
    manifest_path: str | Path,
    annotation_path: str | Path,
    selection_paths: Sequence[str | Path],
) -> dict[str, Any] | None:
    """Recover the exact certificate if the global commit published first."""

    marker_path = (
        canonical_s1_study_root() / "test_open" / "test_open_issued.json"
    ).resolve()
    if not marker_path.is_file():
        return None
    raw_marker = json.loads(marker_path.read_text(encoding="utf-8"))
    certificate = raw_marker.get("certificate")
    if not isinstance(certificate, dict):
        raise ValueError("S1 global marker has no recoverable certificate")
    validate_global_test_open_marker(certificate)
    if Path(certificate["manifest_path"]).resolve() != Path(manifest_path).resolve():
        raise ValueError("recovery manifest differs from the issued S1 certificate")
    if Path(certificate["annotation_path"]).resolve() != Path(annotation_path).resolve():
        raise ValueError("recovery annotation differs from the issued S1 certificate")
    expected_selections = {
        str(Path(value).resolve()) for value in selection_paths
    }
    issued_selections = {
        str(Path(row["selection_path"]).resolve())
        for row in certificate["selection_matrix"]
    }
    if expected_selections != issued_selections:
        raise ValueError("recovery selections differ from the issued S1 certificate")
    manifest_path = Path(manifest_path).resolve()
    annotation_path = Path(annotation_path).resolve()
    manifest = validate_s1_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        annotation_path=annotation_path,
    )
    if manifest["manifest_sha256"] != certificate.get("manifest_sha256"):
        raise ValueError("current manifest differs from the issued S1 certificate")
    if sha256_file(annotation_path) != certificate.get("annotation_sha256"):
        raise ValueError("current annotation differs from the issued S1 certificate")
    issued_rows = {
        str(Path(row["selection_path"]).resolve()): row
        for row in certificate["selection_matrix"]
    }
    for selection_path_text in sorted(expected_selections):
        row = issued_rows[selection_path_text]
        selection_path = Path(selection_path_text)
        if (
            not selection_path.is_file()
            or sha256_file(selection_path) != row.get("selection_file_sha256")
        ):
            raise ValueError(
                "current selection differs from the issued S1 certificate"
            )
        checkpoint_path = Path(row["checkpoint_path"])
        if (
            not checkpoint_path.is_file()
            or sha256_file(checkpoint_path) != row.get("checkpoint_sha256")
        ):
            raise ValueError(
                "current checkpoint differs from the issued S1 certificate"
            )
    output_path = Path(output_path).resolve()
    expected_output = (
        Path(certificate["canonical_experiment_root"])
        / "test_open"
        / "test_open_certificate.json"
    ).resolve()
    if output_path != expected_output:
        raise ValueError(f"S1 test-open certificate path must be canonical: {expected_output}")
    if output_path.is_file():
        observed = json.loads(output_path.read_text(encoding="utf-8"))
        if observed != certificate:
            raise ValueError("existing S1 certificate differs from the committed marker")
    else:
        atomic_publish_json(output_path, certificate)
    return certificate


def build_test_open_certificate(
    *,
    manifest_path: str | Path,
    annotation_path: str | Path,
    selection_paths: Sequence[str | Path],
) -> dict[str, Any]:
    from tools.bata.select_spatial_zoom_s1_checkpoint import (
        validate_checkpoint_selection,
    )

    manifest_path = Path(manifest_path).resolve()
    annotation_path = Path(annotation_path).resolve()
    manifest = validate_s1_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        annotation_path=annotation_path,
    )
    rows = []
    seen = set()
    protocol_fingerprint = None
    code_commit = None
    selection_bindings = []
    for selection_path_value in selection_paths:
        selection_path = Path(selection_path_value).resolve()
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        cfg = Config.fromfile(selection["bound_config_path"])
        binding = validate_bound_s1_training_config(cfg, seed=int(selection["seed"]))
        if not binding["formal_precheck_verified"]:
            raise RuntimeError(
                "S1 test opening requires full-precheck-bound selections"
            )
        checked = validate_checkpoint_selection(
            selection,
            config=cfg,
            seed=int(selection["seed"]),
            manifest=manifest,
            checkpoint_path=selection["selected"]["checkpoint_path"],
            protocol_fingerprint=binding["protocol_fingerprint"],
        )
        key = (int(checked["resolution"]), int(checked["seed"]))
        if key in seen:
            raise ValueError(f"duplicate S1 selection {key}")
        seen.add(key)
        protocol_fingerprint = protocol_fingerprint or checked["protocol_fingerprint"]
        if checked["protocol_fingerprint"] != protocol_fingerprint:
            raise ValueError("S1 selections do not share one protocol fingerprint")
        code_commit = code_commit or binding["code_commit"]
        if binding["code_commit"] != code_commit:
            raise ValueError("S1 selections do not share one Git commit")
        selection_bindings.append(binding)
        rows.append(
            {
                "resolution": key[0],
                "seed": key[1],
                "selection_path": str(selection_path),
                "selection_file_sha256": sha256_file(selection_path),
                "selection_sha256": checked["selection_sha256"],
                "checkpoint_path": checked["selected"]["checkpoint_path"],
                "checkpoint_sha256": checked["selected"]["checkpoint_sha256"],
                "checkpoint_epoch": int(checked["selected"]["epoch"]),
                "state_key": checked["selected"]["state_key"],
                "precheck_file_sha256": binding["precheck_file_sha256"],
                "precheck_sha256": binding["precheck_sha256"],
                "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
                "experiment_namespace": binding["experiment_namespace"],
                "canonical_experiment_root": binding["canonical_experiment_root"],
            }
        )
    expected = {
        (resolution, seed)
        for resolution in S1_RESOLUTIONS
        for seed in S1_TRAINING_SEEDS
    }
    if seen != expected:
        raise ValueError(
            f"S1 test opening requires the complete 3x3 selection matrix: {sorted(expected - seen)}"
        )
    precheck_identity = _shared_precheck_identity(selection_bindings)
    experiment_identity = _shared_experiment_identity(selection_bindings)
    study_root = canonical_s1_study_root()
    global_marker_path = (study_root / "test_open" / "test_open_issued.json").resolve()
    certificate: dict[str, Any] = {
        "schema_version": S1_TEST_OPEN_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "annotation_path": str(annotation_path),
        "annotation_sha256": manifest["annotation_sha256"],
        "sealed_test_split_hash": manifest["split_hashes"]["test"],
        "protocol_fingerprint": protocol_fingerprint,
        "code_commit": code_commit,
        **precheck_identity,
        **experiment_identity,
        "canonical_study_root": str(study_root),
        "global_test_open_marker_path": str(global_marker_path),
        "selection_matrix": sorted(
            rows, key=lambda row: (row["resolution"], row["seed"])
        ),
        "single_open_policy": True,
        "paper_claim_allowed": False,
    }
    certificate["certificate_sha256"] = canonical_sha256(certificate)
    return certificate


def validate_test_open_certificate(
    certificate: Mapping[str, Any],
    *,
    cfg: Config,
    seed: int,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    from tools.bata.select_spatial_zoom_s1_checkpoint import (
        validate_checkpoint_selection,
    )

    binding = validate_bound_s1_training_config(cfg, seed=int(seed))
    if not binding["formal_precheck_verified"]:
        raise RuntimeError("S1 test-open certificate requires a bound full precheck")
    checked = json.loads(json.dumps(dict(certificate)))
    expected_hash = checked.pop("certificate_sha256", None)
    if not expected_hash or canonical_sha256(checked) != expected_hash:
        raise ValueError("S1 test-open certificate hash mismatch")
    checked["certificate_sha256"] = expected_hash
    if checked.get("schema_version") != S1_TEST_OPEN_SCHEMA:
        raise ValueError("unsupported S1 test-open certificate schema")
    expected = {
        "manifest_path": binding["manifest_path"],
        "manifest_sha256": binding["manifest_sha256"],
        "annotation_path": binding["annotation_path"],
        "annotation_sha256": binding["annotation_sha256"],
        "sealed_test_split_hash": validate_s1_manifest(
            json.loads(Path(binding["manifest_path"]).read_text(encoding="utf-8")),
            annotation_path=binding["annotation_path"],
        )["split_hashes"]["test"],
        "protocol_fingerprint": binding["protocol_fingerprint"],
        "code_commit": binding["code_commit"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "canonical_study_root": str(canonical_s1_study_root()),
        "global_test_open_marker_path": str(
            (
                canonical_s1_study_root() / "test_open" / "test_open_issued.json"
            ).resolve()
        ),
        "single_open_policy": True,
    }
    for key, value in expected.items():
        if checked.get(key) != value:
            raise ValueError(f"S1 test-open certificate {key} mismatch")
    validate_global_test_open_marker(checked)
    manifest = validate_s1_manifest(
        json.loads(Path(binding["manifest_path"]).read_text(encoding="utf-8")),
        annotation_path=binding["annotation_path"],
    )
    rows = checked.get("selection_matrix")
    if not isinstance(rows, list):
        raise ValueError("S1 test-open certificate has no selection matrix")
    expected_keys = {
        (resolution, training_seed)
        for resolution in S1_RESOLUTIONS
        for training_seed in S1_TRAINING_SEEDS
    }
    actual_keys = {(int(row["resolution"]), int(row["seed"])) for row in rows}
    if len(rows) != len(expected_keys) or actual_keys != expected_keys:
        raise ValueError("S1 test-open certificate selection matrix is incomplete")
    for row in rows:
        selection_path = Path(row["selection_path"])
        if not selection_path.is_file() or sha256_file(selection_path) != row.get(
            "selection_file_sha256"
        ):
            raise ValueError("S1 test-open selection artifact mismatch")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selection_cfg = Config.fromfile(selection["bound_config_path"])
        selection_binding = validate_bound_s1_training_config(
            selection_cfg, seed=int(row["seed"])
        )
        selected_checkpoint = Path(row["checkpoint_path"])
        validated_selection = validate_checkpoint_selection(
            selection,
            config=selection_cfg,
            seed=int(row["seed"]),
            manifest=manifest,
            checkpoint_path=selected_checkpoint,
            protocol_fingerprint=selection_binding["protocol_fingerprint"],
        )
        expected_row = {
            "resolution": int(validated_selection["resolution"]),
            "seed": int(validated_selection["seed"]),
            "selection_path": str(selection_path.resolve()),
            "selection_file_sha256": sha256_file(selection_path),
            "selection_sha256": validated_selection["selection_sha256"],
            "checkpoint_path": validated_selection["selected"]["checkpoint_path"],
            "checkpoint_sha256": validated_selection["selected"]["checkpoint_sha256"],
            "checkpoint_epoch": int(validated_selection["selected"]["epoch"]),
            "state_key": validated_selection["selected"]["state_key"],
            "precheck_file_sha256": selection_binding["precheck_file_sha256"],
            "precheck_sha256": selection_binding["precheck_sha256"],
            "pretrained_checkpoint_sha256": selection_binding[
                "pretrained_checkpoint_sha256"
            ],
            "experiment_namespace": selection_binding["experiment_namespace"],
            "canonical_experiment_root": selection_binding["canonical_experiment_root"],
        }
        if dict(row) != expected_row:
            raise ValueError("S1 test-open selection row does not match its evidence")
        for key in _PRECHECK_IDENTITY_KEYS:
            if row[key] != checked[key]:
                raise ValueError(f"S1 test-open selection row changed {key}")
    checkpoint_path = Path(checkpoint_path).resolve()
    matching = [
        row
        for row in rows
        if int(row["resolution"]) == int(binding["resolution"])
        and int(row["seed"]) == int(seed)
    ]
    if len(matching) != 1:
        raise ValueError("S1 test-open certificate has no unique run selection")
    row = matching[0]
    if Path(row["checkpoint_path"]).resolve() != checkpoint_path:
        raise ValueError("S1 test-open checkpoint is not the frozen gate selection")
    if row["checkpoint_sha256"] != sha256_file(checkpoint_path):
        raise ValueError("S1 test-open checkpoint hash mismatch")
    expected_state_key = (
        "state_dict_ema" if bool(cfg.solver.get("ema", False)) else "state_dict"
    )
    if row.get("state_key") != expected_state_key:
        raise ValueError("S1 test-open checkpoint state key mismatch")
    return checked


__all__ = [
    "S1_GLOBAL_TEST_OPEN_MARKER_SCHEMA",
    "S1_TEST_OPEN_SCHEMA",
    "_shared_precheck_identity",
    "build_test_open_certificate",
    "canonical_s1_study_root",
    "create_global_test_open_marker",
    "recover_global_test_open_certificate",
    "validate_global_test_open_marker",
    "validate_test_open_certificate",
]
