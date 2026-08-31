from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import stat
import tempfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs" / "methods" / "continuous_roi_s2_v2_2_exact_byte_search_manifest.json"
DEFAULT_PROTOCOL = ROOT / "docs" / "methods" / "continuous_roi_s2_v2_2_protocol.json"
MANIFEST_SCHEMA = "continuous_roi_s2_v2_2_exact_byte_search_manifest_v1"
INVENTORY_SCHEMA = "continuous_roi_s2_v2_2_exact_byte_recovery_inventory_v1"
TERMINAL_SCHEMA = "continuous_roi_s2_v2_2_exact_byte_recovery_terminal_v1"
TASK_ID = "ZOOMTOKEN-CONTINUOUS-ROI-S2-V2.2-EXACT-BYTE-PROVENANCE-CENSUS-AND-ALL-OR-NONE-RECOVERY-CLOSURE-v001"
PASS = "CONTINUOUS_ROI_S2_V2_2_ALL_18_EXACT_BYTES_RECOVERED_FOR_FRESH_PRO"
STOP = "STOP_CONTINUOUS_ROI_S2_HISTORICAL_REFERENCE_ROUTE_ARTIFACTS_UNRECOVERABLE"
EXPECTED_PROTOCOL_SHA256 = "644f0c5648e0f5be004db3a3e7240a8f24a3c1d561f933502bccb8dca200cb46"
IMPLEMENTATION_BASE = "10aed28659a08fa703def278fc0f5f1422dcad89"
EVIDENCE_BASE = "72965e22df8e25471ddc896dd46d2d856cce84f3"
REMOTE_BOUNDARY = PurePosixPath("/data/run01/sczc063/yuzibo")


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _self_hash(payload: Mapping[str, Any], key: str) -> str:
    core = copy.deepcopy(dict(payload))
    core.pop(key, None)
    return canonical_sha256(core)


def protocol_core_sha256(protocol: Mapping[str, Any]) -> str:
    core = copy.deepcopy(dict(protocol))
    core.pop("declared_protocol_sha256", None)
    return canonical_sha256(core)


def expected_artifacts_from_protocol(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for cell in protocol["frozen_training_identities"]["cells"]:
        common = {
            "family": cell["family"],
            "job_id": cell["job_id"],
            "seed": cell["seed"],
        }
        checkpoint_path = cell["checkpoint_relative_path"]
        sidecar_path = cell["checkpoint_sidecar_relative_path"]
        artifacts.extend(
            [
                {
                    **common,
                    "artifact_id": f"{cell['family']}-{cell['seed']}-checkpoint",
                    "kind": "checkpoint",
                    "filename": Path(checkpoint_path).name,
                    "original_relative_path": checkpoint_path,
                    "expected_sha256": cell["checkpoint_sha256"],
                },
                {
                    **common,
                    "artifact_id": f"{cell['family']}-{cell['seed']}-sidecar",
                    "kind": "sidecar",
                    "filename": Path(sidecar_path).name,
                    "original_relative_path": sidecar_path,
                    "expected_sha256": cell["checkpoint_sidecar_sha256"],
                },
            ]
        )
    return sorted(artifacts, key=lambda item: item["artifact_id"])


def load_and_validate_contract(
    manifest_path: Path, protocol_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_json(manifest_path)
    protocol = load_json(protocol_path)
    _require(manifest.get("schema_version") == MANIFEST_SCHEMA, "manifest schema changed")
    _require(manifest.get("task_id") == TASK_ID, "task identity changed")
    _require(manifest["builder_plan"]["scan_ordinal"] == "1/1", "scan ordinal changed")
    _require(
        manifest["builder_plan"]["candidate_content_read_before_freeze"] is False,
        "manifest was not frozen before candidate content access",
    )
    _require(
        manifest["builder_plan"]["result_informed_root_expansion_allowed"] is False,
        "result-informed root expansion is forbidden",
    )
    _require(manifest["implementation_base"] == IMPLEMENTATION_BASE, "implementation base changed")
    _require(manifest["evidence_base"] == EVIDENCE_BASE, "evidence base changed")
    _require(
        _self_hash(manifest, "manifest_sha256") == manifest.get("manifest_sha256"),
        "manifest self-hash mismatch",
    )
    _require(
        protocol_core_sha256(protocol)
        == protocol.get("declared_protocol_sha256")
        == manifest["protocol"]["declared_protocol_sha256"]
        == EXPECTED_PROTOCOL_SHA256,
        "frozen protocol identity changed",
    )
    expected = expected_artifacts_from_protocol(protocol)
    observed = sorted(manifest["expected_artifacts"], key=lambda item: item["artifact_id"])
    _require(observed == expected, "the exact 18-artifact query table changed")
    _require(len(observed) == 18, "the exact artifact count changed")
    _require(
        sum(item["kind"] == "checkpoint" for item in observed) == 9
        and sum(item["kind"] == "sidecar" for item in observed) == 9,
        "the checkpoint/sidecar matrix changed",
    )
    sources = manifest["scan_sources"]
    _require(
        [source["kind"] for source in sources]
        == ["payload_tree", "provenance_catalog"],
        "the finite source set changed",
    )
    _require(
        sources[0]["max_depth"] == 4
        and sources[0]["candidate_basenames"]
        == ["epoch_59.pth", "epoch_59.pth.metadata.json"],
        "payload-tree bounds changed",
    )
    _require(sources[1]["candidate_payload_allowed"] is False, "catalog text cannot be payload")
    for source in sources:
        _require(source["provenance_evidence"], f"{source['id']} lacks provenance evidence")
        source_root = PurePosixPath(source["root"])
        _require(source_root.is_absolute(), f"{source['id']} root is not absolute")
        _require(
            str(source_root).startswith(str(REMOTE_BOUNDARY) + "/"),
            f"{source['id']} root escapes the authorized boundary",
        )
    forbidden = {str(PurePosixPath(item)) for item in manifest["forbidden_roots"]}
    _require(str(REMOTE_BOUNDARY) in forbidden, "the broad storage root must remain forbidden")
    quarantine = PurePosixPath(manifest["quarantine"]["final_root"])
    canonical_root = PurePosixPath(protocol["frozen_training_identities"]["canonical_root"])
    _require(quarantine != canonical_root, "quarantine aliases the original campaign")
    _require(
        str(quarantine).startswith(str(REMOTE_BOUNDARY / "projects") + "/"),
        "quarantine escapes the authorized project boundary",
    )
    _require(
        manifest["quarantine"]["publish_only_after_all_18_exact_matches"] is True
        and manifest["quarantine"]["original_campaign_mutation_allowed"] is False,
        "quarantine all-or-none contract changed",
    )
    return manifest, protocol


def _parse_beijing(value: str) -> float:
    parsed = datetime.fromisoformat(value)
    _require(parsed.tzinfo is not None, "preexisting cutoff lacks timezone")
    return parsed.timestamp()


def _bounded_files(root: Path, max_depth: int) -> Iterable[tuple[Path, str]]:
    _require(root.is_dir(), f"source root is missing: {root}")
    _require(not root.is_symlink(), f"source root is a symlink: {root}")
    for current_text, directories, files in os.walk(root, followlinks=False):
        current = Path(current_text)
        relative_current = current.relative_to(root)
        depth = 0 if relative_current == Path(".") else len(relative_current.parts)
        directories[:] = sorted(
            name for name in directories if not (current / name).is_symlink()
        )
        if depth >= max_depth:
            directories[:] = []
        for name in sorted(files):
            path = current / name
            relative = path.relative_to(root)
            if len(relative.parts) <= max_depth and not path.is_symlink():
                yield path, relative.as_posix()


def _metadata_snapshot(root: Path, max_depth: int) -> dict[str, Any]:
    entries = []
    root_stat = root.stat()
    entries.append(
        {
            "path": ".",
            "kind": "directory",
            "mode": stat.S_IMODE(root_stat.st_mode),
            "mtime_ns": root_stat.st_mtime_ns,
        }
    )
    for path, relative in _bounded_files(root, max_depth):
        item_stat = path.stat()
        entries.append(
            {
                "path": relative,
                "kind": "file",
                "mode": stat.S_IMODE(item_stat.st_mode),
                "mtime_ns": item_stat.st_mtime_ns,
                "size_bytes": item_stat.st_size,
            }
        )
    return {
        "entry_count": len(entries),
        "metadata_sha256": canonical_sha256(entries),
    }


def precheck_sources(manifest: Mapping[str, Any]) -> dict[str, Any]:
    observations = []
    for source in manifest["scan_sources"]:
        root = Path(source["root"])
        _require(root.is_dir(), f"source root is missing: {root}")
        _require(not root.is_symlink(), f"source root is a symlink: {root}")
        cutoff = _parse_beijing(source["preexisting_before_beijing"])
        _require(root.stat().st_mtime <= cutoff, f"source is not proven preexisting: {root}")
        observation = {
            "id": source["id"],
            "kind": source["kind"],
            "root": str(root),
            "root_mtime_ns": root.stat().st_mtime_ns,
            "preexisting": True,
        }
        if source["kind"] == "provenance_catalog":
            catalog_files = []
            for item in source["exact_files"]:
                path = root / item["relative_path"]
                _require(path.is_file() and not path.is_symlink(), f"catalog file missing: {path}")
                actual = sha256_file(path)
                _require(actual == item["sha256"], f"catalog identity changed: {path}")
                catalog_files.append(
                    {"path": str(path), "size_bytes": path.stat().st_size, "sha256": actual}
                )
            observation["catalog_files"] = catalog_files
        observations.append(observation)
    return {
        "schema_version": "continuous_roi_s2_v2_2_exact_byte_precheck_v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "source_count": len(observations),
        "sources": observations,
        "candidate_payload_bytes_read_or_hashed": False,
        "precheck_ready": True,
    }


def scan_payload_sources(
    manifest: Mapping[str, Any], expected: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    candidates: list[dict[str, Any]] = []
    matches: dict[str, list[dict[str, Any]]] = {
        item["artifact_id"]: [] for item in expected
    }
    by_hash = {item["expected_sha256"]: item for item in expected}
    for source in manifest["scan_sources"]:
        if source["kind"] != "payload_tree":
            continue
        root = Path(source["root"])
        allowed = set(source["candidate_basenames"])
        for path, relative in _bounded_files(root, int(source["max_depth"])):
            if path.name not in allowed:
                continue
            actual = sha256_file(path)
            record = {
                "source_id": source["id"],
                "source_path": str(path),
                "relative_path": relative,
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "actual_sha256": actual,
                "provenance_valid": True,
            }
            candidates.append(record)
            target = by_hash.get(actual)
            if target is not None and target["filename"] == path.name:
                matches[target["artifact_id"]].append(record)
    return candidates, matches


def audit_catalogs(
    manifest: Mapping[str, Any], expected: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    reports = []
    needles = {
        item["artifact_id"]: (item["expected_sha256"], item["original_relative_path"])
        for item in expected
    }
    for source in manifest["scan_sources"]:
        if source["kind"] != "provenance_catalog":
            continue
        root = Path(source["root"])
        files = []
        for item in source["exact_files"]:
            path = root / item["relative_path"]
            actual = sha256_file(path)
            _require(actual == item["sha256"], f"catalog identity changed: {path}")
            text = path.read_text(encoding="utf-8")
            references = [
                artifact_id
                for artifact_id, (expected_sha, relative_path) in needles.items()
                if expected_sha in text or relative_path in text
            ]
            files.append(
                {
                    "path": str(path),
                    "sha256": actual,
                    "size_bytes": path.stat().st_size,
                    "exact_artifact_reference_ids": references,
                    "candidate_payload_allowed": False,
                }
            )
        reports.append({"source_id": source["id"], "files": files})
    return reports


def _publish_once(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_result_root(
    result_root: Path,
    inventory: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    result_root = result_root.resolve()
    _require(not result_root.exists(), f"formal result root already exists: {result_root}")
    result_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_root.with_name(f".{result_root.name}.{os.getpid()}.tmp")
    _require(not temporary.exists(), f"temporary result root already exists: {temporary}")
    temporary.mkdir()
    try:
        _publish_once(temporary / "recovery_inventory.json", inventory)
        _publish_once(temporary / "terminal_receipt.json", receipt)
        os.rename(temporary, result_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def materialize_quarantine(
    quarantine_root: Path,
    expected: Sequence[Mapping[str, Any]],
    matches: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    _require(not quarantine_root.exists(), f"quarantine already exists: {quarantine_root}")
    for item in expected:
        _require(
            bool(matches.get(item["artifact_id"])),
            f"all-or-none quarantine is missing {item['artifact_id']}",
        )
    quarantine_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = quarantine_root.with_name(f".{quarantine_root.name}.{os.getpid()}.tmp")
    _require(not temporary.exists(), f"temporary quarantine already exists: {temporary}")
    ledger = []
    try:
        for item in expected:
            source = Path(sorted(matches[item["artifact_id"]], key=lambda row: row["source_path"])[0]["source_path"])
            destination = temporary / item["original_relative_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            actual = sha256_file(destination)
            _require(actual == item["expected_sha256"], f"quarantine hash mismatch: {destination}")
            os.chmod(destination, 0o444)
            ledger.append(
                {
                    "artifact_id": item["artifact_id"],
                    "source_path": str(source),
                    "destination_path": str(quarantine_root / item["original_relative_path"]),
                    "expected_sha256": item["expected_sha256"],
                    "source_sha256": sha256_file(source),
                    "destination_sha256": actual,
                    "parity": True,
                }
            )
        quarantine_manifest = {
            "schema_version": "continuous_roi_s2_v2_2_exact_byte_quarantine_v1",
            "artifact_count": len(ledger),
            "all_or_none": True,
            "entries": ledger,
            "reconstruction_used": False,
        }
        quarantine_manifest["quarantine_manifest_sha256"] = _self_hash(
            quarantine_manifest, "quarantine_manifest_sha256"
        )
        manifest_path = temporary / "quarantine_manifest.json"
        manifest_path.write_text(
            json.dumps(quarantine_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(manifest_path, 0o444)
        for directory, subdirectories, _ in os.walk(temporary, topdown=False):
            for name in subdirectories:
                os.chmod(Path(directory) / name, 0o555)
        os.chmod(temporary, 0o555)
        os.rename(temporary, quarantine_root)
        return quarantine_manifest
    except BaseException:
        if temporary.exists():
            for path in sorted(temporary.rglob("*"), reverse=True):
                try:
                    os.chmod(path, 0o755 if path.is_dir() else 0o644)
                except OSError:
                    pass
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def run_formal(
    manifest: Mapping[str, Any], protocol: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = expected_artifacts_from_protocol(protocol)
    payload_source = next(
        source for source in manifest["scan_sources"] if source["kind"] == "payload_tree"
    )
    original_root = Path(payload_source["root"])
    before = _metadata_snapshot(original_root, int(payload_source["max_depth"]))
    candidates, matches = scan_payload_sources(manifest, expected)
    catalogs = audit_catalogs(manifest, expected)
    after = _metadata_snapshot(original_root, int(payload_source["max_depth"]))
    original_modified = before != after
    rows = []
    for item in expected:
        item_matches = sorted(matches[item["artifact_id"]], key=lambda row: row["source_path"])
        rows.append(
            {
                **dict(item),
                "match_count": len(item_matches),
                "matches": item_matches,
                "exact_match_found": bool(item_matches),
            }
        )
    checkpoint_matches = sum(
        row["kind"] == "checkpoint" and row["exact_match_found"] for row in rows
    )
    sidecar_matches = sum(
        row["kind"] == "sidecar" and row["exact_match_found"] for row in rows
    )
    all_exact = checkpoint_matches == 9 and sidecar_matches == 9
    blockers = []
    for row in rows:
        if not row["exact_match_found"]:
            blockers.append(f"MISSING_EXACT_BYTES::{row['artifact_id']}::{row['expected_sha256']}")
    if original_modified:
        blockers.append("ORIGINAL_CAMPAIGN_ROOT_METADATA_CHANGED_DURING_READ_ONLY_SCAN")
    quarantine_manifest = None
    if all_exact and not blockers:
        quarantine_manifest = materialize_quarantine(
            Path(manifest["quarantine"]["final_root"]), expected, matches
        )
    terminal = PASS if all_exact and not blockers else STOP
    inventory = {
        "schema_version": INVENTORY_SCHEMA,
        "manifest_sha256": manifest["manifest_sha256"],
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "scan_ordinal": "1/1",
        "candidate_file_count": len(candidates),
        "candidate_files": candidates,
        "artifact_rows": rows,
        "catalog_reports": catalogs,
        "checkpoint_matches": checkpoint_matches,
        "sidecar_matches": sidecar_matches,
        "all_sha256_exact": all_exact,
        "all_sources_in_frozen_search_manifest": True,
        "all_sources_preexisting_and_provenance_valid": True,
        "original_root_metadata_before": before,
        "original_root_metadata_after": after,
        "original_campaign_root_modified": original_modified,
        "reconstruction_used": False,
        "blockers": blockers,
    }
    inventory["inventory_sha256"] = _self_hash(inventory, "inventory_sha256")
    source_destination_hash_parity = bool(
        quarantine_manifest
        and all(entry["parity"] for entry in quarantine_manifest["entries"])
    )
    receipt = {
        "schema_version": TERMINAL_SCHEMA,
        "task_id": TASK_ID,
        "terminal_classification": terminal,
        "formal_invocation_ordinal": "1/1",
        "manifest_sha256": manifest["manifest_sha256"],
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "inventory_sha256": inventory["inventory_sha256"],
        "checkpoint_matches": checkpoint_matches,
        "sidecar_matches": sidecar_matches,
        "all_sha256_exact": all_exact,
        "all_sources_in_frozen_search_manifest": True,
        "all_sources_preexisting_and_provenance_valid": True,
        "source_destination_hash_parity": source_destination_hash_parity,
        "original_campaign_root_modified": original_modified,
        "reconstruction_used": False,
        "quarantine_published": quarantine_manifest is not None,
        "quarantine_root": manifest["quarantine"]["final_root"] if quarantine_manifest else None,
        "blockers": blockers,
        "training_run": False,
        "gpu_used": False,
        "model_forward_run": False,
        "raw_inference_run": False,
        "prediction_run": False,
        "metric_run": False,
        "cost_run": False,
        "performance_accessed": False,
        "official_test_opened": False,
        "fresh_pro_required": True,
    }
    receipt["receipt_sha256"] = _self_hash(receipt, "receipt_sha256")
    return inventory, receipt


def formal_failure_artifacts(
    manifest: Mapping[str, Any], error: Exception
) -> tuple[dict[str, Any], dict[str, Any]]:
    blocker = f"FORMAL_ACTION_INCOMPLETE::{type(error).__name__}::{error}"
    inventory = {
        "schema_version": INVENTORY_SCHEMA,
        "manifest_sha256": manifest["manifest_sha256"],
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "scan_ordinal": "1/1",
        "formal_action_incomplete": True,
        "candidate_file_count": 0,
        "candidate_files": [],
        "artifact_rows": [],
        "catalog_reports": [],
        "checkpoint_matches": 0,
        "sidecar_matches": 0,
        "all_sha256_exact": False,
        "all_sources_in_frozen_search_manifest": True,
        "all_sources_preexisting_and_provenance_valid": False,
        "original_campaign_root_modified": None,
        "reconstruction_used": False,
        "blockers": [blocker],
    }
    inventory["inventory_sha256"] = _self_hash(inventory, "inventory_sha256")
    receipt = {
        "schema_version": TERMINAL_SCHEMA,
        "task_id": TASK_ID,
        "terminal_classification": STOP,
        "formal_invocation_ordinal": "1/1",
        "manifest_sha256": manifest["manifest_sha256"],
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "inventory_sha256": inventory["inventory_sha256"],
        "checkpoint_matches": 0,
        "sidecar_matches": 0,
        "all_sha256_exact": False,
        "all_sources_in_frozen_search_manifest": True,
        "all_sources_preexisting_and_provenance_valid": False,
        "source_destination_hash_parity": False,
        "original_campaign_root_modified": None,
        "reconstruction_used": False,
        "quarantine_published": False,
        "quarantine_root": None,
        "formal_action_incomplete": True,
        "blockers": [blocker],
        "training_run": False,
        "gpu_used": False,
        "model_forward_run": False,
        "raw_inference_run": False,
        "prediction_run": False,
        "metric_run": False,
        "cost_run": False,
        "performance_accessed": False,
        "official_test_opened": False,
        "fresh_pro_required": True,
    }
    receipt["receipt_sha256"] = _self_hash(receipt, "receipt_sha256")
    return inventory, receipt


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Bounded all-or-none census of the frozen Continuous-RoI S2 exact-nine bytes"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--precheck-only", action="store_true")
    parser.add_argument("--result-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest, protocol = load_and_validate_contract(args.manifest, args.protocol)
    precheck = precheck_sources(manifest)
    if args.precheck_only:
        print(json.dumps(precheck, indent=2, sort_keys=True))
        return 0
    _require(args.result_root is not None, "formal action requires --result-root")
    _require(not args.result_root.exists(), f"formal result root already exists: {args.result_root}")
    exit_code = 0
    try:
        inventory, receipt = run_formal(manifest, protocol)
    except Exception as error:
        inventory, receipt = formal_failure_artifacts(manifest, error)
        exit_code = 2
    _publish_result_root(args.result_root, inventory, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
