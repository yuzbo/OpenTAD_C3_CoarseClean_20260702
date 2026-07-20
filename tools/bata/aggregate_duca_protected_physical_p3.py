from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.duca_protected_physical_p3 import (
    DURATION_STRATA,
    aggregate_p3_rows,
)
from tools.bata.duca_protected_physical_training import (
    canonical_sha256,
    sha256_file,
)


SCHEMA = "duca_protected_physical_p3_aggregate_v1"
SHARD_SCHEMA = "duca_protected_physical_p3_shard_v1"
PROTOCOL_SCHEMA = "duca_protected_physical_protocol_manifest_v1"
WINDOWS_PER_STRATUM = 16
ROWS_PER_STRATUM = 192
TOTAL_WINDOWS = 48
TOTAL_SWAPS = 576
SWAPS_PER_WINDOW = 12
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260720


class ProtectedPhysicalP3AggregateFailure(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtectedPhysicalP3AggregateFailure(
            f"protected physical P3 aggregation failed: {message}"
        )


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value).strip().lower()
    _require(
        re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
        f"{label} must be an exact SHA256",
    )
    return digest


def _require_git_commit(value: Any, label: str) -> str:
    commit = str(value).strip().lower()
    _require(
        re.fullmatch(r"[0-9a-f]{40}", commit) is not None,
        f"{label} must be an exact Git commit",
    )
    return commit


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a JSON object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    _require(isinstance(value, list), f"{label} must be a JSON array")
    return value


def _require_int(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be an integer",
    )
    return int(value)


def _load_bound_json(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[dict[str, Any], Path, str]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    expected = _require_sha256(expected_sha256, f"{label} expected hash")
    actual = sha256_file(resolved)
    _require(actual == expected, f"{label} SHA256 mismatch")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtectedPhysicalP3AggregateFailure(
            f"{label} is not valid UTF-8 JSON: {exc}"
        ) from exc
    _require(isinstance(payload, dict), f"{label} must contain a JSON object")
    return payload, resolved, actual


def _validate_protocol_manifest(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[dict[str, Any], Path]:
    payload, resolved, _actual = _load_bound_json(
        path,
        expected_sha256,
        label=label,
    )
    _require(payload.get("schema") == PROTOCOL_SCHEMA, f"{label} schema mismatch")
    _require(payload.get("ok") is True, f"{label} did not pass")
    _require_git_commit(payload.get("git_commit"), f"{label} git_commit")
    _require_git_commit(payload.get("git_tree"), f"{label} git_tree")
    content_hash = _require_sha256(
        payload.get("manifest_content_sha256"),
        f"{label} manifest_content_sha256",
    )
    unsigned = dict(payload)
    unsigned.pop("manifest_content_sha256", None)
    _require(
        canonical_sha256(unsigned) == content_hash,
        f"{label} internal content hash mismatch",
    )
    return payload, resolved


def _window_key(row: Mapping[str, Any], label: str) -> tuple[str, int]:
    video_id = row.get("video_id")
    _require(
        isinstance(video_id, str) and bool(video_id),
        f"{label}.video_id must be a nonempty string",
    )
    return video_id, _require_int(row.get("window_start"), f"{label}.window_start")


def _swap_key(
    row: Mapping[str, Any],
    *,
    label: str,
) -> tuple[str, int, int, int]:
    video_id, window_start = _window_key(row, label)
    removed = _require_int(row.get("removed"), f"{label}.removed")
    incoming = _require_int(row.get("incoming"), f"{label}.incoming")
    _require(removed >= 0 and incoming >= 0, f"{label} has a negative swap index")
    _require(removed != incoming, f"{label} does not change the hard set")
    expected_sampling_hash = hashlib.sha256(
        (
            f"{video_id}|{window_start}|{removed}|{incoming}|"
            f"{BOOTSTRAP_SEED}"
        ).encode("utf-8")
    ).hexdigest()
    observed_sampling_hash = _require_sha256(
        row.get("sampling_sha256"),
        f"{label}.sampling_sha256",
    )
    _require(
        observed_sampling_hash == expected_sampling_hash,
        f"{label} deterministic swap hash mismatch",
    )
    return video_id, window_start, removed, incoming


def _validate_shard(
    payload: Mapping[str, Any],
    *,
    expected_stratum: str,
    protocol: Mapping[str, Any],
    label: str,
) -> tuple[list[dict[str, Any]], set[tuple[str, int]], set[tuple[str, int, int, int]]]:
    _require(payload.get("schema") == SHARD_SCHEMA, f"{label} schema mismatch")
    _require(payload.get("ok") is True, f"{label} did not pass")
    _require(
        payload.get("stratum") == expected_stratum,
        f"{label} stratum mismatch",
    )
    _require(
        payload.get("paper_claim_allowed") is False,
        f"{label} weakens the paper-claim contract",
    )
    _require(
        _require_int(payload.get("optimizer_step"), f"{label}.optimizer_step") == 0,
        f"{label} was produced after an optimizer step",
    )
    _require(
        payload.get("train_split_only") is True
        and payload.get("test_loader_built") is False
        and payload.get("checkpoint_written") is False,
        f"{label} is not a train-only, checkpoint-free audit",
    )

    runtime = _require_mapping(payload.get("runtime"), f"{label}.runtime")
    _require(
        _require_git_commit(runtime.get("git_commit"), f"{label} commit")
        == protocol["git_commit"],
        f"{label} commit differs from P0",
    )
    _require(
        _require_git_commit(runtime.get("git_tree"), f"{label} tree")
        == protocol["git_tree"],
        f"{label} tree differs from P0",
    )
    _require(
        _require_sha256(
            payload.get("protocol_manifest_sha256"),
            f"{label}.protocol_manifest_sha256",
        )
        == protocol["_file_sha256"],
        f"{label} is bound to another P0 manifest",
    )
    p3_population = _require_mapping(
        protocol.get("p3_population"),
        "P0 p3_population",
    )
    _require(
        _require_sha256(payload.get("config_sha256"), f"{label}.config_sha256")
        == _require_sha256(
            p3_population.get("config_sha256"),
            "P0 P3 config hash",
        ),
        f"{label} config differs from P0",
    )
    direct_pretrain = payload.get("adatad_pretrain")
    if direct_pretrain is not None:
        direct_pretrain = _require_mapping(
            direct_pretrain,
            f"{label}.adatad_pretrain",
        )
        _require(
            _require_sha256(
                direct_pretrain.get("sha256"),
                f"{label} pretrain hash",
            )
            == protocol["_pretrain_sha256"],
            f"{label} pretrain differs from P0",
        )
    for field in ("pretrain_sha256", "adatad_pretrain_sha256"):
        if payload.get(field) is not None:
            _require(
                _require_sha256(payload.get(field), f"{label}.{field}")
                == protocol["_pretrain_sha256"],
                f"{label} pretrain differs from P0",
            )

    windows = _require_list(payload.get("windows"), f"{label}.windows")
    rows = _require_list(payload.get("rows"), f"{label}.rows")
    _require(
        len(windows) == WINDOWS_PER_STRATUM,
        f"{label} must contain exactly {WINDOWS_PER_STRATUM} windows",
    )
    _require(
        len(rows) == ROWS_PER_STRATUM,
        f"{label} must contain exactly {ROWS_PER_STRATUM} rows",
    )
    _require(
        _require_sha256(payload.get("row_sha256"), f"{label}.row_sha256")
        == canonical_sha256(rows),
        f"{label} row content hash mismatch",
    )

    window_keys: set[tuple[str, int]] = set()
    for index, raw_window in enumerate(windows):
        window = _require_mapping(raw_window, f"{label}.windows[{index}]")
        _require(
            window.get("duration_stratum") == expected_stratum,
            f"{label}.windows[{index}] duration stratum mismatch",
        )
        key = _window_key(window, f"{label}.windows[{index}]")
        _require(key not in window_keys, f"{label} contains a duplicate window")
        window_keys.add(key)

    p0_windows = _require_list(
        p3_population.get("windows"),
        "P0 p3_population.windows",
    )
    registered_keys = {
        _window_key(
            _require_mapping(row, f"P0 p3_population.windows[{index}]"),
            f"P0 p3_population.windows[{index}]",
        )
        for index, row in enumerate(p0_windows)
        if _require_mapping(
            row,
            f"P0 p3_population.windows[{index}]",
        ).get("duration_stratum")
        == expected_stratum
    }
    _require(
        window_keys == registered_keys,
        f"{label} windows differ from the frozen P0 population",
    )

    swap_keys: set[tuple[str, int, int, int]] = set()
    rows_per_window: Counter[tuple[str, int]] = Counter()
    quartiles_per_window: dict[tuple[str, int], Counter[int]] = {}
    typed_rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(rows):
        row = _require_mapping(raw_row, f"{label}.rows[{index}]")
        _require(
            row.get("duration_stratum") == expected_stratum,
            f"{label}.rows[{index}] duration stratum mismatch",
        )
        key = _swap_key(row, label=f"{label}.rows[{index}]")
        _require(key not in swap_keys, f"{label} contains a duplicate swap")
        swap_keys.add(key)
        window_key = key[:2]
        _require(
            window_key in window_keys,
            f"{label}.rows[{index}] refers to an unregistered window",
        )
        rows_per_window[window_key] += 1
        quartile = _require_int(
            row.get("quartile"),
            f"{label}.rows[{index}].quartile",
        )
        _require(0 <= quartile < 4, f"{label}.rows[{index}] has invalid quartile")
        quartiles_per_window.setdefault(window_key, Counter())[quartile] += 1
        typed_rows.append(dict(row))

    _require(
        all(rows_per_window[key] == SWAPS_PER_WINDOW for key in window_keys),
        f"{label} must contain exactly 12 swaps per window",
    )
    _require(
        all(
            quartiles_per_window[key]
            == Counter({0: 3, 1: 3, 2: 3, 3: 3})
            for key in window_keys
        ),
        f"{label} quartile sampling is not 3/3/3/3 per window",
    )
    return typed_rows, window_keys, swap_keys


def _write_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with target.open("x", encoding="utf-8") as handle:
            created = True
            json.dump(
                dict(payload),
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if created and target.exists():
            target.unlink()
        raise
    return target


def aggregate_p3_shards(
    *,
    short_json: str | Path,
    short_sha256: str,
    medium_json: str | Path,
    medium_sha256: str,
    long_json: str | Path,
    long_sha256: str,
    output_json: str | Path,
) -> dict[str, Any]:
    specifications = {
        "short": (short_json, short_sha256),
        "medium": (medium_json, medium_sha256),
        "long": (long_json, long_sha256),
    }
    loaded: dict[str, dict[str, Any]] = {}
    input_paths: dict[str, str] = {}
    input_hashes: dict[str, str] = {}
    protocols: dict[str, dict[str, Any]] = {}

    for expected_stratum in DURATION_STRATA:
        shard_path, expected_hash = specifications[expected_stratum]
        payload, resolved, actual_hash = _load_bound_json(
            shard_path,
            expected_hash,
            label=f"{expected_stratum} P3 shard",
        )
        protocol_hash = _require_sha256(
            payload.get("protocol_manifest_sha256"),
            f"{expected_stratum} shard P0 hash",
        )
        protocol, protocol_path = _validate_protocol_manifest(
            payload.get("protocol_manifest_path"),
            protocol_hash,
            label=f"{expected_stratum} shard P0 manifest",
        )
        protocol["_file_sha256"] = protocol_hash
        protocol["_path"] = str(protocol_path)
        protocol["_pretrain_sha256"] = _require_sha256(
            _require_mapping(
                protocol.get("videomae_pretrain"),
                "P0 videomae_pretrain",
            ).get("sha256"),
            "P0 VideoMAE pretrain hash",
        )
        loaded[expected_stratum] = payload
        protocols[expected_stratum] = protocol
        input_paths[expected_stratum] = str(resolved)
        input_hashes[expected_stratum] = actual_hash

    protocol_hashes = {
        protocol["_file_sha256"] for protocol in protocols.values()
    }
    commits = {protocol["git_commit"] for protocol in protocols.values()}
    trees = {protocol["git_tree"] for protocol in protocols.values()}
    pretrain_hashes = {
        protocol["_pretrain_sha256"] for protocol in protocols.values()
    }
    p3_config_hashes = {
        _require_sha256(
            _require_mapping(
                protocol.get("p3_population"),
                "P0 p3_population",
            ).get("config_sha256"),
            "P0 P3 config hash",
        )
        for protocol in protocols.values()
    }
    _require(len(protocol_hashes) == 1, "shards use different P0 manifests")
    _require(len(commits) == 1, "shards use different Git commits")
    _require(len(trees) == 1, "shards use different Git trees")
    _require(len(pretrain_hashes) == 1, "shards use different pretrain hashes")
    _require(len(p3_config_hashes) == 1, "shards use different P3 config hashes")

    all_rows: list[dict[str, Any]] = []
    all_window_keys: set[tuple[str, int]] = set()
    all_swap_keys: set[tuple[str, int, int, int]] = set()
    observed_strata = set()
    for expected_stratum in DURATION_STRATA:
        payload = loaded[expected_stratum]
        observed_stratum = payload.get("stratum")
        _require(
            observed_stratum not in observed_strata,
            f"duplicate P3 stratum {observed_stratum!r}",
        )
        observed_strata.add(observed_stratum)
        rows, window_keys, swap_keys = _validate_shard(
            payload,
            expected_stratum=expected_stratum,
            protocol=protocols[expected_stratum],
            label=f"{expected_stratum} P3 shard",
        )
        _require(
            all_window_keys.isdisjoint(window_keys),
            "P3 windows are duplicated across strata",
        )
        _require(
            all_swap_keys.isdisjoint(swap_keys),
            "P3 swaps are duplicated across strata",
        )
        all_rows.extend(rows)
        all_window_keys.update(window_keys)
        all_swap_keys.update(swap_keys)

    _require(
        observed_strata == set(DURATION_STRATA),
        "P3 strata are not exactly short/medium/long",
    )
    _require(
        len(all_window_keys) == TOTAL_WINDOWS,
        f"P3 requires {TOTAL_WINDOWS} unique windows",
    )
    _require(
        len(all_rows) == TOTAL_SWAPS and len(all_swap_keys) == TOTAL_SWAPS,
        f"P3 requires {TOTAL_SWAPS} unique swaps",
    )

    aggregate = aggregate_p3_rows(
        all_rows,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    _require(
        aggregate.get("schema") == SCHEMA,
        "aggregate_p3_rows returned an unexpected schema",
    )
    protocol = protocols["short"]
    payload = {
        "schema": SCHEMA,
        "ok": aggregate.get("ok") is True,
        "status": (
            "p3_aggregate_passed"
            if aggregate.get("ok") is True
            else "p3_aggregate_failed"
        ),
        "git_commit": protocol["git_commit"],
        "git_tree": protocol["git_tree"],
        "protocol_manifest_path": protocol["_path"],
        "protocol_manifest_sha256": protocol["_file_sha256"],
        "protocol_manifest_content_sha256": protocol[
            "manifest_content_sha256"
        ],
        "pretrain_sha256": protocol["_pretrain_sha256"],
        "config_sha256": next(iter(p3_config_hashes)),
        "input_paths": input_paths,
        "input_hashes": input_hashes,
        "strata": list(DURATION_STRATA),
        "window_count": len(all_window_keys),
        "swap_count": len(all_swap_keys),
        "aggregate_content_sha256": canonical_sha256(aggregate),
        "aggregate": aggregate,
        "paper_claim_allowed": False,
    }
    _write_json_exclusive(output_json, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate the three frozen DUCA protected-physical P3 shards."
    )
    parser.add_argument(
        "--short-json",
        "--short-shard-json",
        "--short-shard",
        dest="short_json",
        required=True,
    )
    parser.add_argument(
        "--short-sha256",
        "--short-shard-sha256",
        dest="short_sha256",
        required=True,
    )
    parser.add_argument(
        "--medium-json",
        "--medium-shard-json",
        "--medium-shard",
        dest="medium_json",
        required=True,
    )
    parser.add_argument(
        "--medium-sha256",
        "--medium-shard-sha256",
        dest="medium_sha256",
        required=True,
    )
    parser.add_argument(
        "--long-json",
        "--long-shard-json",
        "--long-shard",
        dest="long_json",
        required=True,
    )
    parser.add_argument(
        "--long-sha256",
        "--long-shard-sha256",
        dest="long_sha256",
        required=True,
    )
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    try:
        result = aggregate_p3_shards(
            short_json=args.short_json,
            short_sha256=args.short_sha256,
            medium_json=args.medium_json,
            medium_sha256=args.medium_sha256,
            long_json=args.long_json,
            long_sha256=args.long_sha256,
            output_json=args.output_json,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "ok": False,
                    "status": "p3_aggregate_failed",
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
