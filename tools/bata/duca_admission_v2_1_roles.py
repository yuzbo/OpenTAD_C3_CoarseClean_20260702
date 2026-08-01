from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tools.bata.duca_admission_v2_1_hashing import (
    PROTOCOL_ID,
    canonical_text,
    domain_hash,
    domain_hash_hex,
    draw_mod,
    sha256_bytes,
    u8,
)
from tools.bata.duca_evidence_io import (
    canonical_sha256,
    verify_content_sha256,
    with_content_sha256,
)


ROLE_MANIFEST_SCHEMA = "duca_admission_v2_1_role_manifest_v1"
SEMANTIC_INVENTORY_SCHEMA = "duca_v2_1_source_inventory_semantic_v1"
ROLE_ORDER = ("scale_fit", "calibration", "admission_holdout")
ROLE_PERMUTATIONS = (
    ("scale_fit", "calibration", "admission_holdout"),
    ("scale_fit", "admission_holdout", "calibration"),
    ("calibration", "scale_fit", "admission_holdout"),
    ("calibration", "admission_holdout", "scale_fit"),
    ("admission_holdout", "scale_fit", "calibration"),
    ("admission_holdout", "calibration", "scale_fit"),
)
LONG_STRATUM_BOUNDS = (0, 17, 35, 52, 70)


def _immutable_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    video_id = raw.get("video_id")
    if not isinstance(video_id, str):
        raise TypeError("video_id must be a Unicode string")
    canonical_text(video_id, field_name="video_id")
    source_subset = raw.get("source_subset")
    if not isinstance(source_subset, str):
        raise TypeError("source_subset must be a Unicode string")
    canonical_text(source_subset, field_name="source_subset")
    if source_subset != "training":
        raise ValueError("source_subset must be the frozen training subset")
    frame_count = raw.get("frame_count")
    snippet_count = raw.get("snippet_count")
    if type(frame_count) is not int or type(snippet_count) is not int:
        raise TypeError("frame_count and snippet_count must be integers")
    lengths_raw = raw.get("natural_window_valid_lengths")
    if not isinstance(lengths_raw, Sequence) or isinstance(lengths_raw, (str, bytes)):
        raise ValueError("natural_window_valid_lengths must be a non-empty sequence")
    if any(type(value) is not int for value in lengths_raw):
        raise TypeError("natural valid lengths must be integers")
    lengths = list(lengths_raw)
    if not lengths or frame_count <= 0 or snippet_count <= 0:
        raise ValueError("inventory counts and natural lengths must be positive")
    if all(value == 768 for value in lengths):
        length_stratum = "natural_full"
    elif len(lengths) == 1 and 1 <= lengths[0] <= 767:
        length_stratum = "natural_short"
    else:
        raise ValueError("video is neither canonical natural_full nor natural_short")
    supplied_stratum = raw.get("length_stratum")
    if supplied_stratum is not None:
        if not isinstance(supplied_stratum, str):
            raise TypeError("length_stratum must be a string when supplied")
        if supplied_stratum != length_stratum:
            raise ValueError("length_stratum disagrees with natural window lengths")
    return {
        "video_id": video_id,
        "source_subset": source_subset,
        "frame_count": frame_count,
        "snippet_count": snippet_count,
        "natural_window_valid_lengths": lengths,
        "length_stratum": length_stratum,
    }


def canonicalize_inventory(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    canonical_records = [_immutable_record(record) for record in records]
    canonical_records.sort(
        key=lambda row: canonical_text(row["video_id"], field_name="video_id")
    )
    video_ids = [row["video_id"] for row in canonical_records]
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("inventory contains duplicate video_id values")
    payload = {
        "schema": SEMANTIC_INVENTORY_SCHEMA,
        "source_role": "detector_selector_train",
        "records": canonical_records,
    }
    return canonical_records, payload, canonical_sha256(payload)


def _long_order_key(row: Mapping[str, Any], semantic_sha: str) -> tuple[Any, ...]:
    video = canonical_text(str(row["video_id"]), field_name="video_id")
    return (
        int(row["snippet_count"]),
        domain_hash("long-order-tie", sha256_bytes(semantic_sha), video),
        video,
    )


def _short_order_key(row: Mapping[str, Any], semantic_sha: str) -> tuple[Any, ...]:
    video = canonical_text(str(row["video_id"]), field_name="video_id")
    return (
        int(row["snippet_count"]),
        domain_hash("short-order-tie", sha256_bytes(semantic_sha), video),
        video,
    )


def _chunks(
    values: Sequence[Mapping[str, Any]], size: int
) -> list[list[Mapping[str, Any]]]:
    if len(values) % size:
        raise ValueError("triplet source length is not divisible by three")
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def _canonical_rank(
    *, role_id: str, rows: Sequence[Mapping[str, Any]], semantic_sha: str
) -> list[dict[str, Any]]:
    semantic_bytes = sha256_bytes(semantic_sha, field_name="semantic split sha256")
    keyed = []
    for row in rows:
        video = canonical_text(str(row["video_id"]), field_name="video_id")
        key = domain_hash(
            "canonical-video-rank",
            semantic_bytes,
            canonical_text(role_id, field_name="role_id"),
            video,
        )
        keyed.append((key, video, dict(row)))
    keyed.sort(key=lambda item: (item[0], item[1]))
    return [
        {**row, "canonical_video_rank": rank, "canonical_rank_hash": key.hex()}
        for rank, (key, _video, row) in enumerate(keyed)
    ]


def build_role_manifest(
    *,
    inventory_records: Sequence[Mapping[str, Any]],
    source_split_artifact_sha256: str,
) -> dict[str, Any]:
    sha256_bytes(
        source_split_artifact_sha256, field_name="source split artifact sha256"
    )
    rows, semantic_inventory, semantic_sha = canonicalize_inventory(inventory_records)
    longs = sorted(
        (row for row in rows if row["length_stratum"] == "natural_full"),
        key=lambda row: _long_order_key(row, semantic_sha),
    )
    shorts = sorted(
        (row for row in rows if row["length_stratum"] == "natural_short"),
        key=lambda row: _short_order_key(row, semantic_sha),
    )
    if len(rows) != 100 or len(longs) != 70 or len(shorts) != 30:
        raise ValueError(
            "SOURCE_INVENTORY_MISMATCH: expected exactly 70 full and 30 short videos"
        )

    semantic_bytes = sha256_bytes(semantic_sha, field_name="semantic split sha256")
    reserves: list[dict[str, Any]] = []
    for stratum_index in range(4):
        lower = LONG_STRATUM_BOUNDS[stratum_index]
        upper = LONG_STRATUM_BOUNDS[stratum_index + 1]
        candidates = longs[lower:upper]
        reserve = min(
            candidates,
            key=lambda row: (
                domain_hash(
                    "long-reserve",
                    semantic_bytes,
                    u8(stratum_index),
                    canonical_text(str(row["video_id"]), field_name="video_id"),
                ),
                canonical_text(str(row["video_id"]), field_name="video_id"),
            ),
        )
        reserves.append(
            {
                "reserve_id": f"R{stratum_index:02d}",
                "video_id": reserve["video_id"],
                "reserve_stratum_index": stratum_index,
                "selection_hash": domain_hash_hex(
                    "long-reserve",
                    semantic_bytes,
                    u8(stratum_index),
                    canonical_text(str(reserve["video_id"]), field_name="video_id"),
                ),
            }
        )
    reserve_ids = {row["video_id"] for row in reserves}
    long_used = sorted(
        (row for row in longs if row["video_id"] not in reserve_ids),
        key=lambda row: _long_order_key(row, semantic_sha),
    )
    if len(long_used) != 66:
        raise AssertionError("reserve selection did not leave 66 long videos")

    roles: dict[str, list[dict[str, Any]]] = {role_id: [] for role_id in ROLE_ORDER}
    triplets: list[dict[str, Any]] = []
    for prefix, source, length_stratum in (
        ("L", long_used, "natural_full"),
        ("S", shorts, "natural_short"),
    ):
        for index, members in enumerate(_chunks(source, 3)):
            triplet_id = f"{prefix}{index:02d}"
            member_ids = [str(row["video_id"]) for row in members]
            assignment_digit = draw_mod(
                6,
                "triplet-role-permutation",
                semantic_bytes,
                triplet_id.encode("ascii"),
                *(
                    canonical_text(video_id, field_name="video_id")
                    for video_id in member_ids
                ),
            )
            role_permutation = ROLE_PERMUTATIONS[assignment_digit]
            assignment = {
                role_permutation[position]: member_ids[position]
                for position in range(3)
            }
            assignment_hash = domain_hash_hex(
                "triplet-role-assignment",
                semantic_bytes,
                triplet_id.encode("ascii"),
                u8(assignment_digit),
                *(
                    canonical_text(assignment[role_id], field_name="video_id")
                    for role_id in ROLE_ORDER
                ),
            )
            triplets.append(
                {
                    "triplet_id": triplet_id,
                    "length_stratum": length_stratum,
                    "ordered_member_video_ids": member_ids,
                    **{
                        f"{role_id}_video_id": assignment[role_id]
                        for role_id in ROLE_ORDER
                    },
                    "assignment_digit": assignment_digit,
                    "assignment_hash": assignment_hash,
                }
            )
            row_by_id = {str(row["video_id"]): row for row in members}
            for role_id in ROLE_ORDER:
                video_id = assignment[role_id]
                roles[role_id].append(
                    {
                        "video_id": video_id,
                        "triplet_id": triplet_id,
                        "length_stratum": length_stratum,
                        "frame_count": int(row_by_id[video_id]["frame_count"]),
                        "snippet_count": int(row_by_id[video_id]["snippet_count"]),
                    }
                )

    all_assigned = [
        row["video_id"] for role_rows in roles.values() for row in role_rows
    ]
    if len(all_assigned) != 96 or len(set(all_assigned)) != 96:
        raise AssertionError("roles are not a disjoint 96-video allocation")
    if set(all_assigned) & reserve_ids:
        raise AssertionError("reserve videos leaked into a role")
    for role_id in ROLE_ORDER:
        if sum(row["length_stratum"] == "natural_full" for row in roles[role_id]) != 22:
            raise AssertionError(f"{role_id} does not contain 22 natural-full videos")
        if (
            sum(row["length_stratum"] == "natural_short" for row in roles[role_id])
            != 10
        ):
            raise AssertionError(f"{role_id} does not contain 10 natural-short videos")
        roles[role_id] = _canonical_rank(
            role_id=role_id, rows=roles[role_id], semantic_sha=semantic_sha
        )

    payload = {
        "schema": ROLE_MANIFEST_SCHEMA,
        "status": "PASSED",
        "protocol_id": PROTOCOL_ID,
        "source_split_artifact_sha256": source_split_artifact_sha256,
        "source_split_semantic_sha256": semantic_sha,
        "semantic_inventory": semantic_inventory,
        "reserves": reserves,
        "triplets": triplets,
        "roles": roles,
        "authorization_scope": "NONE",
        "phase1_v2_authorized": False,
        "holdout_open_authorized": False,
        "paper_claim_allowed": False,
        "official_final_sealed": True,
    }
    return with_content_sha256(payload)


def validate_role_manifest(payload: Mapping[str, Any]) -> None:
    verify_content_sha256(payload)
    if (
        payload.get("schema") != ROLE_MANIFEST_SCHEMA
        or payload.get("status") != "PASSED"
    ):
        raise ValueError("invalid role manifest identity")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("role manifest protocol drift")
    if payload.get("authorization_scope") != "NONE":
        raise ValueError("protocol-only role manifest cannot authorize execution")
    if any(
        payload.get(field) is not False
        for field in (
            "phase1_v2_authorized",
            "holdout_open_authorized",
            "paper_claim_allowed",
        )
    ):
        raise ValueError("protocol-only role manifest contains forbidden authorization")
    if payload.get("official_final_sealed") is not True:
        raise ValueError("official-final must remain sealed")
    sha256_bytes(
        payload.get("source_split_artifact_sha256"),
        field_name="source split artifact sha256",
    )
    sha256_bytes(
        payload.get("source_split_semantic_sha256"),
        field_name="source split semantic sha256",
    )
    semantic_inventory = payload.get("semantic_inventory")
    if not isinstance(semantic_inventory, Mapping):
        raise ValueError("semantic inventory is missing")
    if semantic_inventory.get("schema") != SEMANTIC_INVENTORY_SCHEMA:
        raise ValueError("semantic inventory schema drift")
    inventory_records = semantic_inventory.get("records")
    if not isinstance(inventory_records, list):
        raise ValueError("semantic inventory records are missing")
    canonical_rows, canonical_payload, semantic_sha = canonicalize_inventory(
        inventory_records
    )
    if canonical_payload != semantic_inventory:
        raise ValueError("semantic inventory is not canonical")
    if semantic_sha != payload.get("source_split_semantic_sha256"):
        raise ValueError("semantic inventory SHA-256 drift")
    roles = payload.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != set(ROLE_ORDER):
        raise ValueError("role manifest must use the frozen role order")
    seen: set[str] = set()
    for role_id in ROLE_ORDER:
        rows = roles[role_id]
        if not isinstance(rows, list) or len(rows) != 32:
            raise ValueError("each role must contain 32 videos")
        if any(not isinstance(row, Mapping) for row in rows):
            raise ValueError("role entries must be objects")
        ranks = [row.get("canonical_video_rank") for row in rows]
        if any(type(rank) is not int for rank in ranks):
            raise ValueError("canonical role ranks must be integers")
        if ranks != list(range(32)):
            raise ValueError("canonical role ranks are not exact")
        for row in rows:
            video_id = row.get("video_id")
            if not isinstance(video_id, str):
                raise ValueError("role video IDs must be strings")
            canonical_text(video_id, field_name="video_id")
            if video_id in seen:
                raise ValueError("video appears in more than one role")
            seen.add(video_id)
    if len(payload.get("reserves", [])) != 4:
        raise ValueError("role manifest must contain four reserves")
    expected = build_role_manifest(
        inventory_records=canonical_rows,
        source_split_artifact_sha256=str(payload["source_split_artifact_sha256"]),
    )
    for field in (
        "source_split_semantic_sha256",
        "semantic_inventory",
        "reserves",
        "triplets",
        "roles",
    ):
        if payload.get(field) != expected[field]:
            raise ValueError(
                f"role manifest {field} does not match deterministic reconstruction"
            )
    if set(payload) != set(expected):
        raise ValueError("role manifest is not a closed-world object")
