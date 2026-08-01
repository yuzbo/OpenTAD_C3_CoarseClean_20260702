from __future__ import annotations

from collections import Counter, deque
from collections.abc import Mapping
from typing import Any

from tools.bata.duca_admission_v2_1_hashing import (
    PROTOCOL_ID,
    canonical_text,
    domain_hash,
    domain_hash_hex,
    sha256_bytes,
    u8,
)
from tools.bata.duca_admission_v2_1_roles import ROLE_ORDER, validate_role_manifest
from tools.bata.duca_evidence_io import verify_content_sha256, with_content_sha256


INCIDENCE_SCHEMA = "duca_admission_v2_1_incidence_v1"


def process_permutation(role_id: str, semantic_sha: str) -> dict[int, int]:
    semantic_bytes = sha256_bytes(semantic_sha, field_name="semantic split sha256")
    role_bytes = canonical_text(role_id, field_name="role_id")
    ordered = sorted(
        range(8),
        key=lambda process: (
            domain_hash(
                "process-label-permutation",
                semantic_bytes,
                role_bytes,
                u8(process),
            ),
            process,
        ),
    )
    return {base: ordered[base] for base in range(8)}


def _validate_role_graph(cells: list[dict[str, Any]], role_id: str) -> None:
    if len(cells) != 64:
        raise ValueError(f"{role_id} incidence must contain 64 cells")
    video_degree = Counter(cell["video_id"] for cell in cells)
    process_degree = Counter(cell["logical_process_index"] for cell in cells)
    if set(video_degree.values()) != {2} or len(video_degree) != 32:
        raise ValueError(f"{role_id} video degree is not exactly two")
    if process_degree != Counter({index: 8 for index in range(8)}):
        raise ValueError(f"{role_id} process degree is not exactly eight")
    rank_slot = Counter((cell["canonical_video_rank"], cell["slot"]) for cell in cells)
    if rank_slot != Counter((rank, slot) for rank in range(32) for slot in (0, 1)):
        raise ValueError(f"{role_id} rank/slot grid is not exactly 32x2")
    for rank in range(32):
        rank_cells = [cell for cell in cells if cell["canonical_video_rank"] == rank]
        if len({cell["video_id"] for cell in rank_cells}) != 1:
            raise ValueError(f"{role_id} canonical rank maps to multiple videos")
        if len({cell["length_stratum"] for cell in rank_cells}) != 1:
            raise ValueError(f"{role_id} canonical rank maps to multiple strata")
    unique_video_strata = {cell["video_id"]: cell["length_stratum"] for cell in cells}
    if Counter(unique_video_strata.values()) != Counter(
        {"natural_full": 22, "natural_short": 10}
    ):
        raise ValueError(f"{role_id} length-stratum inventory is not 22/10")
    edges = {(cell["video_id"], cell["logical_process_index"]) for cell in cells}
    if len(edges) != 64:
        raise ValueError(f"{role_id} has a duplicate video-process cell")

    adjacency: dict[str, set[str]] = {}
    for video_id, process in edges:
        video_node = f"v:{video_id}"
        process_node = f"p:{process}"
        adjacency.setdefault(video_node, set()).add(process_node)
        adjacency.setdefault(process_node, set()).add(video_node)
    visited: set[str] = set()
    queue = deque([next(iter(adjacency))])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        queue.extend(adjacency[node] - visited)
    if visited != set(adjacency):
        raise ValueError(f"{role_id} incidence graph is disconnected")


def build_incidence(role_manifest: Mapping[str, Any]) -> dict[str, Any]:
    validate_role_manifest(role_manifest)
    semantic_sha = str(role_manifest["source_split_semantic_sha256"])
    role_manifest_sha = str(role_manifest["content_sha256"])
    semantic_bytes = sha256_bytes(semantic_sha, field_name="semantic split sha256")
    role_manifest_bytes = sha256_bytes(
        role_manifest_sha, field_name="role manifest sha256"
    )
    process_permutations: dict[str, list[int]] = {}
    cells: list[dict[str, Any]] = []
    for role_id in ROLE_ORDER:
        permutation = process_permutation(role_id, semantic_sha)
        process_permutations[role_id] = [permutation[index] for index in range(8)]
        role_cells: list[dict[str, Any]] = []
        for video in role_manifest["roles"][role_id]:
            rank = int(video["canonical_video_rank"])
            bases = (rank % 8, (rank + rank // 8 + 1) % 8)
            if bases[0] == bases[1]:
                raise AssertionError("base incidence assigned one process twice")
            for slot, base_process in enumerate(bases):
                logical = permutation[base_process]
                video_id = str(video["video_id"])
                cell = {
                    "role_id": role_id,
                    "canonical_video_rank": rank,
                    "video_id": video_id,
                    "length_stratum": str(video["length_stratum"]),
                    "slot": slot,
                    "base_process_index": base_process,
                    "logical_process_index": logical,
                    "process_id": f"{role_id}:p{logical:02d}",
                }
                cell["cell_id"] = domain_hash_hex(
                    "planned-cell-id",
                    role_manifest_bytes,
                    semantic_bytes,
                    canonical_text(role_id, field_name="role_id"),
                    canonical_text(video_id, field_name="video_id"),
                    u8(slot),
                    u8(logical),
                )
                role_cells.append(cell)
        role_cells.sort(key=lambda row: (row["canonical_video_rank"], row["slot"]))
        _validate_role_graph(role_cells, role_id)
        cells.extend(role_cells)
    payload = {
        "schema": INCIDENCE_SCHEMA,
        "status": "PASSED",
        "protocol_id": PROTOCOL_ID,
        "role_manifest_sha256": role_manifest_sha,
        "source_split_semantic_sha256": semantic_sha,
        "process_permutations": process_permutations,
        "cells": cells,
        "authorization_scope": "NONE",
        "phase1_v2_authorized": False,
        "holdout_open_authorized": False,
        "paper_claim_allowed": False,
        "official_final_sealed": True,
    }
    return with_content_sha256(payload)


def validate_incidence(
    payload: Mapping[str, Any], *, role_manifest: Mapping[str, Any] | None = None
) -> None:
    verify_content_sha256(payload)
    if set(payload) != {
        "schema",
        "status",
        "protocol_id",
        "role_manifest_sha256",
        "source_split_semantic_sha256",
        "process_permutations",
        "cells",
        "authorization_scope",
        "phase1_v2_authorized",
        "holdout_open_authorized",
        "paper_claim_allowed",
        "official_final_sealed",
        "content_sha256",
    }:
        raise ValueError("incidence is not a closed-world object")
    if payload.get("schema") != INCIDENCE_SCHEMA or payload.get("status") != "PASSED":
        raise ValueError("invalid incidence identity")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("incidence protocol drift")
    if payload.get("authorization_scope") != "NONE":
        raise ValueError("protocol incidence cannot authorize execution")
    if any(
        payload.get(field) is not False
        for field in (
            "phase1_v2_authorized",
            "holdout_open_authorized",
            "paper_claim_allowed",
        )
    ):
        raise ValueError("protocol incidence contains forbidden authorization")
    if payload.get("official_final_sealed") is not True:
        raise ValueError("official-final must remain sealed")
    role_manifest_sha = payload.get("role_manifest_sha256")
    semantic_sha = payload.get("source_split_semantic_sha256")
    role_manifest_bytes = sha256_bytes(
        role_manifest_sha, field_name="role manifest sha256"
    )
    semantic_bytes = sha256_bytes(semantic_sha, field_name="semantic split sha256")
    permutations = payload.get("process_permutations")
    expected_permutations = {
        role_id: [
            process_permutation(role_id, semantic_sha)[index] for index in range(8)
        ]
        for role_id in ROLE_ORDER
    }
    if permutations != expected_permutations:
        raise ValueError("incidence process permutations drifted")
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) != 192:
        raise ValueError("incidence must contain exactly 192 cells")
    expected_cell_keys = {
        "role_id",
        "canonical_video_rank",
        "video_id",
        "length_stratum",
        "slot",
        "base_process_index",
        "logical_process_index",
        "process_id",
        "cell_id",
    }
    for cell in cells:
        if not isinstance(cell, Mapping) or set(cell) != expected_cell_keys:
            raise ValueError("incidence cell is not closed-world")
        if cell.get("role_id") not in ROLE_ORDER:
            raise ValueError("incidence role_id drift")
        if not isinstance(cell.get("video_id"), str):
            raise ValueError("incidence video_id must be a string")
        canonical_text(cell["video_id"], field_name="video_id")
        if not isinstance(cell.get("process_id"), str):
            raise ValueError("incidence process_id must be a string")
        if not isinstance(cell.get("cell_id"), str):
            raise ValueError("incidence cell_id must be a string")
        sha256_bytes(cell["cell_id"], field_name="incidence cell_id")
        for field in (
            "canonical_video_rank",
            "slot",
            "base_process_index",
            "logical_process_index",
        ):
            if type(cell.get(field)) is not int:
                raise ValueError(f"incidence {field} must be an integer")
    if len({cell["cell_id"] for cell in cells}) != 192:
        raise ValueError("incidence cell IDs are not unique")
    if any("block_id" in cell for cell in cells):
        raise ValueError("statistical cells must not contain block_id")
    expected_order = []
    for role_id in ROLE_ORDER:
        role_cells = [dict(cell) for cell in cells if cell.get("role_id") == role_id]
        _validate_role_graph(role_cells, role_id)
        permutation = expected_permutations[role_id]
        for cell in role_cells:
            rank = cell["canonical_video_rank"]
            slot = cell["slot"]
            if not 0 <= rank < 32 or slot not in (0, 1):
                raise ValueError("incidence rank/slot is invalid")
            bases = (rank % 8, (rank + rank // 8 + 1) % 8)
            base = bases[slot]
            logical = permutation[base]
            video_id = cell["video_id"]
            if cell.get("base_process_index") != base:
                raise ValueError("incidence base process drift")
            if cell.get("logical_process_index") != logical:
                raise ValueError("incidence logical process drift")
            if cell.get("process_id") != f"{role_id}:p{logical:02d}":
                raise ValueError("incidence process_id drift")
            if cell.get("length_stratum") not in {"natural_full", "natural_short"}:
                raise ValueError("incidence length stratum is invalid")
            expected_cell_id = domain_hash_hex(
                "planned-cell-id",
                role_manifest_bytes,
                semantic_bytes,
                canonical_text(role_id, field_name="role_id"),
                canonical_text(video_id, field_name="video_id"),
                u8(slot),
                u8(logical),
            )
            if cell.get("cell_id") != expected_cell_id:
                raise ValueError("incidence cell_id drift")
        expected_order.extend(
            sorted(
                role_cells, key=lambda row: (row["canonical_video_rank"], row["slot"])
            )
        )
    if list(cells) != expected_order:
        raise ValueError("incidence cells are not in canonical role/rank/slot order")
    if role_manifest is not None:
        validate_role_manifest(role_manifest)
        if role_manifest.get("content_sha256") != role_manifest_sha:
            raise ValueError("incidence role manifest binding drift")
        if role_manifest.get("source_split_semantic_sha256") != semantic_sha:
            raise ValueError("incidence semantic split binding drift")
        expected = build_incidence(role_manifest)
        if payload != expected:
            raise ValueError(
                "incidence does not match deterministic role-manifest reconstruction"
            )
