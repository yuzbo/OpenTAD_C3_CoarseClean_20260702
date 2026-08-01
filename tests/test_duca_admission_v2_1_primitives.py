from __future__ import annotations

import copy
import unicodedata
from collections import Counter

import pytest

import tools.bata.duca_admission_v2_1_roles as roles_module
from tools.bata.duca_admission_v2_1_hashing import (
    canonical_text,
    domain_hash,
    draw_mod,
    u32be,
)
from tools.bata.duca_admission_v2_1_incidence import build_incidence, validate_incidence
from tools.bata.duca_admission_v2_1_roles import (
    ROLE_ORDER,
    _canonical_rank,
    build_role_manifest,
    validate_role_manifest,
)
from tools.bata.duca_evidence_io import verify_content_sha256, with_content_sha256


def make_inventory():
    rows = []
    for index in range(70):
        rows.append(
            {
                "video_id": f"long_{index:03d}",
                "source_subset": "training",
                "frame_count": (900 + index) * 4,
                "snippet_count": 900 + index,
                "natural_window_valid_lengths": [768],
            }
        )
    for index in range(30):
        length = 100 + 20 * index
        rows.append(
            {
                "video_id": f"short_{index:03d}",
                "source_subset": "training",
                "frame_count": length * 4,
                "snippet_count": length,
                "natural_window_valid_lengths": [length],
            }
        )
    return rows


def test_domain_hash_is_length_prefixed_and_draw_mod_is_deterministic():
    assert domain_hash("test", b"ab", b"c") != domain_hash("test", b"a", b"bc")
    assert draw_mod(6, "test-draw", b"fixed") == draw_mod(6, "test-draw", b"fixed")


@pytest.mark.parametrize("value", ["bad\x00id", unicodedata.normalize("NFD", "vidéo")])
def test_canonical_text_rejects_nul_and_non_nfc(value):
    with pytest.raises(ValueError):
        canonical_text(value, field_name="video_id")


def test_role_manifest_and_incidence_are_input_permutation_invariant():
    inventory = make_inventory()
    forward = build_role_manifest(
        inventory_records=inventory, source_split_artifact_sha256="a" * 64
    )
    reverse = build_role_manifest(
        inventory_records=list(reversed(inventory)),
        source_split_artifact_sha256="a" * 64,
    )
    assert forward == reverse
    verify_content_sha256(forward)
    validate_role_manifest(forward)
    assert len(forward["reserves"]) == 4
    assert len(forward["triplets"]) == 32
    assigned = []
    for role_id in ROLE_ORDER:
        rows = forward["roles"][role_id]
        assert len(rows) == 32
        assert Counter(row["length_stratum"] for row in rows) == {
            "natural_full": 22,
            "natural_short": 10,
        }
        assert [row["canonical_video_rank"] for row in rows] == list(range(32))
        assigned.extend(row["video_id"] for row in rows)
    assert len(set(assigned)) == 96
    assert not set(assigned) & {row["video_id"] for row in forward["reserves"]}
    assert all(
        row["triplet_id"] not in row["ordered_member_video_ids"]
        for row in forward["triplets"]
    )

    incidence_forward = build_incidence(forward)
    incidence_reverse = build_incidence(reverse)
    assert incidence_forward == incidence_reverse
    verify_content_sha256(incidence_forward)
    validate_incidence(incidence_forward)
    assert len(incidence_forward["cells"]) == 192
    for role_id in ROLE_ORDER:
        cells = [row for row in incidence_forward["cells"] if row["role_id"] == role_id]
        assert Counter(row["video_id"] for row in cells) == Counter(
            {row["video_id"]: 2 for row in forward["roles"][role_id]}
        )
        assert Counter(row["logical_process_index"] for row in cells) == Counter(
            {index: 8 for index in range(8)}
        )


def test_role_manifest_rejects_inventory_other_than_exact_70_30():
    with pytest.raises(ValueError, match="SOURCE_INVENTORY_MISMATCH"):
        build_role_manifest(
            inventory_records=make_inventory()[:-1],
            source_split_artifact_sha256="a" * 64,
        )


def test_injected_rank_hash_collision_uses_video_id_bytes(monkeypatch):
    monkeypatch.setattr(
        roles_module, "domain_hash", lambda *_args, **_kwargs: b"x" * 32
    )
    rows = [
        {"video_id": "z", "triplet_id": "T", "length_stratum": "natural_full"},
        {"video_id": "a", "triplet_id": "T", "length_stratum": "natural_full"},
    ]
    ranked = _canonical_rank(role_id="scale_fit", rows=rows, semantic_sha="b" * 64)
    assert [row["video_id"] for row in ranked] == ["a", "z"]


def test_role_manifest_authorization_fields_fail_closed():
    manifest = build_role_manifest(
        inventory_records=make_inventory(), source_split_artifact_sha256="a" * 64
    )
    tampered = copy.deepcopy(manifest)
    tampered["phase1_v2_authorized"] = True
    tampered = with_content_sha256(tampered)
    with pytest.raises(ValueError, match="forbidden authorization"):
        validate_role_manifest(tampered)


def test_inventory_rejects_nontraining_and_nonstring_identity_without_coercion():
    nontraining = make_inventory()
    nontraining[0]["source_subset"] = "validation"
    with pytest.raises(ValueError, match="frozen training"):
        build_role_manifest(
            inventory_records=nontraining, source_split_artifact_sha256="a" * 64
        )
    nonstring = make_inventory()
    nonstring[0]["video_id"] = 123
    with pytest.raises(TypeError, match="Unicode string"):
        build_role_manifest(
            inventory_records=nonstring, source_split_artifact_sha256="a" * 64
        )
    with pytest.raises(TypeError, match="integer"):
        u32be(1.0)


def test_rehashed_role_and_incidence_tampering_is_rejected():
    manifest = build_role_manifest(
        inventory_records=make_inventory(), source_split_artifact_sha256="a" * 64
    )
    role_tamper = copy.deepcopy(manifest)
    role_tamper["reserves"][0]["video_id"] = role_tamper["reserves"][1]["video_id"]
    role_tamper = with_content_sha256(role_tamper)
    with pytest.raises(ValueError, match="deterministic reconstruction"):
        validate_role_manifest(role_tamper)

    incidence = build_incidence(manifest)
    process_tamper = copy.deepcopy(incidence)
    process_tamper["cells"][0]["logical_process_index"] ^= 1
    process_tamper = with_content_sha256(process_tamper)
    with pytest.raises(ValueError, match="process degree|logical process drift"):
        validate_incidence(process_tamper)

    permutation_tamper = copy.deepcopy(incidence)
    permutation_tamper["process_permutations"][ROLE_ORDER[0]][:2] = reversed(
        permutation_tamper["process_permutations"][ROLE_ORDER[0]][:2]
    )
    permutation_tamper = with_content_sha256(permutation_tamper)
    with pytest.raises(ValueError, match="permutations drifted"):
        validate_incidence(permutation_tamper)
