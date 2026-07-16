import hashlib
import json
import unicodedata

import pytest

from opentad.models.chronotransport.protocol import (
    R2_PROTOCOL_ID,
    build_window_payload,
    canonical_json_bytes,
    split_video_ids,
    stage_b_exposure_matrix,
    stage_c_exposure_matrix,
    validate_stage_b_exposures,
    window_digest,
)


def test_canonical_json_is_utf8_nfc_sorted_and_compact():
    value = {"z": "e\u0301", "a": [3, True, None]}
    encoded = canonical_json_bytes(value)
    assert encoded == b'{"a":[3,true,null],"z":"\\u00e9"}'
    assert not encoded.startswith(b"\xef\xbb\xbf")
    assert json.loads(encoded) == {"a": [3, True, None], "z": "é"}


def test_split_contract_is_seed_independent_and_exact_140_30_30():
    videos = [f"video_{index:03d}" for index in range(200)]
    manifest = split_video_ids(videos)
    assert R2_PROTOCOL_ID == "CT-P3R-3S-r2"
    assert tuple(manifest) == ("fit", "calibration", "evaluation")
    assert [len(manifest[name]) for name in manifest] == [140, 30, 30]
    assert len(set(sum((list(values) for values in manifest.values()), []))) == 200

    expected_first = min(
        videos,
        key=lambda video: (
            hashlib.sha256(
                b"CT-P3R-3S-r2-split-v1\0" + b"3407\0" + video.encode("utf-8")
            ).digest(),
            video.encode("utf-8"),
        ),
    )
    assert manifest["fit"][0] == expected_first


def test_split_rejects_nonunique_or_wrong_population():
    with pytest.raises(ValueError, match="exactly 200"):
        split_video_ids(["v"] * 200)
    with pytest.raises(ValueError, match="exactly 200"):
        split_video_ids([f"v{i}" for i in range(199)])


def test_window_digest_and_label_free_edge_padding_contract():
    video_id = unicodedata.normalize("NFC", "vide\u0301o")
    media_sha = "ab" * 32
    expected = hashlib.sha256(
        b"CT-P3R-3S-r2-window-v1\0"
        + video_id.encode("utf-8")
        + b"\0"
        + media_sha.encode("ascii")
        + b"\0"
        + b"3"
    ).digest()
    assert window_digest(video_id, media_sha, 3) == expected

    payload = build_window_payload(video_id, media_sha, [11, 15, 19], width=8)
    assert payload["window_start"] == 0
    assert payload["sampled_frame_indices"] == [11, 15, 19, 19, 19, 19, 19, 19]
    assert payload["valid_mask"] == [True, True, True, False, False, False, False, False]
    assert payload["padding_positions"] == [3, 4, 5, 6, 7]
    assert len(payload["payload_sha256"]) == 64
    assert payload["payload_sha256"] == hashlib.sha256(
        canonical_json_bytes({key: value for key, value in payload.items() if key != "payload_sha256"})
    ).hexdigest()


def test_window_contract_rejects_empty_indices_and_invalid_media_hash():
    with pytest.raises(ValueError, match="non-empty"):
        build_window_payload("v", "aa" * 32, [], width=8)
    with pytest.raises(ValueError, match="lowercase"):
        window_digest("v", "AA" * 32, 8)


def test_stage_b_exact_exposure_contract():
    matrix = stage_b_exposure_matrix()
    validate_stage_b_exposures(matrix)
    assert tuple(matrix) == (3407, 3408, 3409)
    assert all(len(rows) == 140 for rows in matrix.values())
    assert [row["candidate"] for row in matrix[3407][-12:]] == [8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3]
    assert [row["candidate"] for row in matrix[3408][-12:]] == [12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7]
    assert [row["candidate"] for row in matrix[3409][-12:]] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    for index in range(140):
        assert len({matrix[seed][index]["candidate"] for seed in matrix}) == 3


def test_stage_c_has_8400_exposures_and_525_per_candidate_per_seed():
    matrix = stage_c_exposure_matrix()
    for seed, rows in matrix.items():
        assert len(rows) == 8400
        assert {candidate: sum(row["candidate"] == candidate for row in rows) for candidate in range(16)} == {
            candidate: 525 for candidate in range(16)
        }
        assert rows[0]["successful_update"] == 0
        assert rows[0]["batch_position"] == 0
        assert rows[-1]["successful_update"] == 4199
        assert rows[-1]["batch_position"] == 1
