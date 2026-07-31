import copy
import json
import os

import pytest

from tools.aggregate_odfcr_k384_replays import (
    _canonical_sha256,
    _read_canonical_json,
    _validate_allocation_ledger,
)


def _selector_receipt():
    records = []
    for video_index in range(40):
        unsigned = {
            "video_id": "video_{:03d}".format(video_index),
            "valid_counts_per_level": [300, 200],
            "selected_indices_per_level": [
                list(range(230)),
                list(range(154)),
            ],
            "selected_count": 384,
        }
        record = dict(unsigned)
        record["allocation_sha256"] = _canonical_sha256(unsigned)
        records.append(record)
    return {
        "policy": "stratified_uniform",
        "budget": 384,
        "hash_seed": 2026073100,
        "allocation_video_count": 40,
        "allocation_records": records,
        "allocation_ledger_sha256": _canonical_sha256(records),
    }


def test_odfcr_k384_allocation_ledger_is_exact_and_hash_bound():
    selector = _selector_receipt()
    holdout_ids = frozenset(
        "video_{:03d}".format(index) for index in range(40)
    )
    _validate_allocation_ledger(selector, holdout_ids)


@pytest.mark.parametrize(
    "mutator,match",
    [
        (
            lambda selector: selector["allocation_records"][0].update(
                selected_count=383
            ),
            "budget",
        ),
        (
            lambda selector: selector["allocation_records"][0][
                "selected_indices_per_level"
            ][0].append(999),
            "indices",
        ),
        (
            lambda selector: selector.update(
                allocation_ledger_sha256="0" * 64
            ),
            "ledger hash",
        ),
    ],
)
def test_odfcr_k384_allocation_ledger_rejects_contract_drift(
    mutator, match
):
    selector = copy.deepcopy(_selector_receipt())
    mutator(selector)
    holdout_ids = frozenset(
        "video_{:03d}".format(index) for index in range(40)
    )
    with pytest.raises(ValueError, match=match):
        _validate_allocation_ledger(selector, holdout_ids)


def test_odfcr_k384_json_input_requires_canonical_absolute_receipt(
    tmp_path, monkeypatch
):
    path = tmp_path / "replay.json"
    path.write_text(json.dumps({"validation_pass": True}), encoding="utf-8")
    payload, receipt = _read_canonical_json(
        os.path.realpath(path), "ODF-CR K384 replay"
    )
    assert payload["validation_pass"] is True
    assert receipt["path"] == os.path.realpath(path)
    assert receipt["size_bytes"] == path.stat().st_size

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="canonical absolute"):
        _read_canonical_json("replay.json", "ODF-CR K384 replay")
