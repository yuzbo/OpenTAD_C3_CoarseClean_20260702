from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tools.bata.validate_duca_paper_code_gate import validate_code_gate_artifact


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "duca_paper_real_short_window_heavy_backbone_gate_v1"
CONFIG_DEFAULT = (
    "configs/adatad/thumos/"
    "duca_paper_uniform_mixed_train_k384_eval_full200.py"
)
REQUESTED_BUDGETS = (192, 256, 384, 512)
AUDITED_PATHS = (
    "configs/adatad/thumos/duca_paper_full200_base.py",
    "configs/adatad/thumos/duca_paper_rime_selected_axis_base.py",
    "configs/adatad/thumos/duca_paper_uniform_mixed_train_k384_eval_full200.py",
    "opentad/datasets/builder.py",
    "opentad/datasets/duca_stateless.py",
    "opentad/datasets/thumos.py",
    "opentad/datasets/transforms/end_to_end.py",
    "opentad/models/backbones/backbone_wrapper.py",
    "opentad/models/detectors/actionformer.py",
    "opentad/models/detectors/single_stage.py",
    "opentad/models/duca/rime.py",
    "opentad/models/duca/structured_selection.py",
    "opentad/models/selectors/duca_protected_e2e_frame_selector.py",
    "opentad/models/selectors/duca_rime_frame_selector.py",
    "tools/bata/duca_paper_training.py",
    "tools/bata/run_duca_paper_short_window_gate.py",
    "tools/bata/validate_duca_paper_code_gate.py",
    "tools/bata/validate_duca_paper_short_window_gate.py",
)


class GateArtifactFailure(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateArtifactFailure(message)


def _path(value: Any) -> Path:
    return Path(str(value)).expanduser().resolve()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("ascii")
    ).hexdigest()


def _hashed_asset(payload: Mapping[str, Any], label: str) -> None:
    path = _path(payload.get("path", ""))
    _require(path.is_file(), f"bound {label} is missing")
    _require(payload.get("sha256") == _sha256(path), f"bound {label} hash drift")


def validate_gate_artifact(
    path: str | Path,
    *,
    expected_commit: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    artifact = _path(path)
    _require(artifact.is_file(), "short-window gate receipt is missing")
    artifact_sha = _sha256(artifact)
    if expected_sha256 is not None:
        _require(artifact_sha == str(expected_sha256), "gate receipt SHA-256 mismatch")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "gate receipt is not an object")
    unsigned = dict(payload)
    content_sha = unsigned.pop("content_sha256", None)
    _require(content_sha == _canonical_sha256(unsigned), "gate content self-hash drift")
    _require(payload.get("schema_version") == SCHEMA, "gate schema drift")
    _require(payload.get("status") == "passed" and payload.get("fail_closed") is True, "gate did not pass")
    _require(payload.get("git_commit") == expected_commit, "gate commit is stale")
    _require(payload.get("synthetic_inputs_used") is False, "gate used synthetic inputs")
    _require(payload.get("validation_or_test_data_used") is False, "gate consumed validation/test data")
    _require(payload.get("input_provenance") == "real_thumos14_full200_train_video_decode", "gate provenance drift")
    _require(payload.get("mixed_training_mode") is True, "gate did not use mixed training mode")
    _require(payload.get("selector_to_unique_gather_to_heavy_backbone_completed") is True, "full execution chain is missing")
    _require(payload.get("requested_budget_order") == list(REQUESTED_BUDGETS), "requested budget order drift")
    code_gate = payload.get("prerequisite_clean_linux_code_gate", {})
    _require(isinstance(code_gate, Mapping), "short-window gate lacks its clean Linux prerequisite")
    code_gate_binding = validate_code_gate_artifact(
        code_gate.get("path", ""),
        expected_commit=expected_commit,
        expected_sha256=code_gate.get("sha256"),
    )
    _require(
        code_gate_binding["path"] == str(_path(code_gate.get("path", ""))),
        "clean Linux code-gate path drift",
    )

    config = payload.get("config", {})
    _require(_path(config.get("path", "")) == (ROOT / CONFIG_DEFAULT).resolve(), "gate used another config")
    _require(config.get("sha256") == _sha256(ROOT / CONFIG_DEFAULT), "gate config hash drift")
    _require(config.get("arm") == "uniform_mixed_train_k384_eval", "gate arm drift")
    assets = payload.get("assets", {})
    for key in ("pretrain", "annotation", "class_map"):
        _require(isinstance(assets.get(key), Mapping), f"gate lacks {key} binding")
        _hashed_asset(assets[key], key)
    _require(_path(assets.get("train_data_path", "")).is_dir(), "gate training video directory vanished")

    audited = payload.get("audited_file_sha256", {})
    _require(set(audited) == set(AUDITED_PATHS), "audited surface drift")
    for relative in AUDITED_PATHS:
        _require(audited.get(relative) == _sha256(ROOT / relative), f"audited hash drift: {relative}")
    dataset = payload.get("dataset", {})
    _require(dataset.get("dataset_class") == "DucaStatelessThumosPaddingDataset", "real dataset class drift")
    _require(int(dataset.get("dataset_size", -1)) == 200, "gate did not enumerate full200")
    _require(int(dataset.get("natural_short_count", 0)) > 0, "gate lacks a natural short sample")
    _require(int(dataset.get("subquantum_count", -1)) == 0, "gate silently accepted subquantum samples")
    selected_sample = dataset.get("selected_sample", {})
    _require(selected_sample.get("video_exists") is True, "selected real video is missing")
    _require(int(selected_sample.get("annotation_valid_length", 9999)) < 512, "selected sample is not naturally short")

    executions = payload.get("executions")
    _require(isinstance(executions, list) and len(executions) == 4, "gate execution count drift")
    for requested, row in zip(REQUESTED_BUDGETS, executions):
        _require(isinstance(row, Mapping), "gate execution row is malformed")
        valid_length = int(row.get("dense_valid_length", -1))
        expected_effective = min(requested, (valid_length // 16) * 16)
        selected = [int(value) for value in row.get("selected_dense_indices", ())]
        contract = row.get("backbone_input_contract", {})
        _require(valid_length >= 16, "gate execution is subquantum")
        _require(int(row.get("requested_k", -1)) == requested, "requested K drift")
        _require(
            int(row.get("effective_k", -1))
            == int(row.get("unique_k", -1))
            == int(row.get("backbone_input_k", -1))
            == expected_effective,
            "requested/effective/unique/backbone equality drift",
        )
        _require(selected == sorted(set(selected)) and len(selected) == expected_effective, "selected positions drift")
        _require(all(0 <= value < valid_length for value in selected), "invalid selected position")
        _require(row.get("no_padding") is True and row.get("no_repetition") is True, "padding/repetition detected")
        _require(row.get("no_invalid_index") is True and row.get("heavy_backbone_forward_completed") is True, "execution proof incomplete")
        _require(
            contract.get("schema_version") == "duca_dynamic_backbone_input_v1"
            and contract.get("measurement_source")
            == "actual_backbone_wrapper_and_videomae_input_tensors"
            and int(contract.get("wrapper_temporal_k", -1)) == expected_effective
            and int(contract.get("inner_reconstructed_k", -1)) == expected_effective
            and contract.get("padding_or_repetition_observed") is False,
            "actual backbone boundary contract drift",
        )
        _require(
            row.get("selected_dense_indices_sha256") == _canonical_sha256(selected),
            "selected-position hash drift",
        )

    slurm = payload.get("slurm_cuda_binding", {})
    _require(str(slurm.get("slurm_job_id", "")).isdigit(), "numeric Slurm job id is missing")
    _require(slurm.get("logical_device") == "cuda:0", "gate did not use logical cuda:0")
    _require(slurm.get("logical_cuda_device_count") == 1, "gate did not use one logical GPU")
    _require(slurm.get("physical_gpu_index_assumed") is False, "gate assumed a physical GPU")
    final = payload.get("final_clean_binding", {})
    for key in ("git_commit_unchanged", "git_tree_unchanged", "git_tree_clean_after_gate"):
        _require(final.get(key) is True, f"final clean binding failed: {key}")
    _require(payload.get("paper_metric_claim_allowed") is False, "gate overclaims metric evidence")
    _require(payload.get("paper_method_performance_evidence") is False, "gate overclaims method evidence")
    _require(payload.get("claim_scope") == "engineering_short_window_execution_only", "claim scope drift")
    _require(payload.get("stage_a_rerun_required") is True, "gate improperly replaces Stage-A")
    _require(payload.get("stage_b_enabled") is False, "gate opened Stage B")
    _require(payload.get("official_final_consumed") is False, "gate consumed official-final")
    forbidden = {"metrics", "mAP", "average_mAP", "loss", "predictions"}
    _require(not forbidden.intersection(payload), "gate receipt contains forbidden performance fields")
    _require(re.fullmatch(r"[0-9a-f]{64}", str(content_sha)) is not None, "invalid content hash")
    return {
        "path": str(artifact),
        "sha256": artifact_sha,
        "git_commit": expected_commit,
        "slurm_job_id": str(slurm["slurm_job_id"]),
        "status": "passed",
        "claim_scope": payload["claim_scope"],
        "code_gate_path": code_gate_binding["path"],
        "code_gate_sha256": code_gate_binding["sha256"],
        "code_gate_slurm_job_id": code_gate_binding["slurm_job_id"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    validate_gate_artifact(
        args.receipt,
        expected_commit=args.expected_commit,
        expected_sha256=args.expected_sha256,
    )
    print("ENGINEERING_STATUS DUCA paper short-window gate receipt validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
