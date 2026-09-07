"""Seal an existing official BAFDR receipt without changing its provenance."""

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path


def receipt_hash(payload):
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def validate_seal(payload):
    if payload.get("receipt_sha256") != receipt_hash(payload):
        raise ValueError("BAFDR receipt self-hash mismatch")


def build_seal(evaluation, training, provenance):
    for receipt in (evaluation, training):
        if receipt.get("protocol_id") != "ZOOMTOKEN-BA-FDR-K16-FULLMATRIX-v001":
            raise ValueError("Unexpected BAFDR protocol")
        if (receipt.get("expected_epochs"), receipt.get("expected_total_updates"),
                receipt.get("total_successful_updates")) != (60, 6000, 6000):
            raise ValueError("BAFDR terminal update contract is incomplete")
    if evaluation.get("phase") != "metric_opening" or evaluation.get("metric_opened") is not True:
        raise ValueError("An official metric-opening receipt is required")
    if training.get("phase") != "training":
        raise ValueError("A training receipt is required")
    for key in ("arm", "seed", "checkpoint", "checkpoint_sha256", "teacher_identity"):
        if evaluation.get(key) != training.get(key):
            raise ValueError(f"Training/evaluation identity mismatch: {key}")
    metrics = evaluation.get("eval_results", {})
    for key in ("average_mAP", "mAP@0.3", "mAP@0.4", "mAP@0.5", "mAP@0.6", "mAP@0.7"):
        if key not in metrics or not 0 <= float(metrics[key]) <= 1:
            raise ValueError(f"Missing or invalid official metric: {key}")
    sealed = dict(evaluation)
    sealed["retrospective_sealing"] = dict(provenance, training_receipt=training)
    sealed["receipt_sha256"] = receipt_hash(sealed)
    validate_seal(sealed)
    return sealed


def verify_file(path, expected):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Recorded artifact hash mismatch: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-evaluator-sha", required=True)
    parser.add_argument("--expected-training-sha", required=True)
    parser.add_argument("--precheck-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip():
        raise ValueError("Receipt sealer checkout must be clean")
    if args.output.exists() or args.source.resolve() == args.output.resolve():
        raise ValueError("Seal requires a new output path; existing evidence is read-only")
    evaluation = json.loads(args.source.read_text(encoding="utf-8"))
    train_path = args.source.with_name("train_receipt.json")
    training = json.loads(train_path.read_text(encoding="utf-8"))
    if evaluation["commit_sha"] != args.expected_evaluator_sha or training["commit_sha"] != args.expected_training_sha:
        raise ValueError("Training/evaluator commit mismatch")
    for receipt in (evaluation, training):
        verify_file(receipt["config"], receipt["config_sha256"])
    verify_file(evaluation["checkpoint"], evaluation["checkpoint_sha256"])
    teacher = evaluation.get("teacher_identity", {})
    if teacher:
        verify_file(teacher["teacher_config"], teacher["teacher_config_sha256"])
        verify_file(teacher["teacher_checkpoint"], teacher["teacher_checkpoint_sha256"])
    import torch
    checkpoint = torch.load(evaluation["checkpoint"], map_location="cpu")
    if checkpoint.get("epoch") != 59 or not checkpoint.get("state_dict_ema"):
        raise ValueError("Expected terminal epoch-59 EMA checkpoint")
    if checkpoint.get("total_successful_updates") != 6000:
        raise ValueError("Checkpoint successful-update count is not 6000")
    sealed = build_seal(evaluation, training, {
        "source_receipt": str(args.source),
        "sealing_commit": commit,
        "sealing_clean_tree": True,
        "sealed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint_epoch": 59,
        "checkpoint_state_key": "state_dict_ema",
        "artifact_bindings_verified": True,
        "inference_repeated": False,
    })
    if args.precheck_only:
        print(f"PRECHECK PASS {evaluation['arm']}: original receipts/artifacts preserved")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(sealed, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    validate_seal(json.loads(args.output.read_text(encoding="utf-8")))
    print(f"SEALED {evaluation['arm']} {args.output}")


if __name__ == "__main__":
    main()
