#!/usr/bin/env python
"""Replay a frozen d3_all checkpoint with deterministic K384 residual support."""

import argparse
import hashlib
import json
import os
import pickle
import subprocess

import numpy as np
import torch
import torch.nn as nn

from libs.core import load_config
from libs.datasets import make_data_loader, make_dataset
from libs.modeling import make_meta_arch
from libs.utils import ANETdetection, fix_random_seed, valid_one_epoch
from tools.evaluate_odfcr_internal_predictions import (
    _load_and_validate_manifest,
    _validate_predictions,
)


SCHEMA_VERSION = "actionformer_odfcr_k384_replay_v1"
BUDGET = 384
POLICY = "stratified_uniform"
HASH_SEED = 2026073100


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fid:
        for chunk in iter(lambda: fid.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_root():
    return os.path.realpath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
    )


def _git_value(repository, *args):
    return subprocess.check_output(
        ("git", "-C", repository) + args,
        text=True,
        encoding="utf-8",
    ).strip()


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fid:
        return json.load(fid)


def _canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _load_and_validate_predictions(path, holdout_ids):
    with open(path, "rb") as fid:
        predictions = pickle.load(fid)
    _validate_predictions(predictions, holdout_ids)
    return predictions


class AllocationRecorder:
    def __init__(self):
        self.records = {}

    def __call__(self, fpn_masks, selected_masks, video_ids):
        if video_ids is None or len(video_ids) != fpn_masks[0].shape[0]:
            raise RuntimeError("K384 allocation audit requires video IDs")
        for batch_idx, video_id in enumerate(video_ids):
            if video_id in self.records:
                raise RuntimeError("duplicate K384 allocation video ID")
            valid_counts = [
                int(mask[batch_idx, 0].sum().item()) for mask in fpn_masks
            ]
            selected_indices = [
                selected[batch_idx, 0]
                .nonzero(as_tuple=True)[0]
                .detach()
                .cpu()
                .tolist()
                for selected in selected_masks
            ]
            selected_count = sum(len(indices) for indices in selected_indices)
            if selected_count != min(BUDGET, sum(valid_counts)):
                raise RuntimeError("K384 allocation violates exact budget")
            allocation = {
                "video_id": str(video_id),
                "valid_counts_per_level": valid_counts,
                "selected_indices_per_level": selected_indices,
                "selected_count": selected_count,
            }
            allocation["allocation_sha256"] = _canonical_sha256(allocation)
            self.records[str(video_id)] = allocation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--previous-manifest", required=True)
    parser.add_argument("--expected-previous-manifest-sha256", required=True)
    parser.add_argument("--source-matrix-receipt", required=True)
    parser.add_argument("--expected-matrix-receipt-sha256", required=True)
    parser.add_argument("--g2-aggregate", required=True)
    parser.add_argument("--expected-g2-aggregate-sha256", required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--expected-code-tree", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--predictions-output", required=True)
    parser.add_argument("--receipt-output", required=True)
    parser.add_argument("--print-freq", type=int, default=10)
    args = parser.parse_args()
    for path in (args.predictions_output, args.receipt_output):
        if os.path.exists(path) or os.path.exists(path + ".tmp"):
            raise FileExistsError("refusing to overwrite K384 replay output")
    for path, expected, description in (
        (args.config, args.expected_config_sha256, "config"),
        (args.checkpoint, args.expected_checkpoint_sha256, "checkpoint"),
        (args.manifest, args.expected_manifest_sha256, "manifest"),
        (
            args.previous_manifest,
            args.expected_previous_manifest_sha256,
            "previous manifest",
        ),
        (
            args.source_matrix_receipt,
            args.expected_matrix_receipt_sha256,
            "source matrix receipt",
        ),
        (
            args.g2_aggregate,
            args.expected_g2_aggregate_sha256,
            "G2 aggregate",
        ),
    ):
        if _sha256(path) != expected:
            raise ValueError(description + " SHA-256 mismatch")
    repository = _repository_root()
    if (
        _git_value(repository, "rev-parse", "HEAD")
        != args.expected_code_commit
        or _git_value(repository, "rev-parse", "HEAD^{tree}")
        != args.expected_code_tree
        or _git_value(repository, "status", "--porcelain")
    ):
        raise ValueError("K384 replay runtime identity mismatch")
    annotation_path = load_config(args.config)["dataset"]["json_file"]
    annotation_path = os.path.expandvars(annotation_path)
    annotation_sha256 = _sha256(annotation_path)
    _, holdout_ids = _load_and_validate_manifest(
        args.manifest,
        args.previous_manifest,
        annotation_path,
        annotation_sha256,
    )
    matrix = _load_json(args.source_matrix_receipt)
    if (
        matrix.get("schema_version")
        != "actionformer_odfcr_internal_matrix_v1"
        or matrix.get("validation_pass") is not True
        or int(matrix.get("seed")) != args.seed
        or matrix.get("git_commit") != args.expected_code_commit
        or matrix.get("git_tree") != args.expected_code_tree
        or (
            matrix["receipts"]["internal_holdout_manifest"]["sha256"]
            != args.expected_manifest_sha256
        )
        or (
            matrix["receipts"]["previous_holdout_manifest"]["sha256"]
            != args.expected_previous_manifest_sha256
        )
        or (
            matrix["arm_artifact_bindings"]["d3_all"]["checkpoint_sha256"]
            != args.expected_checkpoint_sha256
        )
        or (
            matrix["arm_artifact_bindings"]["d3_all"]["config_sha256"]
            != args.expected_config_sha256
        )
        or matrix.get("paper_performance_row_allowed") is not False
        or matrix.get("official_test_authorized") is not False
    ):
        raise ValueError("source d3_all matrix receipt contract mismatch")
    g2 = _load_json(args.g2_aggregate)
    if (
        g2.get("schema_version")
        != "actionformer_odfcr_g2_internal_aggregate_v1"
        or g2.get("validation_pass") is not True
        or g2.get("residual_utility_gate_pass") is not True
        or g2.get("git_commit") != args.expected_code_commit
        or g2.get("git_tree") != args.expected_code_tree
        or g2.get("manifest_sha256") != args.expected_manifest_sha256
        or g2.get("previous_manifest_sha256")
        != args.expected_previous_manifest_sha256
        or args.expected_matrix_receipt_sha256
        not in {
            receipt.get("sha256")
            for receipt in g2.get("matrix_receipts", [])
        }
    ):
        raise ValueError("G2 aggregate did not authorize this K384 replay")

    cfg = load_config(args.config)
    odfcr = cfg["model"].get("odfcr_head", {})
    if (
        cfg["train_split"] != ["validation"]
        or cfg["val_split"] != ["validation"]
        or odfcr.get("mode") != "official_dense_floor_factorial"
        or odfcr.get("scaffold_num_layers") != 3
        or odfcr.get("residual_enabled") is not True
        or odfcr.get("residual_execution_support") != "all_valid"
        or odfcr.get("residual_num_layers") != 3
        or odfcr.get("training_loss_support")
        != "official_all_valid_fpn_queries"
    ):
        raise ValueError("config is not the frozen ODF-CR d3_all contract")
    _ = fix_random_seed(0, include_cuda=True)
    val_dataset = make_dataset(
        cfg["dataset_name"], False, cfg["val_split"], **cfg["dataset"]
    )
    if frozenset(val_dataset.evaluation_video_ids) != holdout_ids:
        raise ValueError("dataset/manifest holdout-v2 mismatch")
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg["loader"]["num_workers"]
    )
    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    model = nn.DataParallel(model, device_ids=cfg["devices"])
    checkpoint = torch.load(
        args.checkpoint,
        map_location=lambda storage, loc: storage.cuda(cfg["devices"][0]),
    )
    checkpoint_epoch = int(checkpoint["epoch"])
    model.load_state_dict(checkpoint["state_dict_ema"])
    del checkpoint
    model.eval()
    module = model.module
    module.configure_odfcr_frozen_replay(
        BUDGET, policy=POLICY, hash_seed=HASH_SEED
    )
    recorder = AllocationRecorder()
    module.odfcr_replay_query_selector.audit_callback = recorder

    temporary_predictions = args.predictions_output + ".tmp"
    os.makedirs(
        os.path.dirname(os.path.abspath(args.predictions_output)),
        exist_ok=True,
    )
    valid_one_epoch(
        val_loader,
        model,
        -1,
        evaluator=None,
        output_file=temporary_predictions,
        ext_score_file=cfg["test_cfg"]["ext_score_file"],
        tb_writer=None,
        print_freq=args.print_freq,
    )
    predictions = _load_and_validate_predictions(
        temporary_predictions, holdout_ids
    )
    if frozenset(recorder.records) != holdout_ids:
        raise ValueError("K384 allocation ledger does not cover holdout-v2")
    allocation_records = [
        recorder.records[video_id] for video_id in sorted(recorder.records)
    ]
    allocation_ledger_sha256 = _canonical_sha256(allocation_records)
    evaluator = ANETdetection(
        val_dataset.json_file,
        split="validation",
        tiou_thresholds=np.linspace(0.3, 0.7, 5),
        video_ids=holdout_ids,
    )
    mAP, average_mAP, _ = evaluator.evaluate(predictions, verbose=True)
    os.replace(temporary_predictions, args.predictions_output)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "validation_pass": True,
        "training_performed": False,
        "checkpoint_changed": False,
        "decoder_or_nms_changed": False,
        "paper_performance_row_allowed": False,
        "official_test_authorized": False,
        "efficiency_claim_allowed": False,
        "source_split": "validation",
        "test_gt_used": False,
        "test_predictions_used": False,
        "seed": args.seed,
        "git_commit": args.expected_code_commit,
        "git_tree": args.expected_code_tree,
        "checkpoint": {
            "path": os.path.realpath(args.checkpoint),
            "sha256": _sha256(args.checkpoint),
            "epoch": checkpoint_epoch,
            "ema_state_used": True,
        },
        "config": {
            "path": os.path.realpath(args.config),
            "sha256": _sha256(args.config),
        },
        "manifest_sha256": _sha256(args.manifest),
        "previous_manifest_sha256": _sha256(args.previous_manifest),
        "source_matrix_receipt_sha256": _sha256(
            args.source_matrix_receipt
        ),
        "g2_aggregate_sha256": _sha256(args.g2_aggregate),
        "selector": {
            "policy": POLICY,
            "budget": BUDGET,
            "hash_seed": HASH_SEED,
            "per_video_count_contract": "min(384, valid_query_count)",
            "allocation_video_count": len(allocation_records),
            "allocation_ledger_sha256": allocation_ledger_sha256,
            "allocation_records": allocation_records,
        },
        "predictions": {
            "path": os.path.realpath(args.predictions_output),
            "sha256": _sha256(args.predictions_output),
            "count": len(predictions["video-id"]),
            "video_count_with_detections": len(
                frozenset(predictions["video-id"])
            ),
        },
        "metrics": {
            "average_mAP": float(average_mAP),
            "mAP_at_0_3_to_0_7": [float(value) for value in mAP],
        },
    }
    temporary_receipt = args.receipt_output + ".tmp"
    os.makedirs(
        os.path.dirname(os.path.abspath(args.receipt_output)), exist_ok=True
    )
    with open(temporary_receipt, "x", encoding="utf-8") as fid:
        json.dump(payload, fid, indent=2, sort_keys=True)
        fid.write("\n")
    os.replace(temporary_receipt, args.receipt_output)
    print(json.dumps(payload["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
