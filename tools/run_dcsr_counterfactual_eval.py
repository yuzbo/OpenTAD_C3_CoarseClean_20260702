#!/usr/bin/env python
"""Run read-only DCSR checkpoint counterfactuals on the frozen dev holdout.

The tool never trains and never reads the THUMOS test split.  It changes only
the inference-time residual execution support of an already trained G1 model:

* ``scaffold_only`` disables the residual branch after loading the checkpoint.
* ``k384_reference`` preserves the preregistered G1 execution contract.
* ``all_query_residual`` executes the trained residual on every valid query.

Every output is diagnostic-only and explicitly ineligible for a paper row.
"""

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


SCHEMA_VERSION = "actionformer_dcsr_counterfactual_v1"
ALL_QUERY_BUDGET = 2 ** 31 - 1


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


def _validate_inputs(args):
    for path, expected, description in (
        (args.config, args.expected_config_sha256, "config"),
        (args.manifest, args.expected_manifest_sha256, "manifest"),
        (
            args.source_pair_receipt,
            args.expected_pair_receipt_sha256,
            "source pair receipt",
        ),
    ):
        if _sha256(path) != expected:
            raise ValueError(description + " SHA-256 mismatch")
    manifest = _load_json(args.manifest)
    if (
        manifest.get("schema_version")
        != "actionformer_dcsr_internal_holdout_v1"
        or manifest.get("source_split") != "validation"
        or manifest.get("test_annotations_used") is not False
        or manifest.get("test_records_selected") is not False
    ):
        raise ValueError("invalid validation-only holdout manifest")
    pair = _load_json(args.source_pair_receipt)
    if (
        pair.get("schema_version") != "actionformer_dcsr_internal_pair_v1"
        or pair.get("validation_pass") is not True
        or pair.get("g0_exact_equivalence_pass") is not True
        or pair.get("seed") != args.seed
        or pair.get("git_commit") != args.source_training_commit
        or pair.get("git_tree") != args.source_training_tree
        or pair.get("source_split") != "validation"
        or pair.get("test_gt_used") is not False
        or pair.get("test_predictions_used") is not False
    ):
        raise ValueError("source G1 pair receipt contract mismatch")
    repository = _repository_root()
    if (
        _git_value(repository, "rev-parse", "HEAD")
        != args.expected_code_commit
    ):
        raise ValueError("diagnostic runtime commit mismatch")
    if (
        _git_value(repository, "rev-parse", "HEAD^{tree}")
        != args.expected_code_tree
    ):
        raise ValueError("diagnostic runtime tree mismatch")
    if _git_value(repository, "status", "--porcelain"):
        raise ValueError("diagnostic runtime is not clean")
    return manifest, pair


def apply_counterfactual(model, arm):
    """Apply one diagnostic arm to a loaded DCSR model and return its receipt."""
    module = model.module if isinstance(model, nn.DataParallel) else model
    if module.dcsr_mode != "cheap_dense_scaffold":
        raise ValueError("counterfactual requires a cheap_dense_scaffold model")
    if module.dcsr_scaffold_num_layers != 1:
        raise ValueError("counterfactual source must be the frozen G1 scaffold")
    original_budget = module.dcsr_query_selector.budget
    original_residual_enabled = module.dcsr_residual_enabled
    if original_budget != 384 or original_residual_enabled is not True:
        raise ValueError("counterfactual source is not the frozen K384 G1 model")

    if arm == "scaffold_only":
        module.dcsr_residual_enabled = False
    elif arm == "k384_reference":
        pass
    elif arm == "all_query_residual":
        module.dcsr_query_selector.budget = ALL_QUERY_BUDGET
    else:
        raise ValueError("unsupported DCSR counterfactual arm")
    return {
        "arm": arm,
        "original_budget": original_budget,
        "effective_budget": module.dcsr_query_selector.budget,
        "original_residual_enabled": original_residual_enabled,
        "effective_residual_enabled": module.dcsr_residual_enabled,
        "all_valid_queries_selected_by_budget_contract": (
            arm == "all_query_residual"
        ),
        "training_changed": False,
        "checkpoint_changed": False,
        "decoder_or_nms_changed": False,
    }


def _validate_predictions(path, holdout_ids):
    with open(path, "rb") as fid:
        predictions = pickle.load(fid)
    if set(predictions) != {
        "video-id",
        "t-start",
        "t-end",
        "label",
        "score",
    }:
        raise ValueError("unexpected raw prediction schema")
    lengths = {len(value) for value in predictions.values()}
    if len(lengths) != 1:
        raise ValueError("raw prediction arrays have inconsistent lengths")
    unexpected = frozenset(predictions["video-id"]) - holdout_ids
    if unexpected:
        raise ValueError("counterfactual predictions escaped the holdout")
    for key in ("t-start", "t-end", "score"):
        if not np.isfinite(np.asarray(predictions[key])).all():
            raise ValueError("non-finite values in " + key)
    if np.any(
        np.asarray(predictions["t-end"])
        <= np.asarray(predictions["t-start"])
    ):
        raise ValueError("counterfactual emitted a non-positive segment")
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--source-pair-receipt", required=True)
    parser.add_argument("--expected-pair-receipt-sha256", required=True)
    parser.add_argument("--source-training-commit", required=True)
    parser.add_argument("--source-training-tree", required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--expected-code-tree", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--arm",
        required=True,
        choices=("scaffold_only", "k384_reference", "all_query_residual"),
    )
    parser.add_argument("--predictions-output", required=True)
    parser.add_argument("--receipt-output", required=True)
    parser.add_argument("--print-freq", type=int, default=10)
    args = parser.parse_args()

    for path in (args.predictions_output, args.receipt_output):
        if os.path.exists(path) or os.path.exists(path + ".tmp"):
            raise FileExistsError("refusing to overwrite diagnostic output")
    manifest, _ = _validate_inputs(args)
    holdout_ids = frozenset(manifest["holdout_video_ids"])

    cfg = load_config(args.config)
    dcsr_cfg = cfg["model"].get("dcsr_head", {})
    if (
        cfg["train_split"] != ["validation"]
        or cfg["val_split"] != ["validation"]
        or dcsr_cfg.get("mode") != "cheap_dense_scaffold"
        or dcsr_cfg.get("budget") != 384
        or dcsr_cfg.get("policy") != "stratified_uniform"
        or dcsr_cfg.get("scaffold_num_layers") != 1
        or dcsr_cfg.get("residual_enabled") is not True
        or dcsr_cfg.get("training_loss_support")
        != "official_all_valid_fpn_queries"
    ):
        raise ValueError("config is not the frozen DCSR G1 contract")

    _ = fix_random_seed(0, include_cuda=True)
    val_dataset = make_dataset(
        cfg["dataset_name"], False, cfg["val_split"], **cfg["dataset"]
    )
    if frozenset(val_dataset.evaluation_video_ids) != holdout_ids:
        raise ValueError("dataset/manifest holdout mismatch")
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg["loader"]["num_workers"]
    )
    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    model = nn.DataParallel(model, device_ids=cfg["devices"])

    checkpoint_sha256 = _sha256(args.checkpoint)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=lambda storage, loc: storage.cuda(cfg["devices"][0]),
    )
    checkpoint_epoch = int(checkpoint["epoch"])
    model.load_state_dict(checkpoint["state_dict_ema"])
    del checkpoint
    arm_receipt = apply_counterfactual(model, args.arm)

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
    predictions = _validate_predictions(temporary_predictions, holdout_ids)
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
        "diagnostic_only": True,
        "model_selection_role": "post_negative_result_diagnostic_only",
        "paper_performance_row_allowed": False,
        "efficiency_claim_allowed": False,
        "source_split": "validation",
        "holdout_only": True,
        "test_gt_used": False,
        "test_predictions_used": False,
        "seed": args.seed,
        "source_training_commit": args.source_training_commit,
        "source_training_tree": args.source_training_tree,
        "diagnostic_code_commit": args.expected_code_commit,
        "diagnostic_code_tree": args.expected_code_tree,
        "diagnostic_runtime": _repository_root(),
        "checkpoint": {
            "path": os.path.realpath(args.checkpoint),
            "sha256": checkpoint_sha256,
            "epoch": checkpoint_epoch,
            "ema_state_used": True,
        },
        "config": {
            "path": os.path.realpath(args.config),
            "sha256": _sha256(args.config),
        },
        "manifest": {
            "path": os.path.realpath(args.manifest),
            "sha256": _sha256(args.manifest),
            "holdout_video_count": len(holdout_ids),
        },
        "source_pair_receipt": {
            "path": os.path.realpath(args.source_pair_receipt),
            "sha256": _sha256(args.source_pair_receipt),
        },
        "counterfactual": arm_receipt,
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
