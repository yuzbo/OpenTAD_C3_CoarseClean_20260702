#!/usr/bin/env python
"""Measure branch learning dynamics in completed DCSR G1 checkpoints.

The script is CPU-only and read-only.  It does not instantiate the model,
evaluate predictions, or touch either THUMOS split.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping

import torch


SCHEMA_VERSION = "actionformer_dcsr_checkpoint_dynamics_v1"
EXPECTED_EPOCHS = (5, 10, 15, 20, 25, 30, 35)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fid:
        for chunk in iter(lambda: fid.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fid:
        return json.load(fid)


def parameter_group(name):
    """Return the diagnostic head group for one checkpoint parameter."""
    normalized = name[7:] if name.startswith("module.") else name
    if normalized.startswith("dcsr_scaffold_cls_head."):
        return "scaffold_classification"
    if normalized.startswith("dcsr_scaffold_reg_head."):
        return "scaffold_regression"
    if re.fullmatch(
        r"cls_head\.cls_head\.conv\.(weight|bias)", normalized
    ):
        return "residual_classification_final"
    if re.fullmatch(
        r"reg_head\.offset_head\.conv\.(weight|bias)", normalized
    ):
        return "residual_regression_final"
    if normalized.startswith("cls_head."):
        return "residual_classification_hidden"
    if normalized.startswith("reg_head."):
        return "residual_regression_hidden"
    return None


def _group_tensors(state_dict):
    groups = {}
    for name, value in state_dict.items():
        group = parameter_group(name)
        if group is None or not torch.is_floating_point(value):
            continue
        groups.setdefault(group, {})[name] = value.detach().cpu().float()
    expected = {
        "scaffold_classification",
        "scaffold_regression",
        "residual_classification_final",
        "residual_regression_final",
        "residual_classification_hidden",
        "residual_regression_hidden",
    }
    if set(groups) != expected:
        raise ValueError(
            "unexpected DCSR parameter groups: {:s}".format(
                ", ".join(sorted(groups))
            )
        )
    return groups


def _stats(tensors):
    numel = sum(value.numel() for value in tensors.values())
    squared = sum(float(torch.sum(value * value)) for value in tensors.values())
    absolute_sum = sum(
        float(torch.sum(torch.abs(value))) for value in tensors.values()
    )
    max_abs = max(
        float(torch.max(torch.abs(value))) for value in tensors.values()
    )
    zeros = sum(int(torch.sum(value == 0).item()) for value in tensors.values())
    return {
        "parameter_tensor_count": len(tensors),
        "numel": numel,
        "l2_norm": math.sqrt(squared),
        "abs_mean": absolute_sum / numel,
        "max_abs": max_abs,
        "exact_zero_fraction": float(zeros) / float(numel),
    }


def _difference(left, right):
    if set(left) != set(right):
        raise ValueError("parameter keys changed between compared states")
    differences = {
        name: left[name] - right[name]
        for name in left
    }
    result = _stats(differences)
    reference = _stats(right)["l2_norm"]
    result["relative_l2_to_reference"] = (
        result["l2_norm"] / reference if reference > 0 else None
    )
    return result


def summarize_checkpoint(path, previous_groups=None):
    try:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if (
        not isinstance(checkpoint, Mapping)
        or "state_dict" not in checkpoint
        or "state_dict_ema" not in checkpoint
        or "epoch" not in checkpoint
        or not isinstance(checkpoint["state_dict"], Mapping)
        or not isinstance(checkpoint["state_dict_ema"], Mapping)
    ):
        raise ValueError("checkpoint lacks model/EMA state")
    groups = _group_tensors(checkpoint["state_dict"])
    ema_groups = _group_tensors(checkpoint["state_dict_ema"])
    summary = {
        "path": os.path.realpath(path),
        "sha256": _sha256(path),
        "epoch_field": int(checkpoint["epoch"]),
        "groups": {},
    }
    for group, tensors in groups.items():
        item = {
            "model": _stats(tensors),
            "ema": _stats(ema_groups[group]),
            "model_minus_ema": _difference(tensors, ema_groups[group]),
        }
        if previous_groups is not None:
            item["model_minus_previous_checkpoint"] = _difference(
                tensors, previous_groups[group]
            )
        summary["groups"][group] = item
    del checkpoint
    return summary, groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-root", required=True)
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--expected-aggregate-sha256", required=True)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if os.path.exists(args.output) or os.path.exists(args.output + ".tmp"):
        raise FileExistsError("refusing to overwrite checkpoint diagnostics")
    if _sha256(args.aggregate) != args.expected_aggregate_sha256:
        raise ValueError("G1 aggregate SHA-256 mismatch")
    aggregate = _load_json(args.aggregate)
    if (
        aggregate.get("schema_version")
        != "actionformer_dcsr_g1_internal_aggregate_v1"
        or aggregate.get("validation_pass") is not True
        or aggregate.get("g1_gate_pass") is not False
        or sorted(args.seeds) != aggregate["development_seeds"]
    ):
        raise ValueError("invalid frozen G1 aggregate")

    seed_summaries = []
    for seed in args.seeds:
        pattern = os.path.join(
            args.source_run_root,
            "seed_{:d}".format(seed),
            "work",
            "ckpt",
            "*_dcsr",
            "epoch_*.pth.tar",
        )
        paths = sorted(glob.glob(pattern))
        epochs = [
            int(re.search(r"epoch_(\d+)\.pth\.tar$", path).group(1))
            for path in paths
        ]
        if tuple(epochs) != EXPECTED_EPOCHS:
            raise ValueError("unexpected checkpoint epochs for seed")
        previous_groups = None
        checkpoints = []
        for epoch, path in zip(epochs, paths):
            summary, previous_groups = summarize_checkpoint(
                path, previous_groups=previous_groups
            )
            if summary["epoch_field"] != epoch:
                raise ValueError("checkpoint epoch field/path mismatch")
            checkpoints.append(summary)
        train_log = os.path.join(
            args.source_run_root,
            "seed_{:d}".format(seed),
            "dcsr",
            "train.log",
        )
        seed_summaries.append(
            {
                "seed": seed,
                "checkpoints": checkpoints,
                "train_log": {
                    "path": os.path.realpath(train_log),
                    "sha256": _sha256(train_log),
                },
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "validation_pass": True,
        "diagnostic_only": True,
        "training_performed": False,
        "predictions_or_metrics_read": False,
        "test_data_read": False,
        "paper_performance_row_allowed": False,
        "source_training_commit": aggregate["git_commit"],
        "source_training_tree": aggregate["git_tree"],
        "source_g1_aggregate": {
            "path": os.path.realpath(args.aggregate),
            "sha256": _sha256(args.aggregate),
            "g1_gate_pass": False,
        },
        "identifiability_limits": [
            "five_epoch_checkpoint_spacing_does_not_measure_first_step_gradients",
            "parameter_update_norms_do_not_prove_causal_optimization_failure",
            "gradient_norms_were_not_logged_during_the_completed_run",
        ],
        "seeds": seed_summaries,
    }
    temporary_path = args.output + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(temporary_path, "x", encoding="utf-8") as fid:
        json.dump(payload, fid, indent=2, sort_keys=True)
        fid.write("\n")
    os.replace(temporary_path, args.output)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "seed_count": len(seed_summaries),
                "epochs": list(EXPECTED_EPOCHS),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
