from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata.build_duca_rime_gate_records import _ledger_by_video, _metrics


PHASE0_SCHEMA = "duca_rime_phase0_source_manifest_v1"
O1_SCHEMA = "duca_rime_o1_source_manifest_v1"
O2_SCHEMA = "duca_rime_o2_source_manifest_v1"
PHASE0_REPLICATE_KINDS = {
    "deterministic_reexecution",
    "independent_training_seed",
}
O1_DETECTOR_TRAINING_EXPOSURES = {
    "mixed_k_registered_panel",
    "fixed_k_384_only",
}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
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
        ).encode("utf-8")
    ).hexdigest()


def _write_immutable(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    sealed = dict(payload)
    sealed["content_sha256"] = _canonical_sha256(sealed)
    text = json.dumps(sealed, indent=2, sort_keys=True) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to overwrite a different source manifest: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return {
        "path": str(target),
        "sha256": _sha256_file(target),
        "payload": sealed,
    }


def _terminal_checkpoint_sha256(metrics: Mapping[str, Any]) -> str:
    terminal_path = Path(str(metrics["terminal_evaluation_path"])).expanduser().resolve()
    if (
        not terminal_path.is_file()
        or _sha256_file(terminal_path) != str(metrics["terminal_evaluation_sha256"])
    ):
        raise ValueError("terminal evaluation binding drifted")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    checkpoint_sha256 = str(terminal.get("checkpoint_sha256", ""))
    if len(checkpoint_sha256) != 64:
        raise ValueError("terminal evaluation lacks a checkpoint identity")
    return checkpoint_sha256


def build_phase0_manifest(
    *,
    replicates: Sequence[Sequence[str]],
    output: str | Path,
) -> dict[str, Any]:
    entries = []
    identities = set()
    common_split = None
    common_checkpoint = None
    for raw in replicates:
        if len(raw) != 4:
            raise ValueError("Phase-0 replicate requires id, kind, metrics path, and SHA-256")
        replicate_id, replicate_kind, metrics_path, expected_sha256 = map(str, raw)
        if (
            not replicate_id
            or replicate_id in identities
            or replicate_kind not in PHASE0_REPLICATE_KINDS
        ):
            raise ValueError("invalid or duplicate Phase-0 replicate identity")
        identities.add(replicate_id)
        resolved, metrics = _metrics(metrics_path, expected_sha256)
        split = str(metrics["split_assignment_sha256"])
        checkpoint = _terminal_checkpoint_sha256(metrics)
        if common_split is None:
            common_split = split
            common_checkpoint = checkpoint
        elif split != common_split or checkpoint != common_checkpoint:
            raise ValueError("Phase-0 replicates must share split and checkpoint identity")
        entries.append(
            {
                "replicate_id": replicate_id,
                "replicate_kind": replicate_kind,
                "path": str(resolved),
                "sha256": _sha256_file(resolved),
                "evaluation_seed": int(metrics["seed"]),
            }
        )
    if len(entries) < 2:
        raise ValueError("Phase-0 requires at least two real evaluation artifacts")
    return _write_immutable(
        output,
        {
            "schema_version": PHASE0_SCHEMA,
            "uses_official_final": False,
            "split_assignment_sha256": common_split,
            "checkpoint_sha256": common_checkpoint,
            "replicate_kinds": sorted({entry["replicate_kind"] for entry in entries}),
            "replicates": entries,
            "claim_scope": (
                "deterministic_reproducibility_only"
                if {entry["replicate_kind"] for entry in entries}
                == {"deterministic_reexecution"}
                else "includes_independent_training_seed_variance"
            ),
        },
    )


def build_o1_manifest(
    *,
    evaluations: Sequence[Sequence[str]],
    mixed_k_detector_identity_sha256: str,
    detector_training_exposure: str,
    output: str | Path,
    training_receipt: str | Path | None = None,
    training_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    if len(str(mixed_k_detector_identity_sha256)) != 64:
        raise ValueError("O1 requires an exact mixed-K detector checkpoint identity")
    if detector_training_exposure not in O1_DETECTOR_TRAINING_EXPOSURES:
        raise ValueError("O1 detector training exposure is not registered")
    receipt_binding = None
    if detector_training_exposure == "mixed_k_registered_panel":
        if training_receipt is None or not training_receipt_sha256:
            raise ValueError(
                "formal O1 requires the mixed-K training receipt and SHA-256"
            )
        receipt_path = Path(training_receipt).expanduser().resolve()
        if (
            not receipt_path.is_file()
            or _sha256_file(receipt_path) != str(training_receipt_sha256)
        ):
            raise ValueError("mixed-K training receipt binding drifted")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("schema_version")
            != "duca_rime_phase2_mixed_k_training_receipt_v1"
            or receipt.get("status") != "passed"
            or receipt.get("arm") != "U-mixed-K"
            or receipt.get("detector_training_exposure")
            != "mixed_k_registered_panel"
            or receipt.get("checkpoint_sha256")
            != str(mixed_k_detector_identity_sha256)
            or int(receipt.get("successful_detector_updates", -1)) != 6000
            or receipt.get("uses_official_final") is not False
        ):
            raise ValueError("invalid mixed-K training receipt for formal O1")
        receipt_binding = {
            "path": str(receipt_path),
            "sha256": _sha256_file(receipt_path),
        }
    elif training_receipt is not None or training_receipt_sha256 is not None:
        raise ValueError(
            "fixed-K diagnostic O1 must not attach a mixed-K training receipt"
        )
    entries = []
    seen_budgets = set()
    common_split = None
    common_role = None
    for raw in evaluations:
        if len(raw) != 4:
            raise ValueError("O1 evaluation requires budget, cost, metrics path, and SHA-256")
        budget = int(raw[0])
        cost = float(raw[1])
        if budget <= 0 or budget in seen_budgets or not math.isfinite(cost) or cost <= 0.0:
            raise ValueError("invalid or duplicate O1 budget/cost")
        seen_budgets.add(budget)
        resolved, metrics = _metrics(raw[2], raw[3])
        if (
            int(round(float(metrics["target_mean_cost"]))) != budget
            or _terminal_checkpoint_sha256(metrics)
            != str(mixed_k_detector_identity_sha256)
        ):
            raise ValueError("O1 metric budget or detector identity drift")
        split = str(metrics["split_assignment_sha256"])
        role = str(metrics["split_role"])
        if common_split is None:
            common_split, common_role = split, role
        elif split != common_split or role != common_role:
            raise ValueError("O1 evaluations must share split role and assignment")
        entries.append(
            {
                "budget": budget,
                "measured_heavy_frame_cost": cost,
                "path": str(resolved),
                "sha256": _sha256_file(resolved),
            }
        )
    if len(entries) < 2:
        raise ValueError("O1 requires at least two budget evaluations")
    entries.sort(key=lambda row: int(row["budget"]))
    return _write_immutable(
        output,
        {
            "schema_version": O1_SCHEMA,
            "uses_official_final": False,
            "position_policy": "exact_uniform",
            "detector_training_exposure": detector_training_exposure,
            "mixed_k_detector_identity_sha256": str(
                mixed_k_detector_identity_sha256
            ),
            "mixed_k_training_receipt": receipt_binding,
            "split_assignment_sha256": common_split,
            "split_role": common_role,
            "budget_evaluations": entries,
            "claim_scope": (
                "formal_o1_mixed_k_headroom"
                if detector_training_exposure == "mixed_k_registered_panel"
                else "diagnostic_cross_budget_transfer_from_fixed_k_384_only"
            ),
        },
    )


def build_o2_manifest(
    *,
    evaluations: Sequence[Sequence[str]],
    output: str | Path,
) -> dict[str, Any]:
    entries = []
    seen = set()
    common_split = None
    common_role = None
    for raw in evaluations:
        if len(raw) != 6:
            raise ValueError(
                "O2 evaluation requires family, budget, metrics path/SHA, ledger path/SHA"
            )
        family = str(raw[0])
        budget = int(raw[1])
        key = (family, budget)
        if not family or budget <= 0 or key in seen:
            raise ValueError("invalid or duplicate O2 family/budget")
        seen.add(key)
        metrics_path, metrics = _metrics(raw[2], raw[3])
        if int(round(float(metrics["target_mean_cost"]))) != budget:
            raise ValueError("O2 metric budget drift")
        _ledger_by_video(raw[4], raw[5], budget=budget)
        split = str(metrics["split_assignment_sha256"])
        role = str(metrics["split_role"])
        if common_split is None:
            common_split, common_role = split, role
        elif split != common_split or role != common_role:
            raise ValueError("O2 evaluations must share split role and assignment")
        entries.append(
            {
                "family": family,
                "budget": budget,
                "metrics_path": str(metrics_path),
                "metrics_sha256": _sha256_file(metrics_path),
                "ledger_path": str(Path(raw[4]).expanduser().resolve()),
                "ledger_sha256": _sha256_file(raw[4]),
            }
        )
    families = {family for family, _budget in seen}
    budgets = {budget for _family, budget in seen}
    if (
        "independent" not in families
        or len(families) < 2
        or len(budgets) < 2
        or seen != {(family, budget) for family in families for budget in budgets}
    ):
        raise ValueError("O2 requires a rectangular panel including independent")
    entries.sort(key=lambda row: (str(row["family"]), int(row["budget"])))
    return _write_immutable(
        output,
        {
            "schema_version": O2_SCHEMA,
            "uses_official_final": False,
            "split_assignment_sha256": common_split,
            "split_role": common_role,
            "decoder_evaluations": entries,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build hash-bound DUCA-RIME Phase-0/O1/O2 source manifests."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    phase0 = subparsers.add_parser("phase0")
    phase0.add_argument(
        "--replicate",
        action="append",
        nargs=4,
        metavar=("ID", "KIND", "METRICS", "SHA256"),
        required=True,
    )
    phase0.add_argument("--output", required=True)

    o1 = subparsers.add_parser("o1")
    o1.add_argument(
        "--evaluation",
        action="append",
        nargs=4,
        metavar=("BUDGET", "COST", "METRICS", "SHA256"),
        required=True,
    )
    o1.add_argument("--mixed-k-detector-identity-sha256", required=True)
    o1.add_argument(
        "--detector-training-exposure",
        choices=sorted(O1_DETECTOR_TRAINING_EXPOSURES),
        required=True,
    )
    o1.add_argument("--training-receipt")
    o1.add_argument("--training-receipt-sha256")
    o1.add_argument("--output", required=True)

    o2 = subparsers.add_parser("o2")
    o2.add_argument(
        "--evaluation",
        action="append",
        nargs=6,
        metavar=(
            "FAMILY",
            "BUDGET",
            "METRICS",
            "METRICS_SHA256",
            "LEDGER",
            "LEDGER_SHA256",
        ),
        required=True,
    )
    o2.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command == "phase0":
        result = build_phase0_manifest(replicates=args.replicate, output=args.output)
    elif args.command == "o1":
        result = build_o1_manifest(
            evaluations=args.evaluation,
            mixed_k_detector_identity_sha256=(
                args.mixed_k_detector_identity_sha256
            ),
            detector_training_exposure=args.detector_training_exposure,
            training_receipt=args.training_receipt,
            training_receipt_sha256=args.training_receipt_sha256,
            output=args.output,
        )
    else:
        result = build_o2_manifest(evaluations=args.evaluation, output=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
