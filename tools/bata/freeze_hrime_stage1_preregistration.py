from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata.create_duca_rime_splits import validate_rime_splits
from tools.bata.hrime_stage1_oracle import (
    DEFAULT_CANDIDATE_BUDGETS,
    PREREGISTRATION_SCHEMA,
    STAGE1_ALLOCATION_CONTRACT,
    STAGE1_EVALUATION_CONTRACT,
    STRATEGY_CONTRACTS,
    canonical_sha256,
    validate_preregistration,
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _nonnegative_finite(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _parse_guardrail(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("guardrail must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("guardrail must be a JSON object")
    return payload


def build_preregistration_payload(
    *,
    git_commit: str,
    split_manifest_sha256: str,
    split_assignment_sha256: str,
    anchor_nominal_budgets: Sequence[int],
    oracle_risk_weight: float,
    primary_endpoint: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    multiplicity: Mapping[str, Any],
    guardrails: Sequence[Mapping[str, Any]],
    surrogate_audit: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": PREREGISTRATION_SCHEMA,
        "status": "frozen",
        "task": "offline_temporal_action_detection",
        "git_commit": str(git_commit).lower(),
        "split_manifest_sha256": str(split_manifest_sha256).lower(),
        "split_assignment_sha256": str(split_assignment_sha256).lower(),
        "development_role": "certification_development",
        "uses_official_final": False,
        "official_final_used_for_selection": False,
        "candidate_budgets": list(DEFAULT_CANDIDATE_BUDGETS),
        "anchor_nominal_budgets": [
            int(value) for value in anchor_nominal_budgets
        ],
        "allocation_contract": dict(STAGE1_ALLOCATION_CONTRACT),
        "evaluation_contract": dict(STAGE1_EVALUATION_CONTRACT),
        "strategy_contract_sha256": canonical_sha256(STRATEGY_CONTRACTS),
        "oracle_risk_weight": _nonnegative_finite(
            oracle_risk_weight,
            "oracle risk weight",
        ),
        "primary_endpoint": dict(primary_endpoint),
        "bootstrap": dict(bootstrap),
        "multiplicity": dict(multiplicity),
        "guardrails": [dict(value) for value in guardrails],
        "surrogate_audit": dict(surrogate_audit),
        "threshold_source": (
            "explicit_pre_execution_preregistration_not_observed_results"
        ),
        "claim_scope": (
            "development_oracle_route_admission_not_paper_empirical_result"
        ),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    validate_preregistration(payload)
    return payload


def _assert_clean_exact_checkout(repo_root: str | Path, expected_commit: str) -> None:
    root = Path(repo_root).expanduser().resolve()
    if not (root / ".git").exists():
        raise ValueError("preregistration requires a complete Git worktree")
    observed = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        encoding="utf-8",
    ).strip()
    if observed != str(expected_commit).lower():
        raise ValueError("preregistration checkout differs from the requested commit")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=root,
        text=True,
        encoding="utf-8",
    ).strip()
    if status:
        raise ValueError("preregistration requires a clean exact-commit checkout")


def freeze_preregistration(
    *,
    repo_root: str | Path,
    split_manifest: str | Path,
    split_manifest_sha256: str,
    output: str | Path,
    payload_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    split = validate_rime_splits(
        split_manifest,
        expected_sha256=split_manifest_sha256,
    )
    commit = str(payload_kwargs["git_commit"]).lower()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("an exact Git commit is required")
    _assert_clean_exact_checkout(repo_root, commit)
    kwargs = dict(payload_kwargs)
    kwargs["split_manifest_sha256"] = split["manifest_sha256"]
    kwargs["split_assignment_sha256"] = split["assignment_sha256"]
    payload = build_preregistration_payload(**kwargs)
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite preregistration: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(target),
        "sha256": _sha256_file(target),
        "content_sha256": payload["content_sha256"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze an explicit, pre-execution H-RIME Stage-1 preregistration. "
            "No admission threshold is inferred or defaulted."
        )
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument(
        "--anchor-nominal-budget",
        type=int,
        action="append",
        required=True,
    )
    parser.add_argument("--oracle-risk-weight", type=float, required=True)
    parser.add_argument("--primary-metric", required=True)
    parser.add_argument(
        "--primary-direction",
        choices=("higher", "lower"),
        required=True,
    )
    parser.add_argument("--primary-alpha", type=float, required=True)
    parser.add_argument("--primary-min-mean-delta", type=float, required=True)
    parser.add_argument("--primary-min-lcb-delta", type=float, required=True)
    parser.add_argument(
        "--primary-noninferiority-margin",
        type=float,
        required=True,
    )
    parser.add_argument("--bootstrap-samples", type=int, required=True)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    parser.add_argument(
        "--multiplicity-method",
        choices=("intersection_union_single_primary_with_guardrails",),
        required=True,
    )
    parser.add_argument(
        "--multiplicity-family",
        action="append",
        required=True,
    )
    parser.add_argument(
        "--guardrail-json",
        action="append",
        type=_parse_guardrail,
        default=[],
    )
    parser.add_argument("--surrogate-min-spearman", type=float, required=True)
    parser.add_argument(
        "--surrogate-min-sign-agreement",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--surrogate-max-worst-rank-error",
        type=float,
        required=True,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = freeze_preregistration(
        repo_root=args.repo_root,
        split_manifest=args.split_manifest,
        split_manifest_sha256=args.split_manifest_sha256,
        output=args.output,
        payload_kwargs={
            "git_commit": args.git_commit,
            "split_manifest_sha256": args.split_manifest_sha256,
            "split_assignment_sha256": "0" * 64,
            "anchor_nominal_budgets": args.anchor_nominal_budget,
            "oracle_risk_weight": args.oracle_risk_weight,
            "primary_endpoint": {
                "metric": args.primary_metric,
                "direction": args.primary_direction,
                "alpha": args.primary_alpha,
                "min_mean_delta": args.primary_min_mean_delta,
                "min_lcb_delta": args.primary_min_lcb_delta,
                "noninferiority_margin": (
                    args.primary_noninferiority_margin
                ),
            },
            "bootstrap": {
                "unit": "video",
                "samples": args.bootstrap_samples,
                "seed": args.bootstrap_seed,
            },
            "multiplicity": {
                "method": args.multiplicity_method,
                "family": args.multiplicity_family,
            },
            "guardrails": args.guardrail_json,
            "surrogate_audit": {
                "min_spearman": args.surrogate_min_spearman,
                "min_sign_agreement": args.surrogate_min_sign_agreement,
                "max_worst_rank_error": (
                    args.surrogate_max_worst_rank_error
                ),
                "error_normalization": "fractional_midrank",
            },
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
