from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.bata.duca_evidence_io import (
    canonical_sha256,
    verify_content_sha256,
    with_content_sha256,
    write_json_exclusive_atomic,
)


SCHEMA = "duca_acquisition_scientific_protocol_v1"
MARGIN_SOURCE_SCHEMA = "duca_acquisition_ni_margin_source_v1"
_COMMIT = re.compile(r"[0-9a-f]{40}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_margin_source(
    path: str | Path,
    *,
    expected_commit: str,
) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"NI-margin source is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "NI-margin source must be a JSON object")
    if "content_sha256" in payload:
        verify_content_sha256(payload)
    variability = float(payload.get("calibration_variability", float("nan")))
    practical = float(payload.get("practical_relevance_floor", float("nan")))
    _require(
        payload.get("schema") == MARGIN_SOURCE_SCHEMA
        and payload.get("status") == "frozen"
        and payload.get("git_commit") == expected_commit
        and payload.get("fit_scope")
        in {"training", "train_only_calibration", "training_and_calibration"}
        and payload.get("uses_development_results") is False
        and payload.get("uses_official_final") is False
        and payload.get("candidate_performance_observed") is False,
        "NI-margin source identity/scope drift",
    )
    _require(
        math.isfinite(variability)
        and variability >= 0.0
        and math.isfinite(practical)
        and practical >= 0.0,
        "NI-margin source values must be finite and non-negative",
    )
    return resolved, dict(payload)


def build_preregistration_anchor(
    *,
    repo_root: str | Path,
    expected_commit: str,
    expected_branch: str,
    candidate_output_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    candidate_root = Path(candidate_output_root).expanduser().resolve()
    _require(root.is_dir(), f"repository is missing: {root}")
    _require(
        not candidate_root.exists(),
        "candidate output root already exists; protocol would be post-hoc",
    )

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            encoding="utf-8",
        ).strip()

    _require(git("rev-parse", "HEAD") == expected_commit, "Git commit drift")
    _require(
        git("rev-parse", "--abbrev-ref", "HEAD") == expected_branch,
        "Git branch drift",
    )
    _require(
        not git("status", "--porcelain", "--untracked-files=normal"),
        "scientific protocol requires a clean Git tree",
    )
    return {
        "schema": "duca_acquisition_preregistration_anchor_v1",
        "repo_root": str(root),
        "remote": git("remote", "get-url", "origin"),
        "branch": expected_branch,
        "git_commit": expected_commit,
        "git_tree": git("rev-parse", "HEAD^{tree}"),
        "candidate_output_root": str(candidate_root),
        "candidate_output_root_absent": True,
        "candidate_results_observed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def freeze_scientific_protocol(
    *,
    expected_commit: str,
    margin_source: str | Path,
    noninferiority_margin: float,
    primary_endpoint: str,
    multiplicity_procedure: str,
    guardrails: Sequence[str],
    stopping_rules: Sequence[str],
    preregistration_anchor: Mapping[str, Any],
) -> dict[str, Any]:
    commit = str(expected_commit)
    _require(_COMMIT.fullmatch(commit) is not None, "exact Git commit is required")
    margin_path, source = _load_margin_source(
        margin_source,
        expected_commit=commit,
    )
    margin = float(noninferiority_margin)
    expected_margin = max(
        float(source["calibration_variability"]),
        float(source["practical_relevance_floor"]),
    )
    _require(
        math.isfinite(margin) and margin >= 0.0,
        "noninferiority margin must be finite and non-negative",
    )
    _require(
        math.isclose(margin, expected_margin, rel_tol=0.0, abs_tol=1.0e-12),
        "NI margin must equal max(calibration variability, practical relevance floor)",
    )
    endpoint = str(primary_endpoint).strip()
    _require(bool(endpoint), "primary endpoint is required")
    multiplicity = str(multiplicity_procedure)
    _require(
        multiplicity in {"holm", "closed_testing"},
        "multiplicity procedure must be holm or closed_testing",
    )
    guardrail_values = [str(value).strip() for value in guardrails if str(value).strip()]
    stopping_values = [
        str(value).strip() for value in stopping_rules if str(value).strip()
    ]
    _require(bool(guardrail_values), "at least one guardrail is required")
    _require(bool(stopping_values), "at least one stopping rule is required")
    anchor = dict(preregistration_anchor)
    _require(
        anchor.get("schema") == "duca_acquisition_preregistration_anchor_v1"
        and anchor.get("git_commit") == commit
        and anchor.get("candidate_output_root_absent") is True
        and anchor.get("candidate_results_observed") is False
        and bool(str(anchor.get("git_tree", "")).strip())
        and bool(str(anchor.get("branch", "")).strip())
        and bool(str(anchor.get("remote", "")).strip())
        and bool(str(anchor.get("candidate_output_root", "")).strip())
        and bool(str(anchor.get("created_at_utc", "")).strip()),
        "scientific preregistration anchor is incomplete",
    )
    payload = {
        "schema": SCHEMA,
        "status": "frozen",
        "git_commit": commit,
        "frozen_before_candidate_development": True,
        "preregistration_anchor": anchor,
        "primary_endpoint": endpoint,
        "noninferiority_margin": margin,
        "margin_rule": (
            "max(train_calibration_variability, practical_relevance_floor)"
        ),
        "margin_source": {
            "path": str(margin_path),
            "sha256": hashlib.sha256(margin_path.read_bytes()).hexdigest(),
            "content_sha256": source.get("content_sha256"),
            "source_values_sha256": canonical_sha256(
                {
                    "calibration_variability": source[
                        "calibration_variability"
                    ],
                    "practical_relevance_floor": source[
                        "practical_relevance_floor"
                    ],
                }
            ),
        },
        "multiplicity_procedure": multiplicity,
        "cluster_unit": "video",
        "guardrails": guardrail_values,
        "stopping_rules": stopping_values,
        "uses_development_results": False,
        "uses_official_final": False,
        "paper_claim_allowed": False,
        "phase4_submission_enabled": False,
        "official_final_sealed": True,
    }
    return with_content_sha256(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the DUCA acquisition-v2 scientific protocol before "
            "candidate development results are read."
        )
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--candidate-output-root", required=True)
    parser.add_argument("--margin-source", required=True)
    parser.add_argument("--noninferiority-margin", type=float, required=True)
    parser.add_argument("--primary-endpoint", required=True)
    parser.add_argument(
        "--multiplicity-procedure",
        choices=("holm", "closed_testing"),
        required=True,
    )
    parser.add_argument("--guardrail", action="append", required=True)
    parser.add_argument("--stopping-rule", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    anchor = build_preregistration_anchor(
        repo_root=args.repo_root,
        expected_commit=args.expected_commit,
        expected_branch=args.expected_branch,
        candidate_output_root=args.candidate_output_root,
    )
    payload = freeze_scientific_protocol(
        expected_commit=args.expected_commit,
        margin_source=args.margin_source,
        noninferiority_margin=args.noninferiority_margin,
        primary_endpoint=args.primary_endpoint,
        multiplicity_procedure=args.multiplicity_procedure,
        guardrails=args.guardrail,
        stopping_rules=args.stopping_rule,
        preregistration_anchor=anchor,
    )
    write_json_exclusive_atomic(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
