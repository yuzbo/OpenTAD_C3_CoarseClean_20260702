#!/usr/bin/env python3
"""Build a live, fail-closed ActionFormer source-diff attestation."""

from __future__ import annotations

import argparse
import json

try:
    from tools.bata import actionformer_source_diff as source_diff
except ModuleNotFoundError:
    import actionformer_source_diff as source_diff


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prove that an exact clean ActionFormer candidate differs from one "
            "base commit only through a predeclared intervention source allowlist."
        )
    )
    parser.add_argument("--repository", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--candidate-commit", default="HEAD")
    parser.add_argument("--base-repository-url", required=True)
    parser.add_argument("--candidate-repository-url", required=True)
    parser.add_argument("--base-remote", default="upstream")
    parser.add_argument("--candidate-remote", default="origin")
    parser.add_argument("--base-remote-ref", default="HEAD")
    parser.add_argument(
        "--candidate-remote-ref",
        required=True,
        help="Full published candidate branch ref, for example refs/heads/codex/foo.",
    )
    parser.add_argument("--base-config-path", default="configs/thumos_i3d.yaml")
    parser.add_argument("--candidate-config-path", required=True)
    parser.add_argument(
        "--intervention",
        required=True,
        choices=sorted(source_diff.SOURCE_INTERVENTION_ALLOWED_PATHS),
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    attestation = source_diff.collect_attestation(
        repository=args.repository,
        base_commit=args.base_commit,
        candidate_commit=args.candidate_commit,
        base_repository_url=args.base_repository_url,
        candidate_repository_url=args.candidate_repository_url,
        base_remote=args.base_remote,
        candidate_remote=args.candidate_remote,
        base_remote_ref=args.base_remote_ref,
        candidate_remote_ref=args.candidate_remote_ref,
        base_config_path=args.base_config_path,
        candidate_config_path=args.candidate_config_path,
        intervention=args.intervention,
    )
    source_diff.atomic_write_json(args.output, attestation)
    print(json.dumps(attestation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
