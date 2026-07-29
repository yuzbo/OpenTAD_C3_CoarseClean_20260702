#!/usr/bin/env python3
"""Live Git source-diff attestation for matched ActionFormer experiments.

The attestation is deliberately narrower than a conventional patch manifest.
It proves that a clean candidate snapshot descends from one exact base
snapshot and that every changed source path belongs to one predeclared method
intervention.  It does not replace record-level checks of data, optimization,
evaluation, or post-processing.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path, PurePosixPath


SOURCE_DIFF_ATTESTATION_SCHEMA = "actionformer_source_diff_attestation_v1"
HEX_SHA1_LENGTH = 40

# Exact paths keep the source proof reviewable.  Extending one of these sets is
# a protocol change and therefore requires code review plus regression tests.
SOURCE_INTERVENTION_ALLOWED_PATHS = {
    "native_grid_sparse_head_k384": {
        "configs/thumos_i3d_sparsehead_k384_hash.yaml",
        "configs/thumos_i3d_sparsehead_k384_uniform.yaml",
        "libs/modeling/meta_archs.py",
        "libs/modeling/sparse_heads.py",
        "tests/test_native_grid_sparse_heads.py",
        "tests/test_sparsehead_official_config.py",
    },
    "selection_budget": {
        "configs/thumos_i3d_random_k384.yaml",
        "libs/datasets/deterministic_selection.py",
        "libs/datasets/thumos14.py",
    },
    "head_projection": {
        "configs/thumos_i3d_random_k384_sdpq.yaml",
        "libs/modeling/meta_archs.py",
        "libs/modeling/sdpq.py",
    },
    "coordinate_geometry": {
        "configs/thumos_i3d_random_k384_physical.yaml",
        "libs/modeling/meta_archs.py",
        "libs/modeling/physical_geometry.py",
    },
}
SOURCE_INTERVENTION_ALLOWED_PATHS["sparsehead_method"] = (
    SOURCE_INTERVENTION_ALLOWED_PATHS["selection_budget"]
    | SOURCE_INTERVENTION_ALLOWED_PATHS["head_projection"]
    | SOURCE_INTERVENTION_ALLOWED_PATHS["coordinate_geometry"]
)

# These are the only effective-config leaves that a matched intervention may
# change.  Everything else—including optimizer, schedule, backbone, losses,
# loader, dataset split, and post-processing—is compared after official config
# expansion and must remain byte-for-byte equivalent under canonical JSON.
EFFECTIVE_CONFIG_ALLOWED_PATHS = {
    "native_grid_sparse_head_k384": {
        "model.sparse_head.budget",
        "model.sparse_head.enabled",
        "model.sparse_head.hash_seed",
        "model.sparse_head.policy",
        "model.sparse_head.training_loss_support",
    },
    "selection_budget": {
        "dataset.selection_budget",
        "dataset.selection_coordinate_mode",
        "dataset.selection_policy",
        "dataset.selection_seed",
    },
    "head_projection": {
        "model_intervention.head",
        "model_intervention.projection",
    },
    "coordinate_geometry": {
        "model_intervention.query_geometry",
    },
}
EFFECTIVE_CONFIG_ALLOWED_PATHS["sparsehead_method"] = (
    EFFECTIVE_CONFIG_ALLOWED_PATHS["selection_budget"]
    | EFFECTIVE_CONFIG_ALLOWED_PATHS["head_projection"]
    | EFFECTIVE_CONFIG_ALLOWED_PATHS["coordinate_geometry"]
)

OFFICIAL_CONFIG_LOADER_PATH = "libs/core/config.py"

FORBIDDEN_SOURCE_PREFIXES = (
    ".github/",
    "eval.py",
    "train.py",
    "README.md",
    "libs/core/",
    "libs/utils/",
)


class SourceDiffError(ValueError):
    """Raised when a source-diff proof is incomplete or inconsistent."""


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _run_git(repo, *arguments, text=False, check=True):
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=check,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", b"")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise SourceDiffError(
            f"Git command failed: git {' '.join(arguments)}: {str(stderr).strip()}"
        ) from error
    return completed


def _git_text(repo, *arguments):
    return _run_git(repo, *arguments, text=True).stdout.strip()


def _resolve_commit(repo, revision):
    commit = _git_text(repo, "rev-parse", "--verify", f"{revision}^{{commit}}")
    if len(commit) != HEX_SHA1_LENGTH:
        raise SourceDiffError(f"Git revision is not a full SHA-1 commit: {revision}")
    return commit


def _tree_for_commit(repo, commit):
    tree = _git_text(repo, "rev-parse", "--verify", f"{commit}^{{tree}}")
    if len(tree) != HEX_SHA1_LENGTH:
        raise SourceDiffError(f"Git tree is not a full SHA-1 object: {commit}")
    return tree


def _git_blob(repo, commit, relative):
    return _run_git(repo, "show", f"{commit}:{relative}").stdout


def _git_tree_entry(repo, commit, relative, *, required):
    payload = _run_git(
        repo,
        "ls-tree",
        "-z",
        commit,
        "--",
        relative,
    ).stdout
    if not payload:
        if required:
            raise SourceDiffError(f"Git tree entry is missing: {commit}:{relative}")
        return None
    records = [record for record in payload.split(b"\0") if record]
    if len(records) != 1 or b"\t" not in records[0]:
        raise SourceDiffError(f"Git tree entry is ambiguous: {commit}:{relative}")
    metadata, raw_path = records[0].split(b"\t", 1)
    fields = metadata.decode("ascii", errors="strict").split()
    if len(fields) != 3:
        raise SourceDiffError(f"Git tree metadata is invalid: {commit}:{relative}")
    observed_path = raw_path.decode("utf-8", errors="strict")
    if observed_path != relative:
        raise SourceDiffError(f"Git tree path mismatch: {commit}:{relative}")
    mode, object_type, object_id = fields
    if mode != "100644" or object_type != "blob":
        raise SourceDiffError(
            f"source path is not a regular non-executable blob: "
            f"{relative}: mode={mode}, type={object_type}"
        )
    return {
        "mode": mode,
        "object_type": object_type,
        "object_id": object_id,
    }


def _validate_relative_path(relative):
    if not isinstance(relative, str) or not relative:
        raise SourceDiffError("source-diff path is empty or not a string")
    if "\\" in relative:
        raise SourceDiffError(f"source-diff path uses a backslash: {relative}")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise SourceDiffError(f"source-diff path is unsafe: {relative}")
    if path.parts and path.parts[0] == ".git":
        raise SourceDiffError(f"source-diff path enters .git: {relative}")
    return relative


def _parse_name_status(payload):
    tokens = payload.split(b"\0")
    if tokens and tokens[-1] == b"":
        tokens.pop()
    entries = []
    index = 0
    while index < len(tokens):
        try:
            status = tokens[index].decode("ascii", errors="strict")
        except UnicodeDecodeError as error:
            raise SourceDiffError("Git name-status contains a non-ASCII status") from error
        index += 1
        if status.startswith(("R", "C")):
            if index + 1 >= len(tokens):
                raise SourceDiffError("truncated rename/copy entry in Git name-status")
            old_path = tokens[index].decode("utf-8", errors="strict")
            new_path = tokens[index + 1].decode("utf-8", errors="strict")
            index += 2
            entries.append(
                {
                    "status": status,
                    "old_path": _validate_relative_path(old_path),
                    "path": _validate_relative_path(new_path),
                }
            )
            continue
        if index >= len(tokens):
            raise SourceDiffError("truncated path entry in Git name-status")
        relative = tokens[index].decode("utf-8", errors="strict")
        index += 1
        entries.append(
            {
                "status": status,
                "path": _validate_relative_path(relative),
            }
        )
    return entries


def _source_diff(repo, base_commit, candidate_commit):
    binary_diff = _run_git(
        repo,
        "diff",
        "--binary",
        "--full-index",
        "--no-ext-diff",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        base_commit,
        candidate_commit,
        "--",
    ).stdout
    name_status = _run_git(
        repo,
        "diff",
        "--name-status",
        "-z",
        "-M",
        "--no-ext-diff",
        base_commit,
        candidate_commit,
        "--",
    ).stdout
    return binary_diff, name_status, _parse_name_status(name_status)


def _remote_url(repo, remote):
    if not isinstance(remote, str) or not remote:
        raise SourceDiffError("Git remote name is empty")
    return _git_text(repo, "remote", "get-url", remote)


def _remote_ref_commit(repo, remote, remote_ref):
    if not isinstance(remote_ref, str) or not remote_ref:
        raise SourceDiffError("Git remote ref is empty")
    if remote_ref != "HEAD" and not remote_ref.startswith("refs/heads/"):
        raise SourceDiffError(
            f"Git remote ref must be HEAD or a full branch ref: {remote_ref}"
        )
    completed = _run_git(
        repo,
        "ls-remote",
        "--exit-code",
        remote,
        remote_ref,
        text=True,
    )
    rows = [line.split("\t") for line in completed.stdout.splitlines() if line]
    if len(rows) != 1 or len(rows[0]) != 2 or rows[0][1] != remote_ref:
        raise SourceDiffError(f"Git remote ref is ambiguous: {remote}:{remote_ref}")
    commit = rows[0][0]
    if len(commit) != HEX_SHA1_LENGTH:
        raise SourceDiffError(f"Git remote ref is not a SHA-1 commit: {remote_ref}")
    return commit


def _assert_clean_candidate(repo, candidate_commit):
    inside = _git_text(repo, "rev-parse", "--is-inside-work-tree")
    if inside != "true":
        raise SourceDiffError(f"candidate source is not a Git worktree: {repo}")
    head = _resolve_commit(repo, "HEAD")
    if head != candidate_commit:
        raise SourceDiffError(
            f"candidate commit is not checked out: HEAD={head}, candidate={candidate_commit}"
        )
    status = _git_text(
        repo,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise SourceDiffError("candidate source worktree is not clean")


def _assert_base_ancestor(repo, base_commit, candidate_commit):
    completed = _run_git(
        repo,
        "merge-base",
        "--is-ancestor",
        base_commit,
        candidate_commit,
        check=False,
    )
    if completed.returncode != 0:
        raise SourceDiffError("base commit is not an ancestor of the candidate commit")


def _validate_changed_entries(repo, base_commit, candidate_commit, entries, intervention):
    allowed_paths = SOURCE_INTERVENTION_ALLOWED_PATHS.get(intervention)
    if allowed_paths is None:
        raise SourceDiffError(f"unknown source intervention: {intervention}")
    if not entries:
        raise SourceDiffError("source-diff is empty")

    observed_paths = []
    detailed_entries = []
    for entry in entries:
        status = entry["status"]
        relative = entry["path"]
        if status not in {"A", "M"}:
            raise SourceDiffError(
                f"source-diff status is forbidden for paper-matched evidence: "
                f"{status}: {relative}"
            )
        if relative not in allowed_paths:
            raise SourceDiffError(
                f"changed source path is outside the {intervention} allowlist: {relative}"
            )
        if any(
            relative == prefix.rstrip("/") or relative.startswith(prefix)
            for prefix in FORBIDDEN_SOURCE_PREFIXES
        ):
            raise SourceDiffError(f"changed source path is protected: {relative}")
        observed_paths.append(relative)
        candidate_tree_entry = _git_tree_entry(
            repo,
            candidate_commit,
            relative,
            required=True,
        )
        candidate_blob = _git_blob(repo, candidate_commit, relative)
        base_blob_sha256 = None
        base_tree_entry = None
        if status == "M":
            base_tree_entry = _git_tree_entry(
                repo,
                base_commit,
                relative,
                required=True,
            )
            base_blob_sha256 = _sha256_bytes(_git_blob(repo, base_commit, relative))
        detailed_entries.append(
            {
                "status": status,
                "path": relative,
                "base_blob_sha256": base_blob_sha256,
                "candidate_blob_sha256": _sha256_bytes(candidate_blob),
                "base_tree_entry": base_tree_entry,
                "candidate_tree_entry": candidate_tree_entry,
            }
        )
    if observed_paths != sorted(set(observed_paths)):
        raise SourceDiffError("changed source paths are not sorted and unique")
    return detailed_entries


def _load_effective_config(repository, config_blob):
    """Expand one config through the exact protected ActionFormer loader."""

    loader_path = repository / OFFICIAL_CONFIG_LOADER_PATH
    namespace = {
        "__file__": str(loader_path),
        "__name__": "_actionformer_attested_config",
    }
    try:
        source = loader_path.read_bytes()
        exec(compile(source, str(loader_path), "exec"), namespace)
    except Exception as error:
        raise SourceDiffError(
            f"cannot execute the protected ActionFormer config loader: {error}"
        ) from error
    defaults = namespace.get("DEFAULTS")
    load_config = namespace.get("load_config")
    if not isinstance(defaults, dict):
        raise SourceDiffError("protected ActionFormer config loader has no DEFAULTS")
    if not callable(load_config):
        raise SourceDiffError("protected ActionFormer config loader has no load_config")

    with tempfile.TemporaryDirectory(prefix="actionformer-config-attestation-") as temp:
        config_path = Path(temp) / "config.yaml"
        config_path.write_bytes(config_blob)
        try:
            config = load_config(
                str(config_path),
                defaults=copy.deepcopy(defaults),
            )
        except Exception as error:
            raise SourceDiffError(
                f"cannot expand ActionFormer effective config: {error}"
            ) from error
    if not isinstance(config, dict) or not config:
        raise SourceDiffError("expanded ActionFormer effective config is empty")
    try:
        _canonical_sha256(config)
    except (TypeError, ValueError) as error:
        raise SourceDiffError(
            f"expanded ActionFormer effective config is not canonical JSON: {error}"
        ) from error
    return config


def _flatten_config(value, prefix=""):
    if isinstance(value, dict):
        flattened = {}
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise SourceDiffError("effective config contains an invalid mapping key")
            dotted = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten_config(value[key], dotted))
        return flattened
    if not prefix:
        raise SourceDiffError("effective config root is not a mapping")
    return {prefix: value}


def _effective_config_attestation(
    repository,
    base_commit,
    candidate_commit,
    base_config,
    candidate_config,
    intervention,
):
    allowed_paths = EFFECTIVE_CONFIG_ALLOWED_PATHS.get(intervention)
    if allowed_paths is None:
        raise SourceDiffError(
            f"unknown effective-config intervention: {intervention}"
        )

    base_loader_entry = _git_tree_entry(
        repository,
        base_commit,
        OFFICIAL_CONFIG_LOADER_PATH,
        required=True,
    )
    candidate_loader_entry = _git_tree_entry(
        repository,
        candidate_commit,
        OFFICIAL_CONFIG_LOADER_PATH,
        required=True,
    )
    base_loader = _git_blob(
        repository,
        base_commit,
        OFFICIAL_CONFIG_LOADER_PATH,
    )
    candidate_loader = _git_blob(
        repository,
        candidate_commit,
        OFFICIAL_CONFIG_LOADER_PATH,
    )
    if base_loader != candidate_loader:
        raise SourceDiffError("protected ActionFormer config loader changed")

    base_effective = _load_effective_config(repository, base_config)
    candidate_effective = _load_effective_config(repository, candidate_config)
    base_flat = _flatten_config(base_effective)
    candidate_flat = _flatten_config(candidate_effective)
    changed_paths = sorted(
        path
        for path in set(base_flat) | set(candidate_flat)
        if base_flat.get(path, object()) != candidate_flat.get(path, object())
    )
    if not changed_paths:
        raise SourceDiffError("candidate effective config has no method-side change")
    outside = sorted(set(changed_paths) - allowed_paths)
    if outside:
        raise SourceDiffError(
            "effective config changes protected paths: " + ", ".join(outside)
        )

    base_protected = {
        path: value for path, value in base_flat.items() if path not in allowed_paths
    }
    candidate_protected = {
        path: value
        for path, value in candidate_flat.items()
        if path not in allowed_paths
    }
    base_protected_sha = _canonical_sha256(base_protected)
    candidate_protected_sha = _canonical_sha256(candidate_protected)
    if base_protected_sha != candidate_protected_sha:
        raise SourceDiffError("protected effective configs are not identical")

    loader_sha = _sha256_bytes(base_loader)
    return {
        "loader_path": OFFICIAL_CONFIG_LOADER_PATH,
        "base_loader_tree_entry": base_loader_entry,
        "candidate_loader_tree_entry": candidate_loader_entry,
        "base_loader_blob_sha256": loader_sha,
        "candidate_loader_blob_sha256": _sha256_bytes(candidate_loader),
        "base_sha256": _canonical_sha256(base_effective),
        "candidate_sha256": _canonical_sha256(candidate_effective),
        "protected_sha256": base_protected_sha,
        "changed_paths": changed_paths,
        "allowed_changed_paths": sorted(allowed_paths),
    }


def collect_attestation(
    *,
    repository,
    base_commit,
    candidate_commit,
    base_repository_url,
    candidate_repository_url,
    base_remote,
    candidate_remote,
    base_remote_ref,
    candidate_remote_ref,
    base_config_path,
    candidate_config_path,
    intervention,
):
    """Collect and live-validate one exact source-diff attestation."""

    repository = Path(repository).resolve()
    if not repository.is_dir():
        raise SourceDiffError(f"candidate repository is missing: {repository}")
    base_commit = _resolve_commit(repository, base_commit)
    candidate_commit = _resolve_commit(repository, candidate_commit)
    _assert_clean_candidate(repository, candidate_commit)
    _assert_base_ancestor(repository, base_commit, candidate_commit)

    observed_base_url = _remote_url(repository, base_remote)
    observed_candidate_url = _remote_url(repository, candidate_remote)
    if observed_base_url != base_repository_url:
        raise SourceDiffError(
            f"base repository URL mismatch: {observed_base_url} != {base_repository_url}"
        )
    if observed_candidate_url != candidate_repository_url:
        raise SourceDiffError(
            "candidate repository URL mismatch: "
            f"{observed_candidate_url} != {candidate_repository_url}"
        )
    observed_base_remote_commit = _remote_ref_commit(
        repository,
        base_remote,
        base_remote_ref,
    )
    observed_candidate_remote_commit = _remote_ref_commit(
        repository,
        candidate_remote,
        candidate_remote_ref,
    )
    if observed_base_remote_commit != base_commit:
        raise SourceDiffError(
            "base remote ref does not resolve to the declared base commit"
        )
    if observed_candidate_remote_commit != candidate_commit:
        raise SourceDiffError(
            "candidate remote ref does not resolve to the declared candidate commit"
        )

    base_config_path = _validate_relative_path(base_config_path)
    candidate_config_path = _validate_relative_path(candidate_config_path)
    base_config_tree_entry = _git_tree_entry(
        repository,
        base_commit,
        base_config_path,
        required=True,
    )
    candidate_config_tree_entry = _git_tree_entry(
        repository,
        candidate_commit,
        candidate_config_path,
        required=True,
    )
    base_config = _git_blob(repository, base_commit, base_config_path)
    candidate_config = _git_blob(repository, candidate_commit, candidate_config_path)
    binary_diff, name_status, entries = _source_diff(
        repository,
        base_commit,
        candidate_commit,
    )
    detailed_entries = _validate_changed_entries(
        repository,
        base_commit,
        candidate_commit,
        entries,
        intervention,
    )
    changed_paths = [entry["path"] for entry in detailed_entries]
    allowed_paths = SOURCE_INTERVENTION_ALLOWED_PATHS[intervention]
    if candidate_config_path not in allowed_paths:
        raise SourceDiffError(
            "candidate config path is outside the intervention source allowlist"
        )
    if candidate_config_path not in changed_paths:
        raise SourceDiffError(
            "candidate config path is not present in the sealed source diff"
        )
    effective_config = _effective_config_attestation(
        repository,
        base_commit,
        candidate_commit,
        base_config,
        candidate_config,
        intervention,
    )

    return {
        "schema_version": SOURCE_DIFF_ATTESTATION_SCHEMA,
        "validation_pass": True,
        "issues": [],
        "intervention": intervention,
        "repository_root": str(repository),
        "base": {
            "repository_url": base_repository_url,
            "remote": base_remote,
            "remote_ref": base_remote_ref,
            "remote_ref_commit": observed_base_remote_commit,
            "commit": base_commit,
            "tree": _tree_for_commit(repository, base_commit),
            "config_path": base_config_path,
            "config_blob_sha256": _sha256_bytes(base_config),
            "config_tree_entry": base_config_tree_entry,
            "effective_config_sha256": effective_config["base_sha256"],
        },
        "candidate": {
            "repository_url": candidate_repository_url,
            "remote": candidate_remote,
            "remote_ref": candidate_remote_ref,
            "remote_ref_commit": observed_candidate_remote_commit,
            "commit": candidate_commit,
            "tree": _tree_for_commit(repository, candidate_commit),
            "config_path": candidate_config_path,
            "config_blob_sha256": _sha256_bytes(candidate_config),
            "config_tree_entry": candidate_config_tree_entry,
            "effective_config_sha256": effective_config["candidate_sha256"],
            "clean": True,
            "head_matches_candidate_commit": True,
            "remote_ref_matches_candidate_commit": True,
        },
        "diff": {
            "binary_sha256": _sha256_bytes(binary_diff),
            "name_status_sha256": _sha256_bytes(name_status),
            "changed_paths": changed_paths,
            "entries": detailed_entries,
        },
        "effective_config": effective_config,
        "policy": {
            "allowed_paths": sorted(
                SOURCE_INTERVENTION_ALLOWED_PATHS[intervention]
            ),
            "forbidden_prefixes": list(FORBIDDEN_SOURCE_PREFIXES),
            "allowed_statuses": ["A", "M"],
            "rename_copy_delete_allowed": False,
            "allowed_effective_config_paths": sorted(
                EFFECTIVE_CONFIG_ALLOWED_PATHS[intervention]
            ),
        },
    }


def validate_attestation_live(attestation):
    """Recompute an attestation from its sealed inputs and require exact equality."""

    if not isinstance(attestation, dict):
        raise SourceDiffError("source-diff attestation is not a JSON object")
    if attestation.get("schema_version") != SOURCE_DIFF_ATTESTATION_SCHEMA:
        raise SourceDiffError("unsupported source-diff attestation schema")
    if attestation.get("validation_pass") is not True or attestation.get("issues") != []:
        raise SourceDiffError("source-diff attestation did not pass cleanly")
    base = attestation.get("base")
    candidate = attestation.get("candidate")
    if not isinstance(base, dict) or not isinstance(candidate, dict):
        raise SourceDiffError("source-diff base/candidate payload is missing")
    rebuilt = collect_attestation(
        repository=attestation.get("repository_root", ""),
        base_commit=base.get("commit", ""),
        candidate_commit=candidate.get("commit", ""),
        base_repository_url=base.get("repository_url", ""),
        candidate_repository_url=candidate.get("repository_url", ""),
        base_remote=base.get("remote", ""),
        candidate_remote=candidate.get("remote", ""),
        base_remote_ref=base.get("remote_ref", ""),
        candidate_remote_ref=candidate.get("remote_ref", ""),
        base_config_path=base.get("config_path", ""),
        candidate_config_path=candidate.get("config_path", ""),
        intervention=attestation.get("intervention", ""),
    )
    if rebuilt != attestation:
        raise SourceDiffError("source-diff attestation differs from live Git recomputation")
    return rebuilt


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
