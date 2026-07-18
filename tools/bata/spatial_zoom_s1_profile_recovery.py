from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import json
import math
import os
import site
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence

from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.spatial_zoom_s1_contract import (  # noqa: E402
    atomic_publish_json,
    build_s1_profile_order,
    canonical_sha256,
    sha256_file,
)
from tools.bata.spatial_zoom_s1_training import (  # noqa: E402
    current_git_commit,
    validate_bound_s1_training_config,
)
from tools.bata.spatial_zoom_s1_power import (  # noqa: E402
    S1_POWER_BUFFERED_TRACE_PUBLICATION_MODE,
    S1_POWER_SIDECAR_CADENCE_FAILURE_PREFIX,
    validate_nvml_sidecar_cadence_failure,
)

S1_PROFILE_RECOVERY_SCHEMA = "spatial_zoom_s1_profile_recovery_v1"
S1_CHAINED_PROFILE_RECOVERY_SCHEMA = "spatial_zoom_s1_profile_recovery_v2"
S1_SIDECAR_PROFILE_RECOVERY_SCHEMA = "spatial_zoom_s1_profile_recovery_v3"
S1_BUFFERED_SIDECAR_PROFILE_RECOVERY_SCHEMA = "spatial_zoom_s1_profile_recovery_v4"
S1_STEP_RUNTIME_PROFILE_RECOVERY_SCHEMA = "spatial_zoom_s1_profile_recovery_v5"
S1_SUPERSEDED_PROFILE_MARKER_SCHEMA = "spatial_zoom_s1_profile_attempt_v4"
S1_PROFILE_RECOVERY_REASON = "duplicate_physical_window_identity"
S1_PROFILE_FAILURE_SIGNATURE = "formal S1 profile window identities must be unique"
S1_CHAINED_PROFILE_MARKER_SCHEMA = "spatial_zoom_s1_profile_attempt_v5"
S1_SIDECAR_PROFILE_MARKER_SCHEMA = "spatial_zoom_s1_profile_attempt_v6"
S1_BUFFERED_SIDECAR_PROFILE_MARKER_SCHEMA = "spatial_zoom_s1_profile_attempt_v7"
S1_CHAINED_RECOVERY_REASON = "duplicate_window_and_power_sampler_cadence"
S1_SIDECAR_RECOVERY_REASON = "out_of_process_power_sidecar"
S1_BUFFERED_SIDECAR_RECOVERY_REASON = "buffered_out_of_process_power_sidecar"
S1_STEP_RUNTIME_RECOVERY_REASON = "step_scoped_formal_test_runtime"
S1_POWER_FAILURE_SIGNATURE = (
    "formal S1 power trace is too sparse for auditable energy integration"
)
S1_POWER_DIAGNOSTIC_SCHEMA = "spatial_zoom_s1_power_sampler_diagnostic_v1"
S1_SIDECAR_POWER_BACKEND = "nvml-sidecar-process-v1"
S1_BUFFERED_TRACE_PUBLICATION_MODE = S1_POWER_BUFFERED_TRACE_PUBLICATION_MODE
S1_BUFFERED_SIDECAR_FAILURE_SIGNATURE = S1_POWER_SIDECAR_CADENCE_FAILURE_PREFIX
S1_STEP_RUNTIME_FAILURE_SIGNATURE = (
    "formal S1 execution requires one auditable SLURM_JOB_GPUS identity"
)
S1_STEP_SCOPED_TEST_RUNTIME_MODE = "step_scoped_profile_runtime_v1"
S1_STEP_RUNTIME_PARENT_JOB_ID = "1170468"
S1_STEP_RUNTIME_FAILED_ORDINAL = 1
S1_STEP_RUNTIME_FAILED_RESOLUTION = 224
S1_STEP_RUNTIME_FAILED_SEED = 3409
S1_STEP_RUNTIME_TRAINING_COMMIT = "18139b930bef6ee234f6220a6adc898eb9c23c0c"
S1_MATRIX_SUBMISSION_SCHEMA = "spatial_zoom_s1_sidecar_matrix_submission_v2"
S1_FORMAL_NUMPY_VERSION = "1.23.5"
S1_FORMAL_NUMPY_PATH = (
    "/data/run01/sczc063/yuzibo/conda_envs/opentad/lib/python3.10/"
    "site-packages/numpy/__init__.py"
)

_ALLOWED_EXACT_PATHS = {
    "docs/methods/spatial_zoom_s1_contract.md",
    "docs/superpowers/specs/2026-07-17-spatial-zoom-s1-power-sidecar-design.md",
    "scripts/run_spatial_zoom_s1_power_sampler_diag_slurm.sh",
    "scripts/run_spatial_zoom_s1_power_sidecar_gate_slurm.sh",
    "scripts/run_spatial_zoom_s1_profile_recovery_matrix_slurm.sh",
    "scripts/run_spatial_zoom_s1_test_profile_slurm.sh",
    "tests/test_spatial_zoom_s1_infrastructure.py",
    "tests/test_spatial_zoom_s1_matrix.py",
    "tools/bata/analyze_spatial_zoom_s1_results.py",
    "tools/bata/build_spatial_zoom_s1_run_descriptor.py",
    "tools/bata/preflight_spatial_zoom_s1_profile.py",
    "tools/bata/profile_spatial_zoom_s1.py",
    "tools/bata/run_spatial_zoom_s1_precheck.py",
    "tools/bata/spatial_zoom_s1_cost.py",
    "tools/bata/spatial_zoom_s1_matrix.py",
    "tools/bata/spatial_zoom_s1_profile_recovery.py",
    "tools/bata/spatial_zoom_s1_power.py",
    "tools/bata/spatial_zoom_s1_sidecar_gate.py",
    "tools/bata/spatial_zoom_s1_training.py",
}
_ALLOWED_EXACT_PATHS_STEP_RUNTIME = _ALLOWED_EXACT_PATHS | {
    "docs/superpowers/specs/2026-07-18-spatial-zoom-s1-step-scoped-test-runtime-recovery.md",
    "tests/test_spatial_zoom_s1_step_runtime_recovery.py",
    "tools/bata/spatial_zoom_s1_evidence.py",
    "tools/test.py",
}
_PARENT_TO_STEP_RUNTIME_PATHS = {
    "docs/methods/spatial_zoom_s1_contract.md",
    "docs/superpowers/specs/2026-07-18-spatial-zoom-s1-step-scoped-test-runtime-recovery.md",
    "scripts/run_spatial_zoom_s1_power_sidecar_gate_slurm.sh",
    "scripts/run_spatial_zoom_s1_profile_recovery_matrix_slurm.sh",
    "scripts/run_spatial_zoom_s1_test_profile_slurm.sh",
    "tests/test_spatial_zoom_s1_infrastructure.py",
    "tests/test_spatial_zoom_s1_step_runtime_recovery.py",
    "tools/bata/build_spatial_zoom_s1_run_descriptor.py",
    "tools/bata/spatial_zoom_s1_evidence.py",
    "tools/bata/spatial_zoom_s1_profile_recovery.py",
    "tools/test.py",
}
_ALLOWED_PREFIXES = ("research-wiki/",)
_REQUIRED_REPAIR_PATHS_V1 = {
    "scripts/run_spatial_zoom_s1_profile_recovery_matrix_slurm.sh",
    "scripts/run_spatial_zoom_s1_test_profile_slurm.sh",
    "tests/test_spatial_zoom_s1_infrastructure.py",
    "tools/bata/analyze_spatial_zoom_s1_results.py",
    "tools/bata/build_spatial_zoom_s1_run_descriptor.py",
    "tools/bata/preflight_spatial_zoom_s1_profile.py",
    "tools/bata/profile_spatial_zoom_s1.py",
    "tools/bata/run_spatial_zoom_s1_precheck.py",
    "tools/bata/spatial_zoom_s1_cost.py",
    "tools/bata/spatial_zoom_s1_profile_recovery.py",
    "tools/bata/spatial_zoom_s1_training.py",
}
_REQUIRED_REPAIR_PATHS_CHAINED = _REQUIRED_REPAIR_PATHS_V1 | {
    "tools/bata/spatial_zoom_s1_power.py",
}
_REQUIRED_REPAIR_PATHS_SIDECAR = _REQUIRED_REPAIR_PATHS_CHAINED | {
    "docs/superpowers/specs/2026-07-17-spatial-zoom-s1-power-sidecar-design.md",
    "scripts/run_spatial_zoom_s1_power_sidecar_gate_slurm.sh",
    "tests/test_spatial_zoom_s1_matrix.py",
    "tools/bata/spatial_zoom_s1_matrix.py",
    "tools/bata/spatial_zoom_s1_sidecar_gate.py",
}
_REQUIRED_REPAIR_PATHS_BUFFERED_SIDECAR = _REQUIRED_REPAIR_PATHS_SIDECAR
_REQUIRED_STEP_RUNTIME_CODE_PATHS = {
    "scripts/run_spatial_zoom_s1_power_sidecar_gate_slurm.sh",
    "scripts/run_spatial_zoom_s1_profile_recovery_matrix_slurm.sh",
    "scripts/run_spatial_zoom_s1_test_profile_slurm.sh",
    "tools/bata/build_spatial_zoom_s1_run_descriptor.py",
    "tools/bata/spatial_zoom_s1_evidence.py",
    "tools/test.py",
}
_REQUIRED_REPAIR_PATHS_STEP_RUNTIME = (
    _REQUIRED_REPAIR_PATHS_BUFFERED_SIDECAR
    | _REQUIRED_STEP_RUNTIME_CODE_PATHS
    | {
        "docs/superpowers/specs/2026-07-18-spatial-zoom-s1-step-scoped-test-runtime-recovery.md",
    }
)
_REQUIRED_PARENT_TO_STEP_RUNTIME_PATHS = _PARENT_TO_STEP_RUNTIME_PATHS
_MODEL_SURFACE_PREFIXES = ("opentad/", "configs/")
_TOOLS_TEST_PATH = "tools/test.py"
_TOOLS_TEST_ALLOWED_AST_COUNTS = {
    "cli_argument": 1,
    "contract_hash_import": 1,
    "recovery_import": 1,
    "recovery_state_assignments": 2,
    "recovery_checkout_block": 1,
    "test_binding_provenance_block": 1,
    "non_s1_recovery_guard": 1,
    "marker_provenance_block": 1,
}
_TOOLS_TEST_SEMANTIC_SCOPE = (
    "cli_argument_s1_recovery_checkout_test_binding_marker_provenance_only"
)
_MATRIX_SUBMISSION_FIELDS = {
    "schema_version",
    "status",
    "submitted_utc",
    "job_id",
    "job_name",
    "profile_campaign_id",
    "profile_campaign_root",
    "profile_code_commit",
    "profile_source_root",
    "training_code_commit",
    "recovery_certificate_path",
    "recovery_certificate_file_sha256",
    "recovery_certificate_sha256",
    "sidecar_gate_path",
    "sidecar_gate_file_sha256",
    "sidecar_gate_sha256",
    "frozen_order",
    "single_serial_allocation",
    "physical_gpu_index_overridden",
    "outer_resources",
    "inner_resources",
    "receipt_sha256",
}


def _run_git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Git command failed")
    return completed.stdout


def _formal_python_environment_evidence() -> dict[str, Any]:
    import numpy

    evidence = {
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        "site_enable_user_site": bool(site.ENABLE_USER_SITE),
        "numpy_version": str(numpy.__version__),
        "numpy_path": str(Path(numpy.__file__).resolve()),
    }
    expected = {
        "python_no_user_site": "1",
        "site_enable_user_site": False,
        "numpy_version": S1_FORMAL_NUMPY_VERSION,
        "numpy_path": S1_FORMAL_NUMPY_PATH,
    }
    if evidence != expected:
        raise ValueError(
            "S1 formal recovery requires the frozen Conda Python environment: "
            f"expected={expected}, actual={evidence}"
        )
    return evidence


def _run_git_bytes(*args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.decode("utf-8", errors="replace").strip()
            or "Git command failed"
        )
    return completed.stdout


def _git_file_sha256(commit: str, path: str) -> str:
    payload = _run_git_bytes("show", f"{str(commit).lower()}:{Path(path).as_posix()}")
    return hashlib.sha256(payload).hexdigest()


def _git_file_text(commit: str, path: str) -> str:
    payload = _run_git_bytes("show", f"{str(commit).lower()}:{Path(path).as_posix()}")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"S1 recovery requires UTF-8 source: {path}") from exc


def _tools_test_zero_context_patch(
    training_commit: str,
    runtime_commit: str,
) -> bytes:
    return _run_git_bytes(
        "diff",
        "--no-ext-diff",
        "--no-color",
        "--unified=0",
        f"{str(training_commit).lower()}..{str(runtime_commit).lower()}",
        "--",
        _TOOLS_TEST_PATH,
    )


def _call_name(node: ast.AST) -> str:
    if not isinstance(node, ast.Call):
        return ""
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def _contains_call(node: ast.AST, name: str) -> bool:
    return any(_call_name(child) == name for child in ast.walk(node))


def _is_args_recovery_flag(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "s1_profile_recovery_certificate"
        and isinstance(node.value, ast.Name)
        and node.value.id == "args"
    )


def _is_recovery_not_none(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "recovery"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.IsNot)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value is None
    )


def _ast_shape(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _single_statement_shape(source: str) -> str:
    tree = ast.parse(textwrap.dedent(source))
    if len(tree.body) != 1:
        raise RuntimeError("invalid frozen tools/test.py AST shape")
    return _ast_shape(tree.body[0])


_TOOLS_TEST_CLI_ARGUMENT_SHAPE = _single_statement_shape(
    """
    parser.add_argument(
        "--s1-profile-recovery-certificate",
        type=str,
        default=None,
        help=(
            "audited profile-runtime certificate required when formal S1 test "
            "runs from an infrastructure-only recovery commit"
        ),
    )
    """
)
_TOOLS_TEST_CHECKOUT_BLOCK_SHAPE = _single_statement_shape(
    """
    if args.s1_profile_recovery_certificate:
        recovery_path = os.path.abspath(args.s1_profile_recovery_certificate)
        recovery = load_profile_recovery_certificate(
            recovery_path,
            binding=s1_binding,
            verify_checkout=True,
        )
        if (
            recovery.get("formal_test_runtime_mode")
            != S1_STEP_SCOPED_TEST_RUNTIME_MODE
        ):
            raise RuntimeError(
                "formal S1 recovery test requires the step-scoped runtime contract"
            )
    else:
        require_clean_git_checkout(expected_commit=s1_binding["code_commit"])
    """
)
_TOOLS_TEST_BINDING_BLOCK_SHAPE = _single_statement_shape(
    """
    if recovery is not None:
        cfg.spatial_zoom_s1_test_binding.update(
            formal_test_runtime_mode=S1_STEP_SCOPED_TEST_RUNTIME_MODE,
            training_code_commit=s1_binding["code_commit"],
            test_runtime_code_commit=recovery["profile_code_commit"],
            profile_recovery_certificate_path=recovery_path,
            profile_recovery_certificate_file_sha256=sha256_file(recovery_path),
            profile_recovery_certificate_sha256=recovery["certificate_sha256"],
            profile_recovery_campaign_id=recovery["campaign_id"],
        )
    """
)
_TOOLS_TEST_NON_S1_GUARD_SHAPE = _single_statement_shape(
    """
    if args.s1_profile_recovery_certificate:
        raise ValueError(
            "--s1-profile-recovery-certificate is valid only for formal S1 configs"
        )
    """
)
_TOOLS_TEST_MARKER_BLOCK_SHAPE = _single_statement_shape(
    """
    if recovery is not None:
        marker.update(
            formal_test_runtime_mode=S1_STEP_SCOPED_TEST_RUNTIME_MODE,
            test_runtime_code_commit=recovery["profile_code_commit"],
            profile_recovery_certificate_path=recovery_path,
            profile_recovery_certificate_file_sha256=sha256_file(recovery_path),
            profile_recovery_certificate_sha256=recovery["certificate_sha256"],
            profile_recovery_campaign_id=recovery["campaign_id"],
        )
    """
)


class _ToolsTestRuntimeNormalizer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.counts = {key: 0 for key in _TOOLS_TEST_ALLOWED_AST_COUNTS}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node = self.generic_visit(node)
        if node.name != "parse_args":
            return node
        kept: list[ast.stmt] = []
        for statement in node.body:
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and _call_name(statement.value) == "add_argument"
                and statement.value.args
                and isinstance(statement.value.args[0], ast.Constant)
                and statement.value.args[0].value == "--s1-profile-recovery-certificate"
            ):
                if _ast_shape(statement) != _TOOLS_TEST_CLI_ARGUMENT_SHAPE:
                    raise ValueError(
                        "tools/test.py recovery CLI argument shape mismatch"
                    )
                self.counts["cli_argument"] += 1
                continue
            kept.append(statement)
        node.body = kept
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST | None:
        if node.module == "tools.bata.spatial_zoom_s1_profile_recovery":
            names = {alias.name for alias in node.names}
            if names != {
                "S1_STEP_SCOPED_TEST_RUNTIME_MODE",
                "load_profile_recovery_certificate",
            }:
                raise ValueError(
                    "tools/test.py recovery import exceeds the allowed scope"
                )
            self.counts["recovery_import"] += 1
            return None
        if node.module == "tools.bata.spatial_zoom_s1_contract":
            kept = [alias for alias in node.names if alias.name != "sha256_file"]
            if len(kept) != len(node.names):
                self.counts["contract_hash_import"] += 1
                node.names = kept
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"recovery", "recovery_path"}
            and isinstance(node.value, ast.Constant)
            and node.value.value is None
        ):
            self.counts["recovery_state_assignments"] += 1
            return None
        return self.generic_visit(node)

    def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt] | None:
        if _is_args_recovery_flag(node.test):
            if node.orelse and any(
                _contains_call(statement, "require_clean_git_checkout")
                for statement in node.orelse
            ):
                if _ast_shape(node) != _TOOLS_TEST_CHECKOUT_BLOCK_SHAPE:
                    raise ValueError(
                        "tools/test.py recovery checkout block exceeds scope"
                    )
                self.counts["recovery_checkout_block"] += 1
                return [self.visit(statement) or statement for statement in node.orelse]
            if (
                not node.orelse
                and len(node.body) == 1
                and isinstance(node.body[0], ast.Raise)
            ):
                if _ast_shape(node) != _TOOLS_TEST_NON_S1_GUARD_SHAPE:
                    raise ValueError(
                        "tools/test.py non-S1 recovery guard exceeds scope"
                    )
                self.counts["non_s1_recovery_guard"] += 1
                return None
        if _is_recovery_not_none(node.test):
            if _contains_call(node, "update"):
                attributes = {
                    child.attr
                    for child in ast.walk(node)
                    if isinstance(child, ast.Attribute)
                }
                if "spatial_zoom_s1_test_binding" in attributes:
                    if _ast_shape(node) != _TOOLS_TEST_BINDING_BLOCK_SHAPE:
                        raise ValueError(
                            "tools/test.py test-binding provenance exceeds scope"
                        )
                    self.counts["test_binding_provenance_block"] += 1
                    return None
                names = {
                    child.id for child in ast.walk(node) if isinstance(child, ast.Name)
                }
                if "marker" in names:
                    if _ast_shape(node) != _TOOLS_TEST_MARKER_BLOCK_SHAPE:
                        raise ValueError(
                            "tools/test.py marker provenance exceeds scope"
                        )
                    self.counts["marker_provenance_block"] += 1
                    return None
        return self.generic_visit(node)


def _normalized_tools_test_ast(
    source: str,
    *,
    strip_runtime_recovery: bool,
) -> tuple[str, dict[str, int]]:
    try:
        tree = ast.parse(source, filename=_TOOLS_TEST_PATH)
    except SyntaxError as exc:
        raise ValueError("tools/test.py is not valid Python") from exc
    counts = {key: 0 for key in _TOOLS_TEST_ALLOWED_AST_COUNTS}
    if strip_runtime_recovery:
        normalizer = _ToolsTestRuntimeNormalizer()
        tree = normalizer.visit(tree)
        ast.fix_missing_locations(tree)
        counts = normalizer.counts
        if counts != _TOOLS_TEST_ALLOWED_AST_COUNTS:
            raise ValueError(
                "tools/test.py recovery AST scope mismatch: "
                f"expected={_TOOLS_TEST_ALLOWED_AST_COUNTS}, actual={counts}"
            )
    normalized = ast.dump(
        tree,
        annotate_fields=True,
        include_attributes=False,
    )
    return normalized, counts


def _build_tools_test_semantic_evidence(
    *,
    training_source: str,
    runtime_source: str,
    zero_context_patch: bytes,
) -> dict[str, Any]:
    training_ast, _ = _normalized_tools_test_ast(
        training_source,
        strip_runtime_recovery=False,
    )
    runtime_ast, counts = _normalized_tools_test_ast(
        runtime_source,
        strip_runtime_recovery=True,
    )
    training_ast_sha256 = hashlib.sha256(training_ast.encode("utf-8")).hexdigest()
    runtime_ast_sha256 = hashlib.sha256(runtime_ast.encode("utf-8")).hexdigest()
    if runtime_ast_sha256 != training_ast_sha256:
        raise ValueError(
            "tools/test.py changes the protected detector/dataloader/evaluation "
            "inference path"
        )
    if not zero_context_patch:
        raise ValueError("tools/test.py zero-context patch is empty")
    return {
        "tools_test_zero_context_patch_sha256": hashlib.sha256(
            zero_context_patch
        ).hexdigest(),
        "tools_test_zero_context_patch_size_bytes": len(zero_context_patch),
        "tools_test_training_normalized_ast_sha256": training_ast_sha256,
        "tools_test_runtime_normalized_ast_sha256": runtime_ast_sha256,
        "tools_test_allowed_ast_counts": counts,
        "tools_test_semantic_scope": _TOOLS_TEST_SEMANTIC_SCOPE,
    }


def _tools_test_semantic_evidence_between_commits(
    training_commit: str,
    runtime_commit: str,
) -> dict[str, Any]:
    return _build_tools_test_semantic_evidence(
        training_source=_git_file_text(training_commit, _TOOLS_TEST_PATH),
        runtime_source=_git_file_text(runtime_commit, _TOOLS_TEST_PATH),
        zero_context_patch=_tools_test_zero_context_patch(
            training_commit,
            runtime_commit,
        ),
    )


def _changed_files_between_commits(
    base_commit: str,
    target_commit: str,
    *,
    required_paths: set[str],
    allowed_exact_paths: set[str] | None = None,
) -> list[dict[str, str]]:
    allowed_exact_paths = allowed_exact_paths or _ALLOWED_EXACT_PATHS
    lines = _run_git(
        "diff",
        "--name-status",
        "--find-renames=100%",
        f"{base_commit}..{target_commit}",
    ).splitlines()
    changed: list[dict[str, str]] = []
    for line in lines:
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"unsupported S1 recovery Git diff row: {line!r}")
        status, path = fields
        normalized = Path(path).as_posix()
        if status not in {"A", "M"}:
            raise ValueError(
                f"S1 profile recovery forbids Git status {status} for {normalized}"
            )
        if normalized not in allowed_exact_paths and not normalized.startswith(
            _ALLOWED_PREFIXES
        ):
            raise ValueError(
                f"S1 profile recovery changed an unauthorized path: {normalized}"
            )
        changed.append(
            {
                "status": status,
                "path": normalized,
                "file_sha256": _git_file_sha256(target_commit, normalized),
            }
        )
    changed_paths = {row["path"] for row in changed}
    missing = sorted(required_paths - changed_paths)
    if missing:
        raise ValueError(
            f"S1 profile recovery is missing audited repair paths: {missing}"
        )
    return sorted(changed, key=lambda row: row["path"])


def require_clean_profile_checkout(*, expected_commit: str) -> None:
    if current_git_commit() != str(expected_commit).lower():
        raise RuntimeError("S1 profile checkout differs from the recovery commit")
    if _run_git("status", "--porcelain", "--untracked-files=all").strip():
        raise RuntimeError("formal S1 profile recovery requires a clean checkout")


def _changed_files(
    base_commit: str,
    profile_commit: str,
    *,
    required_paths: set[str],
    allowed_exact_paths: set[str] | None = None,
) -> list[dict[str, str]]:
    return _changed_files_between_commits(
        base_commit,
        profile_commit,
        required_paths=required_paths,
        allowed_exact_paths=allowed_exact_paths,
    )


def _assert_no_model_surface_changes(
    base_commit: str,
    target_commit: str,
) -> None:
    changed_paths = [
        Path(line).as_posix()
        for line in _run_git(
            "diff", "--name-only", f"{base_commit}..{target_commit}"
        ).splitlines()
        if line.strip()
    ]
    forbidden = sorted(
        path for path in changed_paths if path.startswith(_MODEL_SURFACE_PREFIXES)
    )
    if forbidden:
        raise ValueError(
            "S1 step-runtime recovery changes model/config surfaces: "
            + ", ".join(forbidden)
        )


def _load_parent_recovery_certificate(
    path: str | Path,
    *,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return validate_profile_recovery_certificate(
        json.loads(path.read_text(encoding="utf-8")),
        binding=binding,
        verify_checkout=False,
    )


def _validate_matrix_submission_receipt(
    path: str | Path,
    *,
    parent: Mapping[str, Any],
    matrix_start: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(path).resolve()
    expected_path = (Path(parent["campaign_root"]) / "matrix_submission.json").resolve()
    if path != expected_path or not path.is_file():
        raise ValueError("S1 v5 matrix submission receipt path mismatch")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, Mapping) or set(receipt) != _MATRIX_SUBMISSION_FIELDS:
        raise ValueError("S1 v5 matrix submission receipt fields mismatch")
    receipt = json.loads(json.dumps(dict(receipt), sort_keys=True))
    receipt_hash = receipt.pop("receipt_sha256", None)
    if not receipt_hash or canonical_sha256(receipt) != receipt_hash:
        raise ValueError("S1 v5 matrix submission receipt self-hash mismatch")
    receipt["receipt_sha256"] = receipt_hash
    try:
        submitted = datetime.fromisoformat(str(receipt["submitted_utc"]))
    except ValueError as exc:
        raise ValueError("S1 v5 matrix submission timestamp is invalid") from exc
    if submitted.tzinfo is None:
        raise ValueError("S1 v5 matrix submission timestamp lacks timezone")

    expected = {
        "schema_version": S1_MATRIX_SUBMISSION_SCHEMA,
        "status": "SUBMITTED",
        "job_id": S1_STEP_RUNTIME_PARENT_JOB_ID,
        "profile_campaign_id": parent["campaign_id"],
        "profile_campaign_root": str(Path(parent["campaign_root"]).resolve()),
        "profile_code_commit": parent["profile_code_commit"],
        "training_code_commit": str(binding["code_commit"]).lower(),
        "recovery_certificate_path": str(
            Path(parent["campaign_root"]).resolve() / "recovery_certificate.json"
        ),
        "recovery_certificate_file_sha256": sha256_file(
            Path(parent["campaign_root"]) / "recovery_certificate.json"
        ),
        "recovery_certificate_sha256": parent["certificate_sha256"],
        "sidecar_gate_path": matrix_start["sidecar_gate_evidence_path"],
        "sidecar_gate_file_sha256": matrix_start["sidecar_gate_evidence_file_sha256"],
        "sidecar_gate_sha256": matrix_start["sidecar_gate_sha256"],
        "frozen_order": build_s1_profile_order(),
        "single_serial_allocation": True,
        "physical_gpu_index_overridden": False,
        "outer_resources": {
            "cpus": 8,
            "gpus": 2,
            "memory_request": "site-default",
        },
        "inner_resources": {
            "cpus": 5,
            "gpus": 1,
            "memory_mib": 96000,
        },
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"S1 v5 matrix submission receipt {key} mismatch")
    if not str(receipt.get("job_name", "")).strip():
        raise ValueError("S1 v5 matrix submission receipt lacks a job name")
    if not str(receipt.get("profile_source_root", "")).strip():
        raise ValueError("S1 v5 matrix submission receipt lacks a source root")
    return receipt


def _validate_v5_completed_descriptor(
    path: str | Path,
    *,
    matrix_start: Mapping[str, Any],
    parent: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from tools.bata.spatial_zoom_s1_matrix import _validated_descriptor

    resolved = Path(path).resolve()
    descriptor, record = _validated_descriptor(resolved, start=matrix_start)
    first = build_s1_profile_order()[0]
    expected_identity = {
        "schema_version": "spatial_zoom_s1_run_v8",
        "profile_order_ordinal": int(first["ordinal"]),
        "resolution": int(first["resolution"]),
        "seed": int(first["seed"]),
        "code_commit": str(binding["code_commit"]).lower(),
        "profile_code_commit": parent["profile_code_commit"],
        "profile_recovery_certificate_sha256": parent["certificate_sha256"],
        "profile_recovery_campaign_id": parent["campaign_id"],
        "matrix_sha256": matrix_start["matrix_sha256"],
        "slurm_job_id": matrix_start["slurm_job_id"],
        "slurm_step_id": matrix_start["slurm_step_id"],
        "step_gpu_uuid": matrix_start["step_gpu_uuid"],
    }
    for key, value in expected_identity.items():
        if descriptor.get(key) != value:
            raise ValueError(f"S1 v5 completed descriptor {key} mismatch")

    artifact_pairs = (
        ("checkpoint_path", "checkpoint_sha256"),
        ("checkpoint_selection_path", "checkpoint_selection_sha256"),
        ("test_evidence_path", "test_evidence_file_sha256"),
        ("test_open_certificate_path", "test_open_certificate_file_sha256"),
        ("test_open_marker_path", "test_open_marker_file_sha256"),
        ("prediction_path", "prediction_sha256"),
        ("profile_summary_path", "profile_summary_sha256"),
        ("profile_samples_path", "profile_samples_sha256"),
        ("profile_power_path", "profile_power_sha256"),
        ("profile_power_attempt_path", "profile_power_attempt_file_sha256"),
        (
            "profile_power_attempt_trace_path",
            "profile_power_attempt_trace_sha256",
        ),
        ("profile_attempt_marker_path", "profile_attempt_marker_file_sha256"),
        (
            "profile_recovery_certificate_path",
            "profile_recovery_certificate_file_sha256",
        ),
        ("sidecar_gate_evidence_path", "sidecar_gate_evidence_file_sha256"),
        ("matrix_start_receipt_path", "matrix_start_receipt_file_sha256"),
        ("ground_truth_path", "ground_truth_sha256"),
    )
    for path_key, hash_key in artifact_pairs:
        if path_key not in descriptor or hash_key not in descriptor:
            raise ValueError(f"S1 v5 completed descriptor lacks {path_key}/{hash_key}")
        artifact = Path(descriptor[path_key]).resolve()
        if not artifact.is_file() or sha256_file(artifact) != descriptor[hash_key]:
            raise ValueError(
                f"S1 v5 completed descriptor artifact mismatch: {artifact}"
            )

    cfg_path = Path(descriptor.get("config_path", "")).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    cfg = Config.fromfile(str(cfg_path))
    if canonical_sha256(cfg.to_dict()) != descriptor.get("resolved_config_sha256"):
        raise ValueError("S1 v5 completed descriptor config hash mismatch")
    descriptor_binding = validate_bound_s1_training_config(cfg, seed=int(first["seed"]))
    for key in (
        "code_commit",
        "experiment_namespace",
        "canonical_experiment_root",
        "manifest_sha256",
        "protocol_fingerprint",
        "precheck_file_sha256",
        "precheck_sha256",
        "pretrained_checkpoint_sha256",
    ):
        if (
            descriptor_binding[key] != binding[key]
            or descriptor.get(key) != binding[key]
        ):
            raise ValueError(f"S1 v5 completed descriptor binding {key} mismatch")

    manifest = json.loads(Path(descriptor["manifest_path"]).read_text(encoding="utf-8"))
    from tools.bata.profile_spatial_zoom_s1 import validate_profile_attempt_marker
    from tools.bata.select_spatial_zoom_s1_checkpoint import (
        validate_checkpoint_selection,
    )
    from tools.bata.spatial_zoom_s1_cost import validate_profile_summary
    from tools.bata.spatial_zoom_s1_evidence import validate_s1_test_evidence
    from tools.bata.spatial_zoom_s1_power import validate_nvml_sidecar_attempt
    from tools.bata.spatial_zoom_s1_sidecar_gate import load_sidecar_gate_evidence

    selection = validate_checkpoint_selection(
        json.loads(
            Path(descriptor["checkpoint_selection_path"]).read_text(encoding="utf-8")
        ),
        config=cfg,
        seed=int(first["seed"]),
        manifest=manifest,
        checkpoint_path=descriptor["checkpoint_path"],
        protocol_fingerprint=binding["protocol_fingerprint"],
    )
    if (
        selection["selection_sha256"]
        != descriptor["checkpoint_selection_internal_sha256"]
    ):
        raise ValueError("S1 v5 completed descriptor selection identity mismatch")
    test_evidence = validate_s1_test_evidence(
        json.loads(Path(descriptor["test_evidence_path"]).read_text(encoding="utf-8")),
        cfg=cfg,
        seed=int(first["seed"]),
    )
    if test_evidence["evidence_sha256"] != descriptor["test_evidence_sha256"]:
        raise ValueError("S1 v5 completed descriptor test evidence mismatch")
    marker = validate_profile_attempt_marker(descriptor["profile_attempt_marker_path"])
    if marker["marker_sha256"] != descriptor["profile_attempt_marker_sha256"]:
        raise ValueError("S1 v5 completed descriptor marker mismatch")
    profile = validate_profile_summary(
        json.loads(Path(descriptor["profile_summary_path"]).read_text(encoding="utf-8"))
    )
    if profile["profile_sha256"] != descriptor["profile_summary_internal_sha256"]:
        raise ValueError("S1 v5 completed descriptor profile identity mismatch")
    attempt = validate_nvml_sidecar_attempt(
        descriptor["profile_power_attempt_path"],
        descriptor["profile_power_attempt_trace_path"],
        expected_uuid=matrix_start["step_gpu_uuid"],
        require_pass=True,
    )
    if attempt["attempt_sha256"] != descriptor["profile_power_attempt_sha256"]:
        raise ValueError("S1 v5 completed descriptor power attempt mismatch")
    gate = load_sidecar_gate_evidence(
        descriptor["sidecar_gate_evidence_path"],
        recovery=parent,
    )
    if gate["gate_sha256"] != descriptor["sidecar_gate_sha256"]:
        raise ValueError("S1 v5 completed descriptor sidecar Gate mismatch")
    return descriptor, record


def _validate_failed_ordinal_from_stdout(
    text: str,
    *,
    completed_descriptor_path: Path,
) -> None:
    completed_line = (
        "[SPATIAL_ZOOM_S1_TEST_PROFILE] PASS resolution=256 seed=3408 "
        f"descriptor={completed_descriptor_path}"
    )
    completed_index = text.find(completed_line)
    if completed_index < 0:
        raise ValueError("S1 v5 stdout lacks the ordinal-0 completion record")
    tail = text[completed_index + len(completed_line) :]
    decoder = json.JSONDecoder()
    records: list[dict[str, Any]] = []
    cursor = 0
    while True:
        start = tail.find("{", cursor)
        if start < 0:
            break
        try:
            record, consumed = decoder.raw_decode(tail[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        cursor = start + consumed
        if isinstance(record, Mapping):
            records.append(dict(record))

    ordinal_records: list[dict[str, Any]] = []
    for record in records:
        if "profile_order_ordinal" not in record:
            continue
        try:
            ordinal = int(record["profile_order_ordinal"])
        except (TypeError, ValueError) as exc:
            raise ValueError("S1 v5 stdout has an invalid profile ordinal") from exc
        ordinal_records.append({**record, "profile_order_ordinal": ordinal})

    failed_records = [
        record
        for record in ordinal_records
        if record["profile_order_ordinal"] == S1_STEP_RUNTIME_FAILED_ORDINAL
    ]
    if len(failed_records) != 1:
        raise ValueError(
            "S1 v5 stdout must contain exactly one structured ordinal-1 record"
        )
    failed_record = failed_records[0]
    try:
        failed_resolution = int(failed_record.get("resolution", -1))
        failed_seed = int(failed_record.get("seed", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("S1 v5 stdout ordinal-1 record identity mismatch") from exc
    if (
        failed_resolution != S1_STEP_RUNTIME_FAILED_RESOLUTION
        or failed_seed != S1_STEP_RUNTIME_FAILED_SEED
    ):
        raise ValueError("S1 v5 stdout ordinal-1 record identity mismatch")

    later_ordinals = sorted(
        {
            int(record["profile_order_ordinal"])
            for record in ordinal_records
            if int(record["profile_order_ordinal"]) > S1_STEP_RUNTIME_FAILED_ORDINAL
        }
    )
    if later_ordinals:
        raise ValueError(
            "S1 v5 stdout advances beyond failed ordinal-1: " f"{later_ordinals}"
        )


def _validate_v3_matrix_start_receipt(
    matrix_start_path: str | Path,
    *,
    parent_recovery_path: str | Path,
) -> dict[str, Any]:
    from tools.bata.spatial_zoom_s1_matrix import (
        validate_profile_matrix_start_receipt,
    )

    return validate_profile_matrix_start_receipt(
        Path(matrix_start_path).resolve(),
        recovery=Path(parent_recovery_path).resolve(),
        verify_runtime=False,
    )


def _validate_superseded_marker(
    path: Path, *, expected_schema: str = S1_SUPERSEDED_PROFILE_MARKER_SCHEMA
) -> dict[str, Any]:
    marker = json.loads(path.read_text(encoding="utf-8"))
    marker_hash = marker.pop("marker_sha256", None)
    if not marker_hash or canonical_sha256(marker) != marker_hash:
        raise ValueError("superseded S1 profile marker self-hash mismatch")
    marker["marker_sha256"] = marker_hash
    if marker.get("schema_version") != expected_schema:
        raise ValueError(
            f"S1 recovery requires the exact failed {expected_schema} marker"
        )
    return marker


def _validate_power_diagnostic(path: Path) -> dict[str, Any]:
    diagnostic = json.loads(path.read_text(encoding="utf-8"))
    diagnostic_hash = diagnostic.pop("diagnostic_sha256", None)
    if not diagnostic_hash or canonical_sha256(diagnostic) != diagnostic_hash:
        raise ValueError("S1 power diagnostic self-hash mismatch")
    diagnostic["diagnostic_sha256"] = diagnostic_hash
    code_commit = str(diagnostic.get("code_commit", ""))
    if (
        diagnostic.get("schema_version") != S1_POWER_DIAGNOSTIC_SCHEMA
        or diagnostic.get("reads_test_data") is not False
        or diagnostic.get("paper_claim_allowed") is not False
        or int(diagnostic.get("target_interval_ms", -1)) != 20
        or float(diagnostic.get("duration_seconds_per_backend", 0.0)) < 5.0
        or len(code_commit) != 40
        or any(character not in "0123456789abcdef" for character in code_commit)
        or not str(diagnostic.get("node", "")).strip()
        or not str(diagnostic.get("gpu_uuid", "")).startswith("GPU-")
        or not str(diagnostic.get("slurm_job_id", "")).isdigit()
    ):
        raise ValueError("S1 power diagnostic provenance is invalid")
    backends = {
        str(row.get("backend")): row
        for row in diagnostic.get("backends", ())
        if isinstance(row, Mapping)
    }
    inherited = backends.get("nvidia-smi-persistent-loop-ms")
    candidate = backends.get("nvml-persistent-poll-v1")
    if not inherited or not candidate:
        raise ValueError("S1 power diagnostic is missing a matched backend")
    if (
        inherited.get("status") != "FAIL"
        or inherited.get("cadence", {}).get("formal_cadence_pass") is not False
        or candidate.get("status") != "PASS"
        or candidate.get("cadence", {}).get("formal_cadence_pass") is not True
        or float(candidate.get("cadence", {}).get("max_gap_ms", math.inf)) > 100.0
        or float(candidate.get("cadence", {}).get("max_gap_limit_ms", -1.0)) != 100.0
    ):
        raise ValueError("S1 power diagnostic does not justify the NVML backend")
    return diagnostic


def profile_campaign_prefix(
    certificate: Mapping[str, Any], *, resolution: int, seed: int
) -> Path:
    if int(resolution) not in (160, 224, 256) or int(seed) not in (
        3407,
        3408,
        3409,
    ):
        raise ValueError("invalid S1 profile campaign cell")
    return (
        Path(certificate["campaign_root"])
        / f"dense{int(resolution)}"
        / f"seed{int(seed)}"
        / f"dense{int(resolution)}_seed{int(seed)}"
    ).resolve()


def _certificate_output_path(certificate: Mapping[str, Any]) -> Path:
    return (Path(certificate["campaign_root"]) / "recovery_certificate.json").resolve()


def _legacy_unbound_test_evidence_path(binding: Mapping[str, Any]) -> Path:
    first_profile_cell = build_s1_profile_order()[0]
    return (
        Path(binding["canonical_experiment_root"])
        / f"dense{int(first_profile_cell['resolution'])}"
        / f"seed{int(first_profile_cell['seed'])}"
        / "gpu1_id0"
        / "test_evidence"
        / "test.evidence.json"
    ).resolve()


def build_sidecar_profile_recovery_certificate(
    *,
    binding: Mapping[str, Any],
    failed_marker_path: str | Path,
    failure_log_path: str | Path,
    failed_job_id: str,
    expected_exposure_count: int,
    expected_physical_window_count: int,
    expected_duplicate_physical_window_ids: Sequence[str],
    superseded_recovery_certificate_path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Create the v3 recovery that isolates power polling from the detector."""

    profile_commit = current_git_commit()
    require_clean_profile_checkout(expected_commit=profile_commit)
    training_commit = str(binding["code_commit"]).lower()
    changed_files = _changed_files(
        training_commit,
        profile_commit,
        required_paths=_REQUIRED_REPAIR_PATHS_SIDECAR,
    )
    parent_path = Path(superseded_recovery_certificate_path).resolve()
    failed_marker_path = Path(failed_marker_path).resolve()
    failure_log_path = Path(failure_log_path).resolve()
    for path in (parent_path, failed_marker_path, failure_log_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    parent = validate_profile_recovery_certificate(
        json.loads(parent_path.read_text(encoding="utf-8")),
        binding=binding,
        verify_checkout=False,
    )
    if parent.get("reason") != S1_CHAINED_RECOVERY_REASON:
        raise ValueError("S1 sidecar recovery requires the v2 NVML parent")
    failed_marker = _validate_superseded_marker(
        failed_marker_path,
        expected_schema=S1_CHAINED_PROFILE_MARKER_SCHEMA,
    )
    failure_text = failure_log_path.read_text(encoding="utf-8", errors="replace")
    if S1_POWER_FAILURE_SIGNATURE not in failure_text:
        raise ValueError("S1 sidecar recovery log lacks the frozen cadence failure")
    if not str(failed_job_id).isdigit():
        raise ValueError("S1 sidecar recovery requires a numeric failed Slurm job id")

    first_profile_cell = build_s1_profile_order()[0]
    marker_expected = {
        "resolution": int(first_profile_cell["resolution"]),
        "seed": int(first_profile_cell["seed"]),
        "code_commit": training_commit,
        "profile_code_commit": parent["profile_code_commit"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_sha256": binding["manifest_sha256"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "profile_order_ordinal": 0,
        "test_open_certificate_sha256": parent["test_open_certificate_sha256"],
        "profile_recovery_certificate_sha256": parent["certificate_sha256"],
        "profile_recovery_campaign_id": parent["campaign_id"],
    }
    for key, expected in marker_expected.items():
        if failed_marker.get(key) != expected:
            raise ValueError(f"S1 sidecar recovery marker {key} mismatch")

    expected_first_work_dir = _legacy_unbound_test_evidence_path(binding).parents[2]
    if Path(binding["work_dir"]).resolve() != expected_first_work_dir:
        raise ValueError(
            "S1 sidecar recovery must be issued from the frozen first cell"
        )
    legacy_test_evidence_path = _legacy_unbound_test_evidence_path(binding)
    if not legacy_test_evidence_path.is_file():
        raise FileNotFoundError(legacy_test_evidence_path)
    legacy_test_evidence = json.loads(
        legacy_test_evidence_path.read_text(encoding="utf-8")
    )
    legacy_test_evidence_sha256 = legacy_test_evidence.pop("evidence_sha256", None)
    if (
        not legacy_test_evidence_sha256
        or canonical_sha256(legacy_test_evidence) != legacy_test_evidence_sha256
        or failed_marker.get("test_evidence_sha256") != legacy_test_evidence_sha256
    ):
        raise ValueError("S1 sidecar recovery legacy test evidence mismatch")

    exposure_count = int(expected_exposure_count)
    physical_count = int(expected_physical_window_count)
    duplicates = sorted(set(map(str, expected_duplicate_physical_window_ids)))
    if (
        exposure_count != int(parent["expected_loader_exposure_count"])
        or physical_count != int(parent["expected_physical_window_count"])
        or duplicates != parent["expected_duplicate_physical_window_ids"]
        or exposure_count - physical_count != len(duplicates)
        or not duplicates
    ):
        raise ValueError("S1 sidecar recovery changed the frozen exposure topology")

    basis = {
        "schema_version": S1_SIDECAR_PROFILE_RECOVERY_SCHEMA,
        "reason": S1_SIDECAR_RECOVERY_REASON,
        "failure_signature": parent["failure_signature"],
        "failed_job_id": parent["failed_job_id"],
        "training_code_commit": training_commit,
        "profile_code_commit": profile_commit,
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_sha256": binding["manifest_sha256"],
        "protocol_fingerprint": binding["protocol_fingerprint"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "test_open_certificate_sha256": parent["test_open_certificate_sha256"],
        "legacy_unbound_test_resolution": int(first_profile_cell["resolution"]),
        "legacy_unbound_test_seed": int(first_profile_cell["seed"]),
        "legacy_unbound_test_evidence_path": str(legacy_test_evidence_path),
        "legacy_unbound_test_evidence_file_sha256": sha256_file(
            legacy_test_evidence_path
        ),
        "legacy_unbound_test_evidence_sha256": legacy_test_evidence_sha256,
        "superseded_marker_path": parent["superseded_marker_path"],
        "superseded_marker_file_sha256": parent["superseded_marker_file_sha256"],
        "superseded_marker_sha256": parent["superseded_marker_sha256"],
        "failure_log_path": parent["failure_log_path"],
        "failure_log_sha256": parent["failure_log_sha256"],
        "expected_loader_exposure_count": exposure_count,
        "expected_physical_window_count": physical_count,
        "expected_duplicate_physical_window_ids": duplicates,
        "changed_files": changed_files,
        "repair_scope": "out_of_process_power_sidecar_and_failure_evidence_only",
        "preserve_all_loader_exposures": True,
        "preserve_superseded_attempt": True,
        "reuse_valid_test_evidence": True,
        "preserve_recovery_chain": True,
        "superseded_recovery_certificate_path": str(parent_path),
        "superseded_recovery_certificate_file_sha256": sha256_file(parent_path),
        "superseded_recovery_certificate_sha256": parent["certificate_sha256"],
        "superseded_recovery_campaign_id": parent["campaign_id"],
        "superseded_recovery_profile_code_commit": parent["profile_code_commit"],
        "sidecar_power_failure_signature": S1_POWER_FAILURE_SIGNATURE,
        "sidecar_power_failed_job_id": str(failed_job_id),
        "sidecar_power_failure_marker_path": str(failed_marker_path),
        "sidecar_power_failure_marker_file_sha256": sha256_file(failed_marker_path),
        "sidecar_power_failure_marker_sha256": failed_marker["marker_sha256"],
        "sidecar_power_failure_log_path": str(failure_log_path),
        "sidecar_power_failure_log_sha256": sha256_file(failure_log_path),
        "power_sampler_backend": S1_SIDECAR_POWER_BACKEND,
        "power_target_interval_ms": 20,
        "power_max_gap_limit_ms": 100.0,
        "allocated_cpu_count": 5,
        "detector_cpu_count": 4,
        "sidecar_cpu_count": 1,
        "requires_long_no_open_gate": True,
        "sidecar_gate_relative_path": "sidecar_gate.json",
    }
    campaign_id = canonical_sha256(basis)[:16]
    campaign_root = (
        Path(binding["canonical_experiment_root"]) / "profile_campaigns" / campaign_id
    ).resolve()
    certificate = {
        **basis,
        "campaign_id": campaign_id,
        "campaign_root": str(campaign_root),
    }
    certificate["certificate_sha256"] = canonical_sha256(certificate)
    output_path = _certificate_output_path(certificate)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != certificate:
            raise FileExistsError("S1 sidecar recovery campaign identity collision")
    else:
        atomic_publish_json(output_path, certificate)
    return output_path, certificate


def build_buffered_sidecar_profile_recovery_certificate(
    *,
    binding: Mapping[str, Any],
    failed_marker_path: str | Path,
    failure_log_path: str | Path,
    failed_job_id: str,
    expected_exposure_count: int,
    expected_physical_window_count: int,
    expected_duplicate_physical_window_ids: Sequence[str],
    superseded_recovery_certificate_path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Create the v4 recovery that removes trace I/O from the sampling loop."""

    profile_commit = current_git_commit()
    require_clean_profile_checkout(expected_commit=profile_commit)
    training_commit = str(binding["code_commit"]).lower()
    changed_files = _changed_files(
        training_commit,
        profile_commit,
        required_paths=_REQUIRED_REPAIR_PATHS_BUFFERED_SIDECAR,
    )
    parent_path = Path(superseded_recovery_certificate_path).resolve()
    failed_marker_path = Path(failed_marker_path).resolve()
    failure_log_path = Path(failure_log_path).resolve()
    for path in (parent_path, failed_marker_path, failure_log_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    parent = validate_profile_recovery_certificate(
        json.loads(parent_path.read_text(encoding="utf-8")),
        binding=binding,
        verify_checkout=False,
    )
    if parent.get("reason") != S1_SIDECAR_RECOVERY_REASON:
        raise ValueError("S1 buffered-sidecar recovery requires the v3 parent")
    parent_to_current_changed_files = _changed_files_between_commits(
        parent["profile_code_commit"],
        profile_commit,
        required_paths={"tools/bata/spatial_zoom_s1_power.py"},
    )
    parent_sampling_implementation_sha256 = _git_file_sha256(
        parent["profile_code_commit"],
        "tools/bata/spatial_zoom_s1_power.py",
    )
    sampling_implementation_sha256 = _git_file_sha256(
        profile_commit,
        "tools/bata/spatial_zoom_s1_power.py",
    )
    if parent_sampling_implementation_sha256 == sampling_implementation_sha256:
        raise ValueError(
            "S1 buffered-sidecar recovery did not change the sampling implementation"
        )
    failed_marker = _validate_superseded_marker(
        failed_marker_path,
        expected_schema=S1_BUFFERED_SIDECAR_PROFILE_MARKER_SCHEMA,
    )
    failure_text = failure_log_path.read_text(encoding="utf-8", errors="replace")
    if S1_BUFFERED_SIDECAR_FAILURE_SIGNATURE not in failure_text:
        raise ValueError("S1 buffered-sidecar recovery log lacks cadence failure")
    if not str(failed_job_id).isdigit():
        raise ValueError(
            "S1 buffered-sidecar recovery requires a numeric failed Slurm job id"
        )

    first_profile_cell = build_s1_profile_order()[0]
    expected_first_work_dir = _legacy_unbound_test_evidence_path(binding).parents[2]
    if Path(binding["work_dir"]).resolve() != expected_first_work_dir:
        raise ValueError(
            "S1 buffered-sidecar recovery must be issued from the frozen first cell"
        )
    marker_expected = {
        "resolution": int(first_profile_cell["resolution"]),
        "seed": int(first_profile_cell["seed"]),
        "code_commit": training_commit,
        "profile_code_commit": parent["profile_code_commit"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_sha256": binding["manifest_sha256"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "profile_order_ordinal": 0,
        "test_open_certificate_sha256": parent["test_open_certificate_sha256"],
        "profile_recovery_certificate_sha256": parent["certificate_sha256"],
        "profile_recovery_campaign_id": parent["campaign_id"],
        "power_sampler_backend": S1_SIDECAR_POWER_BACKEND,
        "slurm_job_id": str(failed_job_id),
    }
    for key, expected in marker_expected.items():
        if failed_marker.get(key) != expected:
            raise ValueError(f"S1 buffered-sidecar marker {key} mismatch")
    if failed_marker.get("gate_only") is not False:
        raise ValueError("S1 buffered-sidecar recovery requires a formal attempt")

    legacy_test_evidence_path = _legacy_unbound_test_evidence_path(binding)
    if not legacy_test_evidence_path.is_file():
        raise FileNotFoundError(legacy_test_evidence_path)
    legacy_test_evidence = json.loads(
        legacy_test_evidence_path.read_text(encoding="utf-8")
    )
    legacy_test_evidence_sha256 = legacy_test_evidence.pop("evidence_sha256", None)
    if (
        not legacy_test_evidence_sha256
        or canonical_sha256(legacy_test_evidence) != legacy_test_evidence_sha256
        or failed_marker.get("test_evidence_sha256") != legacy_test_evidence_sha256
        or parent.get("legacy_unbound_test_evidence_sha256")
        != legacy_test_evidence_sha256
    ):
        raise ValueError("S1 buffered-sidecar legacy test evidence mismatch")

    canonical_prefix = Path(failed_marker["canonical_output_prefix"]).resolve()
    expected_prefix = profile_campaign_prefix(
        parent,
        resolution=int(first_profile_cell["resolution"]),
        seed=int(first_profile_cell["seed"]),
    )
    if canonical_prefix != expected_prefix:
        raise ValueError("S1 buffered-sidecar failure prefix mismatch")
    attempt_report_path = Path(f"{canonical_prefix}.power_attempt.json")
    attempt_trace_path = Path(f"{canonical_prefix}.power_attempt.jsonl")
    parent_failure_path = Path(f"{canonical_prefix}.power_parent_failure.json")
    for path in (attempt_report_path, attempt_trace_path, parent_failure_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    attempt = validate_nvml_sidecar_cadence_failure(
        attempt_report_path,
        attempt_trace_path,
        expected_uuid=str(failed_marker["step_gpu_uuid"]),
    )
    if attempt.get("backend") != S1_SIDECAR_POWER_BACKEND or float(
        attempt.get("cadence", {}).get("max_gap_limit_ms", -1.0)
    ) != float(parent["power_max_gap_limit_ms"]):
        raise ValueError("S1 buffered-sidecar attempt does not prove cadence failure")

    parent_failure = json.loads(parent_failure_path.read_text(encoding="utf-8"))
    parent_failure_hash = parent_failure.pop("parent_failure_sha256", None)
    if (
        not parent_failure_hash
        or canonical_sha256(parent_failure) != parent_failure_hash
        or parent_failure.get("schema_version")
        != "spatial_zoom_s1_profile_parent_failure_v1"
        or parent_failure.get("status") != "FAIL"
        or parent_failure.get("paper_claim_allowed") is not False
        or parent_failure.get("power_attempt_sha256") != attempt["attempt_sha256"]
        or parent_failure.get("power_attempt_report_file_sha256")
        != sha256_file(attempt_report_path)
        or parent_failure.get("power_attempt_trace_file_sha256")
        != sha256_file(attempt_trace_path)
    ):
        raise ValueError("S1 buffered-sidecar parent-failure evidence mismatch")
    parent_failure["parent_failure_sha256"] = parent_failure_hash

    matrix_start_path = Path(failed_marker["matrix_start_receipt_path"]).resolve()
    if not matrix_start_path.is_file() or sha256_file(
        matrix_start_path
    ) != failed_marker.get("matrix_start_receipt_file_sha256"):
        raise ValueError("S1 buffered-sidecar matrix-start file mismatch")
    matrix_start = _validate_v3_matrix_start_receipt(
        matrix_start_path,
        parent_recovery_path=parent_path,
    )
    matrix_hash = matrix_start["matrix_sha256"]
    if (
        matrix_hash != failed_marker.get("matrix_sha256")
        or matrix_start.get("slurm_job_id") != str(failed_job_id)
        or matrix_start.get("slurm_step_id") != failed_marker.get("slurm_step_id")
        or matrix_start.get("step_gpu_uuid") != failed_marker.get("step_gpu_uuid")
        or matrix_start.get("profile_code_commit") != parent["profile_code_commit"]
        or matrix_start.get("profile_recovery_certificate_sha256")
        != parent["certificate_sha256"]
        or matrix_start.get("profile_recovery_campaign_id") != parent["campaign_id"]
        or matrix_start.get("frozen_order") != build_s1_profile_order()
    ):
        raise ValueError("S1 buffered-sidecar matrix-start identity mismatch")

    exposure_count = int(expected_exposure_count)
    physical_count = int(expected_physical_window_count)
    duplicates = sorted(set(map(str, expected_duplicate_physical_window_ids)))
    if (
        exposure_count != int(parent["expected_loader_exposure_count"])
        or physical_count != int(parent["expected_physical_window_count"])
        or duplicates != parent["expected_duplicate_physical_window_ids"]
        or exposure_count - physical_count != len(duplicates)
        or not duplicates
    ):
        raise ValueError(
            "S1 buffered-sidecar recovery changed the frozen exposure topology"
        )

    basis = {
        "schema_version": S1_BUFFERED_SIDECAR_PROFILE_RECOVERY_SCHEMA,
        "reason": S1_BUFFERED_SIDECAR_RECOVERY_REASON,
        "failure_signature": parent["failure_signature"],
        "failed_job_id": parent["failed_job_id"],
        "training_code_commit": training_commit,
        "profile_code_commit": profile_commit,
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_sha256": binding["manifest_sha256"],
        "protocol_fingerprint": binding["protocol_fingerprint"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "test_open_certificate_sha256": parent["test_open_certificate_sha256"],
        "legacy_unbound_test_resolution": int(first_profile_cell["resolution"]),
        "legacy_unbound_test_seed": int(first_profile_cell["seed"]),
        "legacy_unbound_test_evidence_path": str(legacy_test_evidence_path),
        "legacy_unbound_test_evidence_file_sha256": sha256_file(
            legacy_test_evidence_path
        ),
        "legacy_unbound_test_evidence_sha256": legacy_test_evidence_sha256,
        "superseded_marker_path": parent["superseded_marker_path"],
        "superseded_marker_file_sha256": parent["superseded_marker_file_sha256"],
        "superseded_marker_sha256": parent["superseded_marker_sha256"],
        "failure_log_path": parent["failure_log_path"],
        "failure_log_sha256": parent["failure_log_sha256"],
        "expected_loader_exposure_count": exposure_count,
        "expected_physical_window_count": physical_count,
        "expected_duplicate_physical_window_ids": duplicates,
        "changed_files": changed_files,
        "parent_to_current_changed_files": parent_to_current_changed_files,
        "parent_sampling_implementation_sha256": (
            parent_sampling_implementation_sha256
        ),
        "sampling_implementation_sha256": sampling_implementation_sha256,
        "repair_scope": "buffered_sidecar_trace_publication_only",
        "preserve_all_loader_exposures": True,
        "preserve_superseded_attempt": True,
        "reuse_valid_test_evidence": True,
        "preserve_recovery_chain": True,
        "superseded_recovery_certificate_path": str(parent_path),
        "superseded_recovery_certificate_file_sha256": sha256_file(parent_path),
        "superseded_recovery_certificate_sha256": parent["certificate_sha256"],
        "superseded_recovery_campaign_id": parent["campaign_id"],
        "superseded_recovery_profile_code_commit": parent["profile_code_commit"],
        "buffered_sidecar_failure_signature": (S1_BUFFERED_SIDECAR_FAILURE_SIGNATURE),
        "buffered_sidecar_failed_job_id": str(failed_job_id),
        "buffered_sidecar_failed_slurm_step_id": failed_marker["slurm_step_id"],
        "buffered_sidecar_failed_gpu_uuid": failed_marker["step_gpu_uuid"],
        "buffered_sidecar_failure_marker_path": str(failed_marker_path),
        "buffered_sidecar_failure_marker_file_sha256": sha256_file(failed_marker_path),
        "buffered_sidecar_failure_marker_sha256": failed_marker["marker_sha256"],
        "buffered_sidecar_failure_log_path": str(failure_log_path),
        "buffered_sidecar_failure_log_sha256": sha256_file(failure_log_path),
        "buffered_sidecar_attempt_report_path": str(attempt_report_path),
        "buffered_sidecar_attempt_report_file_sha256": sha256_file(attempt_report_path),
        "buffered_sidecar_attempt_sha256": attempt["attempt_sha256"],
        "buffered_sidecar_attempt_trace_path": str(attempt_trace_path),
        "buffered_sidecar_attempt_trace_file_sha256": sha256_file(attempt_trace_path),
        "buffered_sidecar_parent_failure_path": str(parent_failure_path),
        "buffered_sidecar_parent_failure_file_sha256": sha256_file(parent_failure_path),
        "buffered_sidecar_parent_failure_sha256": parent_failure_hash,
        "buffered_sidecar_matrix_start_path": str(matrix_start_path),
        "buffered_sidecar_matrix_start_file_sha256": sha256_file(matrix_start_path),
        "buffered_sidecar_matrix_sha256": matrix_hash,
        "power_sampler_backend": S1_SIDECAR_POWER_BACKEND,
        "trace_publication_mode": S1_BUFFERED_TRACE_PUBLICATION_MODE,
        "trace_io_inside_sampling_loop": False,
        "power_target_interval_ms": 20,
        "power_max_gap_limit_ms": 100.0,
        "allocated_cpu_count": 5,
        "detector_cpu_count": 4,
        "sidecar_cpu_count": 1,
        "requires_long_no_open_gate": True,
        "sidecar_gate_relative_path": "sidecar_gate.json",
    }
    campaign_id = canonical_sha256(basis)[:16]
    campaign_root = (
        Path(binding["canonical_experiment_root"]) / "profile_campaigns" / campaign_id
    ).resolve()
    certificate = {
        **basis,
        "campaign_id": campaign_id,
        "campaign_root": str(campaign_root),
    }
    certificate["certificate_sha256"] = canonical_sha256(certificate)
    output_path = _certificate_output_path(certificate)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != certificate:
            raise FileExistsError(
                "S1 buffered-sidecar recovery campaign identity collision"
            )
    else:
        atomic_publish_json(output_path, certificate)
    return output_path, certificate


def build_step_runtime_profile_recovery_certificate(
    *,
    binding: Mapping[str, Any],
    superseded_recovery_certificate_path: str | Path,
    matrix_start_receipt_path: str | Path,
    matrix_submission_receipt_path: str | Path,
    matrix_stdout_path: str | Path,
    matrix_stderr_path: str | Path,
    completed_descriptor_path: str | Path,
    failed_job_id: str,
) -> tuple[Path, dict[str, Any]]:
    """Create v5 after the v4 matrix fails in ordinal-1 formal test startup."""

    profile_commit = current_git_commit()
    require_clean_profile_checkout(expected_commit=profile_commit)
    formal_python_environment = _formal_python_environment_evidence()
    training_commit = str(binding["code_commit"]).lower()
    if training_commit != S1_STEP_RUNTIME_TRAINING_COMMIT:
        raise ValueError(
            "S1 v5 is frozen to training/model commit "
            f"{S1_STEP_RUNTIME_TRAINING_COMMIT}"
        )
    parent_path = Path(superseded_recovery_certificate_path).resolve()
    parent = _load_parent_recovery_certificate(parent_path, binding=binding)
    if parent.get("reason") != S1_BUFFERED_SIDECAR_RECOVERY_REASON:
        raise ValueError("S1 step-runtime recovery requires the v4 parent")
    if str(failed_job_id) != S1_STEP_RUNTIME_PARENT_JOB_ID:
        raise ValueError(
            f"S1 v5 is frozen to failed Job {S1_STEP_RUNTIME_PARENT_JOB_ID}"
        )

    changed_files = _changed_files(
        training_commit,
        profile_commit,
        required_paths=_REQUIRED_REPAIR_PATHS_STEP_RUNTIME,
        allowed_exact_paths=_ALLOWED_EXACT_PATHS_STEP_RUNTIME,
    )
    parent_to_current_changed_files = _changed_files_between_commits(
        parent["profile_code_commit"],
        profile_commit,
        required_paths=_REQUIRED_PARENT_TO_STEP_RUNTIME_PATHS,
        allowed_exact_paths=_PARENT_TO_STEP_RUNTIME_PATHS,
    )
    _assert_no_model_surface_changes(training_commit, profile_commit)
    _assert_no_model_surface_changes(parent["profile_code_commit"], profile_commit)
    tools_test_semantic_evidence = _tools_test_semantic_evidence_between_commits(
        training_commit,
        profile_commit,
    )
    parent_runtime_hashes = {
        path: _git_file_sha256(parent["profile_code_commit"], path)
        for path in sorted(_REQUIRED_STEP_RUNTIME_CODE_PATHS)
    }
    runtime_hashes = {
        path: _git_file_sha256(profile_commit, path)
        for path in sorted(_REQUIRED_STEP_RUNTIME_CODE_PATHS)
    }
    unchanged = sorted(
        path
        for path in _REQUIRED_STEP_RUNTIME_CODE_PATHS
        if parent_runtime_hashes[path] == runtime_hashes[path]
    )
    if unchanged:
        raise ValueError(
            "S1 v5 runtime repair did not change required files: "
            + ", ".join(unchanged)
        )

    matrix_start_path = Path(matrix_start_receipt_path).resolve()
    expected_start_path = (
        Path(parent["campaign_root"]) / "matrix.lock" / "matrix.started.json"
    ).resolve()
    if matrix_start_path != expected_start_path or not matrix_start_path.is_file():
        raise ValueError("S1 v5 matrix start receipt path mismatch")
    matrix_start = _validate_v3_matrix_start_receipt(
        matrix_start_path,
        parent_recovery_path=parent_path,
    )
    frozen_order = build_s1_profile_order()
    if (
        matrix_start.get("slurm_job_id") != S1_STEP_RUNTIME_PARENT_JOB_ID
        or matrix_start.get("frozen_order") != frozen_order
        or int(matrix_start.get("effective_step_memory_limit_mb", -1)) != 96000
        or int(matrix_start.get("slurm_cpus_per_task", -1)) != 5
        or matrix_start.get("profile_code_commit") != parent["profile_code_commit"]
        or matrix_start.get("profile_recovery_certificate_sha256")
        != parent["certificate_sha256"]
        or matrix_start.get("profile_recovery_campaign_id") != parent["campaign_id"]
    ):
        raise ValueError("S1 v5 matrix start identity mismatch")
    if (
        not str(matrix_start.get("slurm_step_id", "")).strip()
        or not str(matrix_start.get("step_gpu_uuid", "")).startswith("GPU-")
        or str(matrix_start.get("slurm_step_gpus", "")).count(",") != 0
        or str(matrix_start["slurm_step_gpus"])
        not in str(matrix_start.get("slurm_job_gpus", "")).split(",")
    ):
        raise ValueError("S1 v5 matrix start step/GPU identity is invalid")

    submission_path = Path(matrix_submission_receipt_path).resolve()
    submission = _validate_matrix_submission_receipt(
        submission_path,
        parent=parent,
        matrix_start=matrix_start,
        binding=binding,
    )
    stdout_path = Path(matrix_stdout_path).resolve()
    stderr_path = Path(matrix_stderr_path).resolve()
    expected_logs = {
        stdout_path: (
            Path(parent["campaign_root"])
            / "logs"
            / f"matrix-{S1_STEP_RUNTIME_PARENT_JOB_ID}.out"
        ).resolve(),
        stderr_path: (
            Path(parent["campaign_root"])
            / "logs"
            / f"matrix-{S1_STEP_RUNTIME_PARENT_JOB_ID}.err"
        ).resolve(),
    }
    for path, expected in expected_logs.items():
        if path != expected or not path.is_file():
            raise ValueError(f"S1 v5 matrix log path mismatch: {path}")
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    if stderr_text.count(S1_STEP_RUNTIME_FAILURE_SIGNATURE) != 1:
        raise ValueError("S1 v5 stderr lacks the exact frozen failure signature")

    descriptor_path = Path(completed_descriptor_path).resolve()
    descriptor_paths = sorted(
        (Path(parent["campaign_root"]) / "descriptors").glob("*.run.json")
    )
    if descriptor_paths != [descriptor_path]:
        raise ValueError("S1 v5 requires exactly one completed matrix descriptor")
    descriptor, descriptor_record = _validate_v5_completed_descriptor(
        descriptor_path,
        matrix_start=matrix_start,
        parent=parent,
        binding=binding,
    )
    if (
        int(descriptor_record["profile_order_ordinal"]) != 0
        or int(descriptor_record["resolution"]) != 256
        or int(descriptor_record["seed"]) != 3408
    ):
        raise ValueError("S1 v5 completed descriptor is not frozen ordinal-0")
    _validate_failed_ordinal_from_stdout(
        stdout_path.read_text(encoding="utf-8", errors="replace"),
        completed_descriptor_path=descriptor_path,
    )

    completion_path = (
        Path(parent["campaign_root"]) / "matrix.lock" / "matrix.completed.json"
    ).resolve()
    if completion_path.exists():
        raise ValueError("S1 v5 parent matrix unexpectedly has a completion receipt")

    basis = {
        "schema_version": S1_STEP_RUNTIME_PROFILE_RECOVERY_SCHEMA,
        "reason": S1_STEP_RUNTIME_RECOVERY_REASON,
        "failure_signature": parent["failure_signature"],
        "failed_job_id": parent["failed_job_id"],
        "training_code_commit": training_commit,
        "profile_code_commit": profile_commit,
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_sha256": binding["manifest_sha256"],
        "protocol_fingerprint": binding["protocol_fingerprint"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "test_open_certificate_sha256": parent["test_open_certificate_sha256"],
        "legacy_unbound_test_resolution": parent["legacy_unbound_test_resolution"],
        "legacy_unbound_test_seed": parent["legacy_unbound_test_seed"],
        "legacy_unbound_test_evidence_path": parent[
            "legacy_unbound_test_evidence_path"
        ],
        "legacy_unbound_test_evidence_file_sha256": parent[
            "legacy_unbound_test_evidence_file_sha256"
        ],
        "legacy_unbound_test_evidence_sha256": parent[
            "legacy_unbound_test_evidence_sha256"
        ],
        "superseded_marker_path": parent["superseded_marker_path"],
        "superseded_marker_file_sha256": parent["superseded_marker_file_sha256"],
        "superseded_marker_sha256": parent["superseded_marker_sha256"],
        "failure_log_path": parent["failure_log_path"],
        "failure_log_sha256": parent["failure_log_sha256"],
        "expected_loader_exposure_count": parent["expected_loader_exposure_count"],
        "expected_physical_window_count": parent["expected_physical_window_count"],
        "expected_duplicate_physical_window_ids": parent[
            "expected_duplicate_physical_window_ids"
        ],
        "changed_files": changed_files,
        "parent_to_current_changed_files": parent_to_current_changed_files,
        "parent_runtime_implementation_sha256": parent_runtime_hashes,
        "runtime_implementation_sha256": runtime_hashes,
        "training_to_runtime_model_surface_changes": [],
        **tools_test_semantic_evidence,
        "repair_scope": "step_scoped_formal_test_runtime_only",
        "preserve_all_loader_exposures": True,
        "preserve_superseded_attempt": True,
        "reuse_valid_test_evidence": True,
        "preserve_recovery_chain": True,
        "superseded_recovery_certificate_path": str(parent_path),
        "superseded_recovery_certificate_file_sha256": sha256_file(parent_path),
        "superseded_recovery_certificate_sha256": parent["certificate_sha256"],
        "superseded_recovery_campaign_id": parent["campaign_id"],
        "superseded_recovery_profile_code_commit": parent["profile_code_commit"],
        "formal_test_runtime_mode": S1_STEP_SCOPED_TEST_RUNTIME_MODE,
        "formal_python_environment": formal_python_environment,
        "step_runtime_failure_signature": S1_STEP_RUNTIME_FAILURE_SIGNATURE,
        "step_runtime_failed_job_id": S1_STEP_RUNTIME_PARENT_JOB_ID,
        "step_runtime_failed_slurm_step_id": matrix_start["slurm_step_id"],
        "step_runtime_failed_gpu_uuid": matrix_start["step_gpu_uuid"],
        "step_runtime_failed_profile_order_ordinal": (S1_STEP_RUNTIME_FAILED_ORDINAL),
        "step_runtime_failed_resolution": S1_STEP_RUNTIME_FAILED_RESOLUTION,
        "step_runtime_failed_seed": S1_STEP_RUNTIME_FAILED_SEED,
        "step_runtime_matrix_submission_path": str(submission_path),
        "step_runtime_matrix_submission_file_sha256": sha256_file(submission_path),
        "step_runtime_matrix_submission_sha256": submission["receipt_sha256"],
        "step_runtime_matrix_start_path": str(matrix_start_path),
        "step_runtime_matrix_start_file_sha256": sha256_file(matrix_start_path),
        "step_runtime_matrix_sha256": matrix_start["matrix_sha256"],
        "step_runtime_matrix_stdout_path": str(stdout_path),
        "step_runtime_matrix_stdout_sha256": sha256_file(stdout_path),
        "step_runtime_matrix_stderr_path": str(stderr_path),
        "step_runtime_matrix_stderr_sha256": sha256_file(stderr_path),
        "step_runtime_completed_descriptor_count": 1,
        "step_runtime_completed_descriptor_path": str(descriptor_path),
        "step_runtime_completed_descriptor_file_sha256": sha256_file(descriptor_path),
        "step_runtime_completed_descriptor_sha256": descriptor["descriptor_sha256"],
        "step_runtime_completed_descriptor_record": descriptor_record,
        "step_runtime_completion_receipt_path": str(completion_path),
        "step_runtime_completion_receipt_absent": True,
        "power_sampler_backend": parent["power_sampler_backend"],
        "trace_publication_mode": parent["trace_publication_mode"],
        "trace_io_inside_sampling_loop": parent["trace_io_inside_sampling_loop"],
        "power_target_interval_ms": parent["power_target_interval_ms"],
        "power_max_gap_limit_ms": parent["power_max_gap_limit_ms"],
        "allocated_cpu_count": parent["allocated_cpu_count"],
        "detector_cpu_count": parent["detector_cpu_count"],
        "sidecar_cpu_count": parent["sidecar_cpu_count"],
        "requires_long_no_open_gate": parent["requires_long_no_open_gate"],
        "sidecar_gate_relative_path": parent["sidecar_gate_relative_path"],
    }
    campaign_id = canonical_sha256(basis)[:16]
    campaign_root = (
        Path(binding["canonical_experiment_root"]) / "profile_campaigns" / campaign_id
    ).resolve()
    certificate = {
        **basis,
        "campaign_id": campaign_id,
        "campaign_root": str(campaign_root),
    }
    certificate["certificate_sha256"] = canonical_sha256(certificate)
    output_path = _certificate_output_path(certificate)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != certificate:
            raise FileExistsError("S1 v5 recovery campaign identity collision")
    else:
        atomic_publish_json(output_path, certificate)
    return output_path, certificate


def build_profile_recovery_certificate(
    *,
    binding: Mapping[str, Any],
    failed_marker_path: str | Path | None,
    failure_log_path: str | Path | None,
    failed_job_id: str,
    expected_exposure_count: int | None,
    expected_physical_window_count: int | None,
    expected_duplicate_physical_window_ids: Sequence[str] | None,
    superseded_recovery_certificate_path: str | Path | None = None,
    power_diagnostic_path: str | Path | None = None,
    matrix_start_receipt_path: str | Path | None = None,
    matrix_submission_receipt_path: str | Path | None = None,
    matrix_stdout_path: str | Path | None = None,
    matrix_stderr_path: str | Path | None = None,
    completed_descriptor_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    if (
        superseded_recovery_certificate_path is not None
        and power_diagnostic_path is None
    ):
        parent_path = Path(superseded_recovery_certificate_path).resolve()
        if not parent_path.is_file():
            raise FileNotFoundError(parent_path)
        parent_reason = json.loads(parent_path.read_text(encoding="utf-8")).get(
            "reason"
        )
        if parent_reason == S1_BUFFERED_SIDECAR_RECOVERY_REASON:
            required_v5 = {
                "matrix_start_receipt_path": matrix_start_receipt_path,
                "matrix_submission_receipt_path": matrix_submission_receipt_path,
                "matrix_stdout_path": matrix_stdout_path,
                "matrix_stderr_path": matrix_stderr_path,
                "completed_descriptor_path": completed_descriptor_path,
            }
            missing = sorted(key for key, value in required_v5.items() if value is None)
            if missing:
                raise ValueError(
                    "S1 v5 recovery is missing required evidence: " + ", ".join(missing)
                )
            return build_step_runtime_profile_recovery_certificate(
                binding=binding,
                superseded_recovery_certificate_path=parent_path,
                matrix_start_receipt_path=matrix_start_receipt_path,
                matrix_submission_receipt_path=matrix_submission_receipt_path,
                matrix_stdout_path=matrix_stdout_path,
                matrix_stderr_path=matrix_stderr_path,
                completed_descriptor_path=completed_descriptor_path,
                failed_job_id=failed_job_id,
            )
        legacy_args = {
            "failed_marker_path": failed_marker_path,
            "failure_log_path": failure_log_path,
            "expected_exposure_count": expected_exposure_count,
            "expected_physical_window_count": expected_physical_window_count,
            "expected_duplicate_physical_window_ids": (
                expected_duplicate_physical_window_ids
            ),
        }
        missing = sorted(key for key, value in legacy_args.items() if value is None)
        if missing:
            raise ValueError(
                "S1 legacy recovery is missing required evidence: " + ", ".join(missing)
            )
        builder = (
            build_buffered_sidecar_profile_recovery_certificate
            if parent_reason == S1_SIDECAR_RECOVERY_REASON
            else build_sidecar_profile_recovery_certificate
        )
        return builder(
            binding=binding,
            failed_marker_path=failed_marker_path,
            failure_log_path=failure_log_path,
            failed_job_id=failed_job_id,
            expected_exposure_count=expected_exposure_count,
            expected_physical_window_count=expected_physical_window_count,
            expected_duplicate_physical_window_ids=(
                expected_duplicate_physical_window_ids
            ),
            superseded_recovery_certificate_path=parent_path,
        )
    if (
        failed_marker_path is None
        or failure_log_path is None
        or expected_exposure_count is None
        or expected_physical_window_count is None
        or expected_duplicate_physical_window_ids is None
    ):
        raise ValueError("S1 v1/v2 recovery requires the legacy failure evidence")
    profile_commit = current_git_commit()
    require_clean_profile_checkout(expected_commit=profile_commit)
    training_commit = str(binding["code_commit"]).lower()
    chained = superseded_recovery_certificate_path is not None
    if chained != (power_diagnostic_path is not None):
        raise ValueError(
            "S1 chained recovery requires both parent certificate and power diagnostic"
        )
    required_paths = (
        _REQUIRED_REPAIR_PATHS_CHAINED if chained else _REQUIRED_REPAIR_PATHS_V1
    )
    changed_files = _changed_files(
        training_commit, profile_commit, required_paths=required_paths
    )

    failed_marker_path = Path(failed_marker_path).resolve()
    failure_log_path = Path(failure_log_path).resolve()
    if not failed_marker_path.is_file() or not failure_log_path.is_file():
        raise FileNotFoundError(
            "S1 recovery requires the failed marker and failure log"
        )
    parent = None
    diagnostic = None
    marker_schema = (
        S1_CHAINED_PROFILE_MARKER_SCHEMA
        if chained
        else S1_SUPERSEDED_PROFILE_MARKER_SCHEMA
    )
    failure_signature = (
        S1_POWER_FAILURE_SIGNATURE if chained else S1_PROFILE_FAILURE_SIGNATURE
    )
    failed_marker = _validate_superseded_marker(
        failed_marker_path, expected_schema=marker_schema
    )
    if failure_signature not in failure_log_path.read_text(
        encoding="utf-8", errors="replace"
    ):
        raise ValueError(
            "S1 recovery failure log does not contain the frozen signature"
        )
    if chained:
        parent_path = Path(superseded_recovery_certificate_path).resolve()
        if not parent_path.is_file():
            raise FileNotFoundError(parent_path)
        parent = validate_profile_recovery_certificate(
            json.loads(parent_path.read_text(encoding="utf-8")),
            binding=binding,
            verify_checkout=False,
        )
        if parent.get("reason") != S1_PROFILE_RECOVERY_REASON:
            raise ValueError("S1 chained recovery requires the identity-repair parent")
        diagnostic_path = Path(power_diagnostic_path).resolve()
        if not diagnostic_path.is_file():
            raise FileNotFoundError(diagnostic_path)
        diagnostic = _validate_power_diagnostic(diagnostic_path)
    first_profile_cell = build_s1_profile_order()[0]
    expected_marker = {
        "resolution": int(first_profile_cell["resolution"]),
        "seed": int(first_profile_cell["seed"]),
        "code_commit": training_commit,
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_sha256": binding["manifest_sha256"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "profile_order_ordinal": 0,
    }
    if chained:
        expected_marker.update(
            {
                "profile_code_commit": parent["profile_code_commit"],
                "profile_recovery_certificate_sha256": parent["certificate_sha256"],
                "profile_recovery_campaign_id": parent["campaign_id"],
            }
        )
    for key, expected in expected_marker.items():
        if failed_marker.get(key) != expected:
            raise ValueError(f"superseded S1 profile marker {key} mismatch")

    exposure_count = int(expected_exposure_count)
    physical_count = int(expected_physical_window_count)
    duplicates = sorted(set(map(str, expected_duplicate_physical_window_ids)))
    if exposure_count <= 0 or physical_count <= 0 or physical_count >= exposure_count:
        raise ValueError("S1 recovery requires a real duplicate exposure topology")
    if exposure_count - physical_count != len(duplicates) or not duplicates:
        raise ValueError(
            "S1 recovery duplicate identities do not explain the exposure surplus"
        )
    if not str(failed_job_id).isdigit():
        raise ValueError("S1 recovery requires a numeric failed Slurm job id")

    original = parent or {
        "failed_job_id": str(failed_job_id),
        "failure_signature": S1_PROFILE_FAILURE_SIGNATURE,
        "test_open_certificate_sha256": failed_marker["test_open_certificate_sha256"],
        "superseded_marker_path": str(failed_marker_path),
        "superseded_marker_file_sha256": sha256_file(failed_marker_path),
        "superseded_marker_sha256": failed_marker["marker_sha256"],
        "failure_log_path": str(failure_log_path),
        "failure_log_sha256": sha256_file(failure_log_path),
    }
    basis = {
        "schema_version": (
            S1_CHAINED_PROFILE_RECOVERY_SCHEMA
            if chained
            else S1_PROFILE_RECOVERY_SCHEMA
        ),
        "reason": (
            S1_CHAINED_RECOVERY_REASON if chained else S1_PROFILE_RECOVERY_REASON
        ),
        "failure_signature": original["failure_signature"],
        "failed_job_id": original["failed_job_id"],
        "training_code_commit": training_commit,
        "profile_code_commit": profile_commit,
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_sha256": binding["manifest_sha256"],
        "protocol_fingerprint": binding["protocol_fingerprint"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "test_open_certificate_sha256": original["test_open_certificate_sha256"],
        "superseded_marker_path": original["superseded_marker_path"],
        "superseded_marker_file_sha256": original["superseded_marker_file_sha256"],
        "superseded_marker_sha256": original["superseded_marker_sha256"],
        "failure_log_path": original["failure_log_path"],
        "failure_log_sha256": original["failure_log_sha256"],
        "expected_loader_exposure_count": exposure_count,
        "expected_physical_window_count": physical_count,
        "expected_duplicate_physical_window_ids": duplicates,
        "changed_files": changed_files,
        "repair_scope": (
            "profile_identity_power_sampling_and_postprocessing_only"
            if chained
            else "profile_identity_and_postprocessing_only"
        ),
        "preserve_all_loader_exposures": True,
        "preserve_superseded_attempt": True,
        "reuse_valid_test_evidence": True,
    }
    if chained:
        parent_path = Path(superseded_recovery_certificate_path).resolve()
        diagnostic_path = Path(power_diagnostic_path).resolve()
        basis.update(
            {
                "preserve_recovery_chain": True,
                "superseded_recovery_certificate_path": str(parent_path),
                "superseded_recovery_certificate_file_sha256": sha256_file(parent_path),
                "superseded_recovery_certificate_sha256": parent["certificate_sha256"],
                "superseded_recovery_campaign_id": parent["campaign_id"],
                "superseded_recovery_profile_code_commit": parent[
                    "profile_code_commit"
                ],
                "power_failure_signature": S1_POWER_FAILURE_SIGNATURE,
                "power_failed_job_id": str(failed_job_id),
                "power_failure_marker_path": str(failed_marker_path),
                "power_failure_marker_file_sha256": sha256_file(failed_marker_path),
                "power_failure_marker_sha256": failed_marker["marker_sha256"],
                "power_failure_log_path": str(failure_log_path),
                "power_failure_log_sha256": sha256_file(failure_log_path),
                "power_diagnostic_path": str(diagnostic_path),
                "power_diagnostic_file_sha256": sha256_file(diagnostic_path),
                "power_diagnostic_sha256": diagnostic["diagnostic_sha256"],
                "power_diagnostic_job_id": diagnostic["slurm_job_id"],
                "power_diagnostic_code_commit": diagnostic["code_commit"],
                "power_sampler_backend": "nvml-persistent-poll-v1",
            }
        )
    campaign_id = canonical_sha256(basis)[:16]
    campaign_root = (
        Path(binding["canonical_experiment_root"]) / "profile_campaigns" / campaign_id
    ).resolve()
    certificate = {
        **basis,
        "campaign_id": campaign_id,
        "campaign_root": str(campaign_root),
    }
    certificate["certificate_sha256"] = canonical_sha256(certificate)
    output_path = _certificate_output_path(certificate)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != certificate:
            raise FileExistsError("S1 profile recovery campaign identity collision")
    else:
        atomic_publish_json(output_path, certificate)
    return output_path, certificate


def _validate_step_runtime_profile_recovery_certificate(
    checked: dict[str, Any],
    *,
    binding: Mapping[str, Any],
    verify_checkout: bool,
) -> dict[str, Any]:
    if checked.get("schema_version") != S1_STEP_RUNTIME_PROFILE_RECOVERY_SCHEMA:
        raise ValueError("S1 v5 recovery certificate schema/reason mismatch")
    if str(binding["code_commit"]).lower() != S1_STEP_RUNTIME_TRAINING_COMMIT:
        raise ValueError(
            "S1 v5 is frozen to training/model commit "
            f"{S1_STEP_RUNTIME_TRAINING_COMMIT}"
        )
    expected_static = {
        "training_code_commit": str(binding["code_commit"]).lower(),
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_sha256": binding["manifest_sha256"],
        "protocol_fingerprint": binding["protocol_fingerprint"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "repair_scope": "step_scoped_formal_test_runtime_only",
        "preserve_all_loader_exposures": True,
        "preserve_superseded_attempt": True,
        "reuse_valid_test_evidence": True,
        "preserve_recovery_chain": True,
        "formal_test_runtime_mode": S1_STEP_SCOPED_TEST_RUNTIME_MODE,
        "formal_python_environment": {
            "python_no_user_site": "1",
            "site_enable_user_site": False,
            "numpy_version": S1_FORMAL_NUMPY_VERSION,
            "numpy_path": S1_FORMAL_NUMPY_PATH,
        },
        "step_runtime_failure_signature": S1_STEP_RUNTIME_FAILURE_SIGNATURE,
        "step_runtime_failed_job_id": S1_STEP_RUNTIME_PARENT_JOB_ID,
        "step_runtime_failed_profile_order_ordinal": (S1_STEP_RUNTIME_FAILED_ORDINAL),
        "step_runtime_failed_resolution": S1_STEP_RUNTIME_FAILED_RESOLUTION,
        "step_runtime_failed_seed": S1_STEP_RUNTIME_FAILED_SEED,
        "step_runtime_completed_descriptor_count": 1,
        "step_runtime_completion_receipt_absent": True,
        "power_sampler_backend": S1_SIDECAR_POWER_BACKEND,
        "trace_publication_mode": S1_BUFFERED_TRACE_PUBLICATION_MODE,
        "trace_io_inside_sampling_loop": False,
        "power_target_interval_ms": 20,
        "power_max_gap_limit_ms": 100.0,
        "allocated_cpu_count": 5,
        "detector_cpu_count": 4,
        "sidecar_cpu_count": 1,
        "requires_long_no_open_gate": True,
        "sidecar_gate_relative_path": "sidecar_gate.json",
        "training_to_runtime_model_surface_changes": [],
    }
    for key, value in expected_static.items():
        if checked.get(key) != value:
            raise ValueError(f"S1 v5 recovery certificate {key} mismatch")

    campaign_basis = {
        key: value
        for key, value in checked.items()
        if key not in {"campaign_id", "campaign_root", "certificate_sha256"}
    }
    expected_campaign_id = canonical_sha256(campaign_basis)[:16]
    expected_campaign_root = (
        Path(binding["canonical_experiment_root"])
        / "profile_campaigns"
        / expected_campaign_id
    ).resolve()
    if (
        checked.get("campaign_id") != expected_campaign_id
        or Path(checked.get("campaign_root", "")).resolve() != expected_campaign_root
    ):
        raise ValueError("S1 v5 recovery campaign identity mismatch")

    parent_path = Path(checked["superseded_recovery_certificate_path"]).resolve()
    if not parent_path.is_file() or sha256_file(parent_path) != checked.get(
        "superseded_recovery_certificate_file_sha256"
    ):
        raise ValueError("S1 v5 parent recovery certificate file mismatch")
    parent = _load_parent_recovery_certificate(parent_path, binding=binding)
    if (
        parent.get("reason") != S1_BUFFERED_SIDECAR_RECOVERY_REASON
        or parent["certificate_sha256"]
        != checked.get("superseded_recovery_certificate_sha256")
        or parent["campaign_id"] != checked.get("superseded_recovery_campaign_id")
        or parent["profile_code_commit"]
        != checked.get("superseded_recovery_profile_code_commit")
    ):
        raise ValueError("S1 v5 parent recovery identity mismatch")
    parent_preserved = {
        "failure_signature": parent["failure_signature"],
        "failed_job_id": parent["failed_job_id"],
        "test_open_certificate_sha256": parent["test_open_certificate_sha256"],
        "legacy_unbound_test_resolution": parent["legacy_unbound_test_resolution"],
        "legacy_unbound_test_seed": parent["legacy_unbound_test_seed"],
        "legacy_unbound_test_evidence_path": parent[
            "legacy_unbound_test_evidence_path"
        ],
        "legacy_unbound_test_evidence_file_sha256": parent[
            "legacy_unbound_test_evidence_file_sha256"
        ],
        "legacy_unbound_test_evidence_sha256": parent[
            "legacy_unbound_test_evidence_sha256"
        ],
        "superseded_marker_path": parent["superseded_marker_path"],
        "superseded_marker_file_sha256": parent["superseded_marker_file_sha256"],
        "superseded_marker_sha256": parent["superseded_marker_sha256"],
        "failure_log_path": parent["failure_log_path"],
        "failure_log_sha256": parent["failure_log_sha256"],
        "expected_loader_exposure_count": parent["expected_loader_exposure_count"],
        "expected_physical_window_count": parent["expected_physical_window_count"],
        "expected_duplicate_physical_window_ids": parent[
            "expected_duplicate_physical_window_ids"
        ],
    }
    for key, value in parent_preserved.items():
        if checked.get(key) != value:
            raise ValueError(f"S1 v5 failed to preserve parent field {key}")

    changed_files = checked.get("changed_files")
    if not isinstance(changed_files, list):
        raise ValueError("S1 v5 recovery has no audited Git diff")
    changed_paths: set[str] = set()
    for row in changed_files:
        if not isinstance(row, Mapping):
            raise ValueError("S1 v5 recovery Git diff row is invalid")
        path = str(row.get("path", ""))
        if (
            row.get("status") not in {"A", "M"}
            or (
                path not in _ALLOWED_EXACT_PATHS_STEP_RUNTIME
                and not path.startswith(_ALLOWED_PREFIXES)
            )
            or len(str(row.get("file_sha256", ""))) != 64
        ):
            raise ValueError("S1 v5 recovery Git diff is outside audited scope")
        changed_paths.add(path)
    if not _REQUIRED_REPAIR_PATHS_STEP_RUNTIME.issubset(changed_paths):
        raise ValueError("S1 v5 recovery Git diff omits a required repair path")
    if any(path.startswith(_MODEL_SURFACE_PREFIXES) for path in changed_paths):
        raise ValueError("S1 v5 recovery Git diff changes model/config surfaces")

    parent_delta = _changed_files_between_commits(
        parent["profile_code_commit"],
        checked["profile_code_commit"],
        required_paths=_REQUIRED_PARENT_TO_STEP_RUNTIME_PATHS,
        allowed_exact_paths=_PARENT_TO_STEP_RUNTIME_PATHS,
    )
    parent_hashes = {
        path: _git_file_sha256(parent["profile_code_commit"], path)
        for path in sorted(_REQUIRED_STEP_RUNTIME_CODE_PATHS)
    }
    runtime_hashes = {
        path: _git_file_sha256(checked["profile_code_commit"], path)
        for path in sorted(_REQUIRED_STEP_RUNTIME_CODE_PATHS)
    }
    if (
        checked.get("parent_to_current_changed_files") != parent_delta
        or checked.get("parent_runtime_implementation_sha256") != parent_hashes
        or checked.get("runtime_implementation_sha256") != runtime_hashes
        or any(parent_hashes[path] == runtime_hashes[path] for path in parent_hashes)
    ):
        raise ValueError("S1 v5 runtime implementation binding mismatch")
    _assert_no_model_surface_changes(
        checked["training_code_commit"], checked["profile_code_commit"]
    )
    _assert_no_model_surface_changes(
        parent["profile_code_commit"], checked["profile_code_commit"]
    )
    tools_test_semantic_evidence = _tools_test_semantic_evidence_between_commits(
        checked["training_code_commit"],
        checked["profile_code_commit"],
    )
    for key, value in tools_test_semantic_evidence.items():
        if checked.get(key) != value:
            raise ValueError(f"S1 v5 tools/test.py semantic evidence {key} mismatch")

    matrix_start_path = Path(checked["step_runtime_matrix_start_path"]).resolve()
    if (
        matrix_start_path
        != (
            Path(parent["campaign_root"]) / "matrix.lock" / "matrix.started.json"
        ).resolve()
        or not matrix_start_path.is_file()
        or sha256_file(matrix_start_path)
        != checked.get("step_runtime_matrix_start_file_sha256")
    ):
        raise ValueError("S1 v5 matrix start receipt file mismatch")
    matrix_start = _validate_v3_matrix_start_receipt(
        matrix_start_path,
        parent_recovery_path=parent_path,
    )
    matrix_expected = {
        "slurm_job_id": S1_STEP_RUNTIME_PARENT_JOB_ID,
        "slurm_step_id": checked["step_runtime_failed_slurm_step_id"],
        "step_gpu_uuid": checked["step_runtime_failed_gpu_uuid"],
        "matrix_sha256": checked["step_runtime_matrix_sha256"],
        "profile_code_commit": parent["profile_code_commit"],
        "profile_recovery_certificate_sha256": parent["certificate_sha256"],
        "profile_recovery_campaign_id": parent["campaign_id"],
        "frozen_order": build_s1_profile_order(),
        "effective_step_memory_limit_mb": 96000,
        "slurm_cpus_per_task": 5,
    }
    for key, value in matrix_expected.items():
        if matrix_start.get(key) != value:
            raise ValueError(f"S1 v5 matrix start {key} mismatch")
    if (
        not str(matrix_start.get("slurm_step_id", "")).strip()
        or not str(matrix_start.get("step_gpu_uuid", "")).startswith("GPU-")
        or str(matrix_start.get("slurm_step_gpus", "")).count(",") != 0
        or str(matrix_start["slurm_step_gpus"])
        not in str(matrix_start.get("slurm_job_gpus", "")).split(",")
    ):
        raise ValueError("S1 v5 matrix start step/GPU identity is invalid")

    submission_path = Path(checked["step_runtime_matrix_submission_path"]).resolve()
    if not submission_path.is_file() or sha256_file(submission_path) != checked.get(
        "step_runtime_matrix_submission_file_sha256"
    ):
        raise ValueError("S1 v5 matrix submission file mismatch")
    submission = _validate_matrix_submission_receipt(
        submission_path,
        parent=parent,
        matrix_start=matrix_start,
        binding=binding,
    )
    if submission["receipt_sha256"] != checked.get(
        "step_runtime_matrix_submission_sha256"
    ):
        raise ValueError("S1 v5 matrix submission identity mismatch")

    stdout_path = Path(checked["step_runtime_matrix_stdout_path"]).resolve()
    stderr_path = Path(checked["step_runtime_matrix_stderr_path"]).resolve()
    expected_logs = {
        stdout_path: (
            Path(parent["campaign_root"])
            / "logs"
            / f"matrix-{S1_STEP_RUNTIME_PARENT_JOB_ID}.out"
        ).resolve(),
        stderr_path: (
            Path(parent["campaign_root"])
            / "logs"
            / f"matrix-{S1_STEP_RUNTIME_PARENT_JOB_ID}.err"
        ).resolve(),
    }
    for path, expected in expected_logs.items():
        hash_key = (
            "step_runtime_matrix_stdout_sha256"
            if path == stdout_path
            else "step_runtime_matrix_stderr_sha256"
        )
        if (
            path != expected
            or not path.is_file()
            or sha256_file(path) != checked.get(hash_key)
        ):
            raise ValueError(f"S1 v5 matrix log evidence mismatch: {path}")
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    if stderr_text.count(S1_STEP_RUNTIME_FAILURE_SIGNATURE) != 1:
        raise ValueError("S1 v5 stderr lacks the exact frozen failure signature")

    descriptor_path = Path(checked["step_runtime_completed_descriptor_path"]).resolve()
    descriptor_paths = sorted(
        (Path(parent["campaign_root"]) / "descriptors").glob("*.run.json")
    )
    if descriptor_paths != [descriptor_path]:
        raise ValueError("S1 v5 does not have exactly one completed descriptor")
    if not descriptor_path.is_file() or sha256_file(descriptor_path) != checked.get(
        "step_runtime_completed_descriptor_file_sha256"
    ):
        raise ValueError("S1 v5 completed descriptor file mismatch")
    descriptor, record = _validate_v5_completed_descriptor(
        descriptor_path,
        matrix_start=matrix_start,
        parent=parent,
        binding=binding,
    )
    if (
        descriptor["descriptor_sha256"]
        != checked.get("step_runtime_completed_descriptor_sha256")
        or record != checked.get("step_runtime_completed_descriptor_record")
        or int(record["profile_order_ordinal"]) != 0
        or int(record["resolution"]) != 256
        or int(record["seed"]) != 3408
    ):
        raise ValueError("S1 v5 completed descriptor identity mismatch")
    _validate_failed_ordinal_from_stdout(
        stdout_path.read_text(encoding="utf-8", errors="replace"),
        completed_descriptor_path=descriptor_path,
    )

    completion_path = Path(checked["step_runtime_completion_receipt_path"]).resolve()
    if (
        completion_path
        != (
            Path(parent["campaign_root"]) / "matrix.lock" / "matrix.completed.json"
        ).resolve()
        or completion_path.exists()
    ):
        raise ValueError("S1 v5 parent completion-receipt absence mismatch")
    if verify_checkout:
        if (
            _formal_python_environment_evidence()
            != checked["formal_python_environment"]
        ):
            raise ValueError("S1 v5 formal Python environment changed")
        require_clean_profile_checkout(expected_commit=checked["profile_code_commit"])
        rebuilt = _changed_files(
            checked["training_code_commit"],
            checked["profile_code_commit"],
            required_paths=_REQUIRED_REPAIR_PATHS_STEP_RUNTIME,
            allowed_exact_paths=_ALLOWED_EXACT_PATHS_STEP_RUNTIME,
        )
        if rebuilt != changed_files:
            raise ValueError("S1 v5 recovery Git diff changed after certification")
    return checked


def validate_profile_recovery_certificate(
    certificate: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
    verify_checkout: bool = True,
) -> dict[str, Any]:
    checked = json.loads(json.dumps(dict(certificate)))
    certificate_hash = checked.pop("certificate_sha256", None)
    if not certificate_hash or canonical_sha256(checked) != certificate_hash:
        raise ValueError("S1 profile recovery certificate self-hash mismatch")
    checked["certificate_sha256"] = certificate_hash
    if checked.get("reason") == S1_STEP_RUNTIME_RECOVERY_REASON:
        return _validate_step_runtime_profile_recovery_certificate(
            checked,
            binding=binding,
            verify_checkout=verify_checkout,
        )
    chained = checked.get("reason") == S1_CHAINED_RECOVERY_REASON
    sidecar = checked.get("reason") == S1_SIDECAR_RECOVERY_REASON
    buffered_sidecar = checked.get("reason") == S1_BUFFERED_SIDECAR_RECOVERY_REASON
    sidecar_family = sidecar or buffered_sidecar
    has_parent = chained or sidecar_family
    if checked.get("reason") not in {
        S1_PROFILE_RECOVERY_REASON,
        S1_CHAINED_RECOVERY_REASON,
        S1_SIDECAR_RECOVERY_REASON,
        S1_BUFFERED_SIDECAR_RECOVERY_REASON,
    }:
        raise ValueError("unsupported S1 profile recovery reason")
    expected_schema = (
        S1_BUFFERED_SIDECAR_PROFILE_RECOVERY_SCHEMA
        if buffered_sidecar
        else (
            S1_SIDECAR_PROFILE_RECOVERY_SCHEMA
            if sidecar
            else (
                S1_CHAINED_PROFILE_RECOVERY_SCHEMA
                if chained
                else S1_PROFILE_RECOVERY_SCHEMA
            )
        )
    )
    if checked.get("schema_version") != expected_schema:
        raise ValueError("S1 profile recovery certificate schema/reason mismatch")
    expected = {
        "failure_signature": S1_PROFILE_FAILURE_SIGNATURE,
        "training_code_commit": str(binding["code_commit"]).lower(),
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_sha256": binding["manifest_sha256"],
        "protocol_fingerprint": binding["protocol_fingerprint"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "repair_scope": (
            "buffered_sidecar_trace_publication_only"
            if buffered_sidecar
            else (
                "out_of_process_power_sidecar_and_failure_evidence_only"
                if sidecar
                else (
                    "profile_identity_power_sampling_and_postprocessing_only"
                    if chained
                    else "profile_identity_and_postprocessing_only"
                )
            )
        ),
        "preserve_all_loader_exposures": True,
        "preserve_superseded_attempt": True,
        "reuse_valid_test_evidence": True,
    }
    if sidecar_family:
        first_profile_cell = build_s1_profile_order()[0]
        legacy_test_evidence_path = _legacy_unbound_test_evidence_path(binding)
        expected.update(
            {
                "legacy_unbound_test_resolution": int(first_profile_cell["resolution"]),
                "legacy_unbound_test_seed": int(first_profile_cell["seed"]),
                "legacy_unbound_test_evidence_path": str(legacy_test_evidence_path),
            }
        )
    for key, value in expected.items():
        if checked.get(key) != value:
            raise ValueError(f"S1 profile recovery certificate {key} mismatch")
    if bool(checked.get("preserve_recovery_chain", False)) != has_parent:
        raise ValueError("S1 profile recovery chain-preservation flag mismatch")
    campaign_basis = {
        key: value
        for key, value in checked.items()
        if key not in {"campaign_id", "campaign_root", "certificate_sha256"}
    }
    expected_campaign_id = canonical_sha256(campaign_basis)[:16]
    expected_campaign_root = (
        Path(binding["canonical_experiment_root"])
        / "profile_campaigns"
        / expected_campaign_id
    ).resolve()
    if (
        checked.get("campaign_id") != expected_campaign_id
        or Path(checked.get("campaign_root", "")).resolve() != expected_campaign_root
    ):
        raise ValueError("S1 profile recovery campaign identity mismatch")

    failed_marker_path = Path(checked["superseded_marker_path"]).resolve()
    failure_log_path = Path(checked["failure_log_path"]).resolve()
    for path, key in (
        (failed_marker_path, "superseded_marker_file_sha256"),
        (failure_log_path, "failure_log_sha256"),
    ):
        if not path.is_file() or sha256_file(path) != checked[key]:
            raise ValueError(f"S1 profile recovery artifact mismatch: {path}")
    if sidecar_family:
        legacy_test_evidence_path = Path(
            checked["legacy_unbound_test_evidence_path"]
        ).resolve()
        if not legacy_test_evidence_path.is_file() or sha256_file(
            legacy_test_evidence_path
        ) != checked.get("legacy_unbound_test_evidence_file_sha256"):
            raise ValueError("S1 profile recovery legacy test-evidence file mismatch")
        legacy_test_evidence = json.loads(
            legacy_test_evidence_path.read_text(encoding="utf-8")
        )
        legacy_hash = legacy_test_evidence.pop("evidence_sha256", None)
        if (
            not legacy_hash
            or canonical_sha256(legacy_test_evidence) != legacy_hash
            or legacy_hash != checked.get("legacy_unbound_test_evidence_sha256")
        ):
            raise ValueError("S1 profile recovery legacy test-evidence hash mismatch")
    failed_marker = _validate_superseded_marker(failed_marker_path)
    if failed_marker["marker_sha256"] != checked["superseded_marker_sha256"]:
        raise ValueError("S1 profile recovery superseded-marker identity mismatch")
    first_profile_cell = build_s1_profile_order()[0]
    marker_expected = {
        "resolution": int(first_profile_cell["resolution"]),
        "seed": int(first_profile_cell["seed"]),
        "code_commit": str(binding["code_commit"]).lower(),
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_sha256": binding["manifest_sha256"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "profile_order_ordinal": 0,
        "test_open_certificate_sha256": checked["test_open_certificate_sha256"],
    }
    for key, value in marker_expected.items():
        if failed_marker.get(key) != value:
            raise ValueError(f"S1 profile recovery superseded marker {key} mismatch")
    if S1_PROFILE_FAILURE_SIGNATURE not in failure_log_path.read_text(
        encoding="utf-8", errors="replace"
    ):
        raise ValueError("S1 profile recovery failure signature disappeared")

    if chained:
        parent_path = Path(checked["superseded_recovery_certificate_path"]).resolve()
        power_marker_path = Path(checked["power_failure_marker_path"]).resolve()
        power_log_path = Path(checked["power_failure_log_path"]).resolve()
        diagnostic_path = Path(checked["power_diagnostic_path"]).resolve()
        for path, key in (
            (parent_path, "superseded_recovery_certificate_file_sha256"),
            (power_marker_path, "power_failure_marker_file_sha256"),
            (power_log_path, "power_failure_log_sha256"),
            (diagnostic_path, "power_diagnostic_file_sha256"),
        ):
            if not path.is_file() or sha256_file(path) != checked.get(key):
                raise ValueError(f"S1 chained recovery artifact mismatch: {path}")
        parent = validate_profile_recovery_certificate(
            json.loads(parent_path.read_text(encoding="utf-8")),
            binding=binding,
            verify_checkout=False,
        )
        if (
            parent.get("reason") != S1_PROFILE_RECOVERY_REASON
            or parent["certificate_sha256"]
            != checked.get("superseded_recovery_certificate_sha256")
            or parent["campaign_id"] != checked.get("superseded_recovery_campaign_id")
            or parent["profile_code_commit"]
            != checked.get("superseded_recovery_profile_code_commit")
        ):
            raise ValueError("S1 chained recovery parent identity mismatch")
        power_marker = _validate_superseded_marker(
            power_marker_path, expected_schema=S1_CHAINED_PROFILE_MARKER_SCHEMA
        )
        if (
            power_marker["marker_sha256"] != checked.get("power_failure_marker_sha256")
            or S1_POWER_FAILURE_SIGNATURE
            not in power_log_path.read_text(encoding="utf-8", errors="replace")
            or checked.get("power_failure_signature") != S1_POWER_FAILURE_SIGNATURE
            or str(checked.get("power_failed_job_id", "")).isdigit() is False
        ):
            raise ValueError("S1 chained recovery power-failure evidence mismatch")
        current_marker_expected = {
            "resolution": int(first_profile_cell["resolution"]),
            "seed": int(first_profile_cell["seed"]),
            "code_commit": str(binding["code_commit"]).lower(),
            "profile_code_commit": parent["profile_code_commit"],
            "experiment_namespace": binding["experiment_namespace"],
            "canonical_experiment_root": binding["canonical_experiment_root"],
            "manifest_sha256": binding["manifest_sha256"],
            "precheck_file_sha256": binding["precheck_file_sha256"],
            "precheck_sha256": binding["precheck_sha256"],
            "profile_order_ordinal": 0,
            "test_open_certificate_sha256": checked["test_open_certificate_sha256"],
            "profile_recovery_certificate_sha256": parent["certificate_sha256"],
            "profile_recovery_campaign_id": parent["campaign_id"],
        }
        for key, value in current_marker_expected.items():
            if power_marker.get(key) != value:
                raise ValueError(f"S1 chained recovery marker {key} mismatch")
        diagnostic = _validate_power_diagnostic(diagnostic_path)
        if (
            diagnostic["diagnostic_sha256"] != checked.get("power_diagnostic_sha256")
            or diagnostic["slurm_job_id"] != checked.get("power_diagnostic_job_id")
            or diagnostic["code_commit"] != checked.get("power_diagnostic_code_commit")
            or checked.get("power_sampler_backend") != "nvml-persistent-poll-v1"
        ):
            raise ValueError("S1 chained recovery power diagnostic mismatch")
    elif sidecar:
        parent_path = Path(checked["superseded_recovery_certificate_path"]).resolve()
        power_marker_path = Path(checked["sidecar_power_failure_marker_path"]).resolve()
        power_log_path = Path(checked["sidecar_power_failure_log_path"]).resolve()
        for path, key in (
            (parent_path, "superseded_recovery_certificate_file_sha256"),
            (power_marker_path, "sidecar_power_failure_marker_file_sha256"),
            (power_log_path, "sidecar_power_failure_log_sha256"),
        ):
            if not path.is_file() or sha256_file(path) != checked.get(key):
                raise ValueError(f"S1 sidecar recovery artifact mismatch: {path}")
        parent = validate_profile_recovery_certificate(
            json.loads(parent_path.read_text(encoding="utf-8")),
            binding=binding,
            verify_checkout=False,
        )
        if (
            parent.get("reason") != S1_CHAINED_RECOVERY_REASON
            or parent["certificate_sha256"]
            != checked.get("superseded_recovery_certificate_sha256")
            or parent["campaign_id"] != checked.get("superseded_recovery_campaign_id")
            or parent["profile_code_commit"]
            != checked.get("superseded_recovery_profile_code_commit")
        ):
            raise ValueError("S1 sidecar recovery parent identity mismatch")
        power_marker = _validate_superseded_marker(
            power_marker_path,
            expected_schema=S1_CHAINED_PROFILE_MARKER_SCHEMA,
        )
        current_marker_expected = {
            "resolution": int(first_profile_cell["resolution"]),
            "seed": int(first_profile_cell["seed"]),
            "code_commit": str(binding["code_commit"]).lower(),
            "profile_code_commit": parent["profile_code_commit"],
            "experiment_namespace": binding["experiment_namespace"],
            "canonical_experiment_root": binding["canonical_experiment_root"],
            "manifest_sha256": binding["manifest_sha256"],
            "precheck_file_sha256": binding["precheck_file_sha256"],
            "precheck_sha256": binding["precheck_sha256"],
            "profile_order_ordinal": 0,
            "test_open_certificate_sha256": checked["test_open_certificate_sha256"],
            "profile_recovery_certificate_sha256": parent["certificate_sha256"],
            "profile_recovery_campaign_id": parent["campaign_id"],
        }
        for key, value in current_marker_expected.items():
            if power_marker.get(key) != value:
                raise ValueError(f"S1 sidecar recovery marker {key} mismatch")
        if (
            power_marker["marker_sha256"]
            != checked.get("sidecar_power_failure_marker_sha256")
            or S1_POWER_FAILURE_SIGNATURE
            not in power_log_path.read_text(encoding="utf-8", errors="replace")
            or checked.get("sidecar_power_failure_signature")
            != S1_POWER_FAILURE_SIGNATURE
            or str(checked.get("sidecar_power_failed_job_id", "")).isdigit() is False
            or checked.get("power_sampler_backend") != S1_SIDECAR_POWER_BACKEND
            or int(checked.get("power_target_interval_ms", -1)) != 20
            or float(checked.get("power_max_gap_limit_ms", -1.0)) != 100.0
            or int(checked.get("allocated_cpu_count", -1)) != 5
            or int(checked.get("detector_cpu_count", -1)) != 4
            or int(checked.get("sidecar_cpu_count", -1)) != 1
            or checked.get("requires_long_no_open_gate") is not True
            or checked.get("sidecar_gate_relative_path") != "sidecar_gate.json"
        ):
            raise ValueError("S1 sidecar recovery contract mismatch")
    elif buffered_sidecar:
        parent_path = Path(checked["superseded_recovery_certificate_path"]).resolve()
        buffered_marker_path = Path(
            checked["buffered_sidecar_failure_marker_path"]
        ).resolve()
        buffered_log_path = Path(checked["buffered_sidecar_failure_log_path"]).resolve()
        attempt_report_path = Path(
            checked["buffered_sidecar_attempt_report_path"]
        ).resolve()
        attempt_trace_path = Path(
            checked["buffered_sidecar_attempt_trace_path"]
        ).resolve()
        parent_failure_path = Path(
            checked["buffered_sidecar_parent_failure_path"]
        ).resolve()
        matrix_start_path = Path(
            checked["buffered_sidecar_matrix_start_path"]
        ).resolve()
        for path, key in (
            (parent_path, "superseded_recovery_certificate_file_sha256"),
            (
                buffered_marker_path,
                "buffered_sidecar_failure_marker_file_sha256",
            ),
            (buffered_log_path, "buffered_sidecar_failure_log_sha256"),
            (
                attempt_report_path,
                "buffered_sidecar_attempt_report_file_sha256",
            ),
            (
                attempt_trace_path,
                "buffered_sidecar_attempt_trace_file_sha256",
            ),
            (
                parent_failure_path,
                "buffered_sidecar_parent_failure_file_sha256",
            ),
            (
                matrix_start_path,
                "buffered_sidecar_matrix_start_file_sha256",
            ),
        ):
            if not path.is_file() or sha256_file(path) != checked.get(key):
                raise ValueError(
                    f"S1 buffered-sidecar recovery artifact mismatch: {path}"
                )
        parent = validate_profile_recovery_certificate(
            json.loads(parent_path.read_text(encoding="utf-8")),
            binding=binding,
            verify_checkout=False,
        )
        if (
            parent.get("reason") != S1_SIDECAR_RECOVERY_REASON
            or parent["certificate_sha256"]
            != checked.get("superseded_recovery_certificate_sha256")
            or parent["campaign_id"] != checked.get("superseded_recovery_campaign_id")
            or parent["profile_code_commit"]
            != checked.get("superseded_recovery_profile_code_commit")
        ):
            raise ValueError("S1 buffered-sidecar recovery parent identity mismatch")
        parent_to_current_changed_files = _changed_files_between_commits(
            parent["profile_code_commit"],
            checked["profile_code_commit"],
            required_paths={"tools/bata/spatial_zoom_s1_power.py"},
        )
        parent_sampling_implementation_sha256 = _git_file_sha256(
            parent["profile_code_commit"],
            "tools/bata/spatial_zoom_s1_power.py",
        )
        sampling_implementation_sha256 = _git_file_sha256(
            checked["profile_code_commit"],
            "tools/bata/spatial_zoom_s1_power.py",
        )
        if (
            checked.get("parent_to_current_changed_files")
            != parent_to_current_changed_files
            or checked.get("parent_sampling_implementation_sha256")
            != parent_sampling_implementation_sha256
            or checked.get("sampling_implementation_sha256")
            != sampling_implementation_sha256
            or parent_sampling_implementation_sha256 == sampling_implementation_sha256
        ):
            raise ValueError(
                "S1 buffered-sidecar sampling implementation binding mismatch"
            )

        buffered_marker = _validate_superseded_marker(
            buffered_marker_path,
            expected_schema=S1_BUFFERED_SIDECAR_PROFILE_MARKER_SCHEMA,
        )
        current_marker_expected = {
            "resolution": int(first_profile_cell["resolution"]),
            "seed": int(first_profile_cell["seed"]),
            "code_commit": str(binding["code_commit"]).lower(),
            "profile_code_commit": parent["profile_code_commit"],
            "experiment_namespace": binding["experiment_namespace"],
            "canonical_experiment_root": binding["canonical_experiment_root"],
            "manifest_sha256": binding["manifest_sha256"],
            "precheck_file_sha256": binding["precheck_file_sha256"],
            "precheck_sha256": binding["precheck_sha256"],
            "profile_order_ordinal": 0,
            "test_open_certificate_sha256": checked["test_open_certificate_sha256"],
            "profile_recovery_certificate_sha256": parent["certificate_sha256"],
            "profile_recovery_campaign_id": parent["campaign_id"],
            "power_sampler_backend": S1_SIDECAR_POWER_BACKEND,
            "gate_only": False,
            "slurm_job_id": checked["buffered_sidecar_failed_job_id"],
            "slurm_step_id": checked["buffered_sidecar_failed_slurm_step_id"],
            "step_gpu_uuid": checked["buffered_sidecar_failed_gpu_uuid"],
        }
        for key, value in current_marker_expected.items():
            if buffered_marker.get(key) != value:
                raise ValueError(f"S1 buffered-sidecar marker {key} mismatch")
        if (
            buffered_marker["marker_sha256"]
            != checked.get("buffered_sidecar_failure_marker_sha256")
            or S1_BUFFERED_SIDECAR_FAILURE_SIGNATURE
            not in buffered_log_path.read_text(encoding="utf-8", errors="replace")
            or checked.get("buffered_sidecar_failure_signature")
            != S1_BUFFERED_SIDECAR_FAILURE_SIGNATURE
            or str(checked.get("buffered_sidecar_failed_job_id", "")).isdigit() is False
        ):
            raise ValueError("S1 buffered-sidecar failure marker/log identity mismatch")

        attempt = validate_nvml_sidecar_cadence_failure(
            attempt_report_path,
            attempt_trace_path,
            expected_uuid=str(buffered_marker["step_gpu_uuid"]),
        )
        if (
            attempt.get("attempt_sha256")
            != checked.get("buffered_sidecar_attempt_sha256")
            or attempt.get("backend") != S1_SIDECAR_POWER_BACKEND
            or float(attempt.get("cadence", {}).get("max_gap_limit_ms", -1.0))
            != float(parent["power_max_gap_limit_ms"])
        ):
            raise ValueError(
                "S1 buffered-sidecar attempt does not prove cadence failure"
            )

        parent_failure = json.loads(parent_failure_path.read_text(encoding="utf-8"))
        parent_failure_hash = parent_failure.pop("parent_failure_sha256", None)
        if (
            not parent_failure_hash
            or canonical_sha256(parent_failure) != parent_failure_hash
            or parent_failure_hash
            != checked.get("buffered_sidecar_parent_failure_sha256")
            or parent_failure.get("schema_version")
            != "spatial_zoom_s1_profile_parent_failure_v1"
            or parent_failure.get("status") != "FAIL"
            or parent_failure.get("paper_claim_allowed") is not False
            or parent_failure.get("power_attempt_sha256") != attempt["attempt_sha256"]
            or parent_failure.get("power_attempt_report_file_sha256")
            != sha256_file(attempt_report_path)
            or parent_failure.get("power_attempt_trace_file_sha256")
            != sha256_file(attempt_trace_path)
        ):
            raise ValueError("S1 buffered-sidecar parent-failure evidence mismatch")

        matrix_start = _validate_v3_matrix_start_receipt(
            matrix_start_path,
            parent_recovery_path=parent_path,
        )
        matrix_hash = matrix_start["matrix_sha256"]
        if (
            matrix_hash != checked.get("buffered_sidecar_matrix_sha256")
            or matrix_hash != buffered_marker.get("matrix_sha256")
            or matrix_start.get("slurm_job_id")
            != str(checked["buffered_sidecar_failed_job_id"])
            or matrix_start.get("slurm_step_id")
            != checked.get("buffered_sidecar_failed_slurm_step_id")
            or matrix_start.get("step_gpu_uuid")
            != checked.get("buffered_sidecar_failed_gpu_uuid")
            or matrix_start.get("profile_code_commit") != parent["profile_code_commit"]
            or matrix_start.get("profile_recovery_certificate_sha256")
            != parent["certificate_sha256"]
            or matrix_start.get("profile_recovery_campaign_id") != parent["campaign_id"]
            or matrix_start.get("frozen_order") != build_s1_profile_order()
            or buffered_marker.get("matrix_start_receipt_path")
            != str(matrix_start_path)
            or buffered_marker.get("matrix_start_receipt_file_sha256")
            != sha256_file(matrix_start_path)
        ):
            raise ValueError("S1 buffered-sidecar matrix-start identity mismatch")

        if (
            checked.get("power_sampler_backend") != S1_SIDECAR_POWER_BACKEND
            or checked.get("trace_publication_mode")
            != S1_BUFFERED_TRACE_PUBLICATION_MODE
            or checked.get("trace_io_inside_sampling_loop") is not False
            or int(checked.get("power_target_interval_ms", -1)) != 20
            or float(checked.get("power_max_gap_limit_ms", -1.0)) != 100.0
            or int(checked.get("allocated_cpu_count", -1)) != 5
            or int(checked.get("detector_cpu_count", -1)) != 4
            or int(checked.get("sidecar_cpu_count", -1)) != 1
            or checked.get("requires_long_no_open_gate") is not True
            or checked.get("sidecar_gate_relative_path") != "sidecar_gate.json"
        ):
            raise ValueError("S1 buffered-sidecar recovery contract mismatch")

    exposure_count = int(checked.get("expected_loader_exposure_count", -1))
    physical_count = int(checked.get("expected_physical_window_count", -1))
    duplicates = checked.get("expected_duplicate_physical_window_ids")
    if (
        not isinstance(duplicates, list)
        or sorted(set(map(str, duplicates))) != duplicates
        or exposure_count - physical_count != len(duplicates)
        or not duplicates
    ):
        raise ValueError("S1 profile recovery exposure topology is invalid")

    changed_files = checked.get("changed_files")
    if not isinstance(changed_files, list):
        raise ValueError("S1 profile recovery has no audited Git diff")
    changed_paths = set()
    for row in changed_files:
        if not isinstance(row, Mapping):
            raise ValueError("S1 profile recovery Git diff row is invalid")
        path = str(row.get("path", ""))
        if (
            row.get("status") not in {"A", "M"}
            or (
                path not in _ALLOWED_EXACT_PATHS
                and not path.startswith(_ALLOWED_PREFIXES)
            )
            or len(str(row.get("file_sha256", ""))) != 64
        ):
            raise ValueError(
                "S1 profile recovery Git diff is outside the audited scope"
            )
        changed_paths.add(path)
    required_paths = (
        _REQUIRED_REPAIR_PATHS_BUFFERED_SIDECAR
        if buffered_sidecar
        else (
            _REQUIRED_REPAIR_PATHS_SIDECAR
            if sidecar
            else (
                _REQUIRED_REPAIR_PATHS_CHAINED if chained else _REQUIRED_REPAIR_PATHS_V1
            )
        )
    )
    if not required_paths.issubset(changed_paths):
        raise ValueError("S1 profile recovery Git diff omits a required repair path")

    if verify_checkout:
        require_clean_profile_checkout(expected_commit=checked["profile_code_commit"])
        if (
            _changed_files(
                checked["training_code_commit"],
                checked["profile_code_commit"],
                required_paths=required_paths,
            )
            != checked["changed_files"]
        ):
            raise ValueError("S1 profile recovery Git diff changed after certification")
    return checked


def load_profile_recovery_certificate(
    path: str | Path,
    *,
    binding: Mapping[str, Any],
    verify_checkout: bool = True,
) -> dict[str, Any]:
    path = Path(path).resolve()
    checked = validate_profile_recovery_certificate(
        json.loads(path.read_text(encoding="utf-8")),
        binding=binding,
        verify_checkout=verify_checkout,
    )
    if path != _certificate_output_path(checked):
        raise ValueError("S1 profile recovery certificate is outside its campaign")
    raw_text = path.read_text(encoding="utf-8")
    canonical_text = json.dumps(json.loads(raw_text), indent=2, sort_keys=True) + "\n"
    if raw_text != canonical_text:
        raise ValueError("S1 profile recovery certificate is not canonical JSON")
    return checked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create one immutable S1 post-profile recovery campaign"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--failed-marker", type=Path)
    parser.add_argument("--failure-log", type=Path)
    parser.add_argument("--failed-job-id", required=True)
    parser.add_argument("--superseded-recovery-certificate", type=Path)
    parser.add_argument("--power-diagnostic", type=Path)
    parser.add_argument("--expected-exposure-count", type=int)
    parser.add_argument("--expected-physical-window-count", type=int)
    parser.add_argument("--expected-duplicate-physical-window-id", action="append")
    parser.add_argument("--matrix-start-receipt", type=Path)
    parser.add_argument("--matrix-submission-receipt", type=Path)
    parser.add_argument("--matrix-stdout", type=Path)
    parser.add_argument("--matrix-stderr", type=Path)
    parser.add_argument("--completed-descriptor", type=Path)
    args = parser.parse_args(argv)
    try:
        cfg = Config.fromfile(str(args.config.resolve()))
        binding = validate_bound_s1_training_config(cfg, seed=int(args.seed))
        path, certificate = build_profile_recovery_certificate(
            binding=binding,
            failed_marker_path=args.failed_marker,
            failure_log_path=args.failure_log,
            failed_job_id=args.failed_job_id,
            expected_exposure_count=args.expected_exposure_count,
            expected_physical_window_count=args.expected_physical_window_count,
            expected_duplicate_physical_window_ids=(
                args.expected_duplicate_physical_window_id
            ),
            superseded_recovery_certificate_path=(args.superseded_recovery_certificate),
            power_diagnostic_path=args.power_diagnostic,
            matrix_start_receipt_path=args.matrix_start_receipt,
            matrix_submission_receipt_path=args.matrix_submission_receipt,
            matrix_stdout_path=args.matrix_stdout,
            matrix_stderr_path=args.matrix_stderr,
            completed_descriptor_path=args.completed_descriptor,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)},
                indent=2,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "certificate_path": str(path),
                "campaign_id": certificate["campaign_id"],
                "certificate_sha256": certificate["certificate_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "S1_PROFILE_RECOVERY_SCHEMA",
    "S1_CHAINED_PROFILE_RECOVERY_SCHEMA",
    "S1_SIDECAR_PROFILE_RECOVERY_SCHEMA",
    "S1_BUFFERED_SIDECAR_PROFILE_RECOVERY_SCHEMA",
    "S1_STEP_RUNTIME_PROFILE_RECOVERY_SCHEMA",
    "S1_CHAINED_RECOVERY_REASON",
    "S1_SIDECAR_RECOVERY_REASON",
    "S1_BUFFERED_SIDECAR_RECOVERY_REASON",
    "S1_STEP_RUNTIME_RECOVERY_REASON",
    "S1_POWER_FAILURE_SIGNATURE",
    "S1_SIDECAR_POWER_BACKEND",
    "S1_BUFFERED_TRACE_PUBLICATION_MODE",
    "S1_BUFFERED_SIDECAR_FAILURE_SIGNATURE",
    "S1_STEP_RUNTIME_FAILURE_SIGNATURE",
    "S1_STEP_SCOPED_TEST_RUNTIME_MODE",
    "build_profile_recovery_certificate",
    "build_sidecar_profile_recovery_certificate",
    "build_buffered_sidecar_profile_recovery_certificate",
    "build_step_runtime_profile_recovery_certificate",
    "load_profile_recovery_certificate",
    "profile_campaign_prefix",
    "require_clean_profile_checkout",
    "validate_profile_recovery_certificate",
]
