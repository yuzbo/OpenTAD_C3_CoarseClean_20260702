import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "bata" / "actionformer_source_diff.py"
BUILDER_PATH = (
    ROOT / "tools" / "bata" / "build_actionformer_source_diff_attestation.py"
)
SPEC = importlib.util.spec_from_file_location("actionformer_source_diff", MODULE_PATH)
source_diff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source_diff)

BASE_REF = "refs/heads/main"
CANDIDATE_REF = "refs/heads/candidate"

CONFIG_LOADER = """\
import yaml

DEFAULTS = {
    "init_rand_seed": 1234567891,
    "train_split": ("validation",),
    "val_split": ("test",),
    "dataset": {
        "input_dim": 2048,
        "num_classes": 20,
        "feat_stride": 4,
        "num_frames": 16,
        "max_seq_len": 2304,
    },
    "loader": {"batch_size": 2},
    "model": {"backbone_type": "convTransformer"},
    "train_cfg": {"center_sample": "radius"},
    "test_cfg": {"nms_method": "soft", "max_seg_num": 200},
    "opt": {"type": "AdamW", "learning_rate": 0.0001, "epochs": 30},
}

def _merge(src, dst):
    for key, value in src.items():
        if key in dst:
            if isinstance(value, dict):
                _merge(value, dst[key])
        else:
            dst[key] = value

def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    _merge(defaults, config)
    config["model"]["input_dim"] = config["dataset"]["input_dim"]
    config["model"]["num_classes"] = config["dataset"]["num_classes"]
    config["model"]["max_seq_len"] = config["dataset"]["max_seq_len"]
    config["model"]["train_cfg"] = config["train_cfg"]
    config["model"]["test_cfg"] = config["test_cfg"]
    return config
"""

BASE_CONFIG = """\
dataset:
  input_dim: 2048
  num_classes: 20
  feat_stride: 4
  num_frames: 16
  max_seq_len: 2304
"""

CANDIDATE_CONFIG = BASE_CONFIG + """\
  selection_budget: 384
  selection_coordinate_mode: uniform_rank
  selection_policy: deterministic_random_max_budget
  selection_seed: 1234567891
"""

NATIVE_SPARSE_CONFIG = BASE_CONFIG + """\
model:
  sparse_head:
    enabled: true
    budget: 384
    policy: stratified_uniform
    hash_seed: 1234567891
    training_loss_support: selected_native_grid_queries
"""


def _git(repo, *arguments):
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit(repo, message):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _selection_repo(tmp_path):
    upstream = tmp_path / "upstream.git"
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", str(upstream))
    _git(tmp_path, "init", "--bare", str(origin))
    repo = tmp_path / "candidate"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Unit Test")
    _git(repo, "config", "user.email", "unit@test.invalid")
    _git(repo, "remote", "add", "upstream", upstream.resolve().as_uri())
    _git(repo, "remote", "add", "origin", origin.resolve().as_uri())
    _write(repo / "README.md", "official readme\n")
    _write(repo / "eval.py", "official evaluator\n")
    _write(repo / "train.py", "official trainer\n")
    _write(repo / "configs" / "thumos_i3d.yaml", BASE_CONFIG)
    _write(repo / "libs" / "core" / "config.py", CONFIG_LOADER)
    _write(repo / "libs" / "datasets" / "thumos14.py", "SELECTOR = None\n")
    base = _commit(repo, "base")
    _git(repo, "push", "upstream", f"{base}:{BASE_REF}")

    _write(
        repo / "configs" / "thumos_i3d_random_k384.yaml",
        CANDIDATE_CONFIG,
    )
    _write(
        repo / "libs" / "datasets" / "deterministic_selection.py",
        "def select(values):\n    return values[:384]\n",
    )
    _write(
        repo / "libs" / "datasets" / "thumos14.py",
        "from .deterministic_selection import select\n",
    )
    candidate = _commit(repo, "selection")
    _git(repo, "push", "origin", f"{candidate}:{CANDIDATE_REF}")
    return repo, base, candidate


def _publish_candidate(repo, candidate):
    _git(repo, "push", "--force", "origin", f"{candidate}:{CANDIDATE_REF}")


def _native_sparse_repo(tmp_path):
    upstream = tmp_path / "native-upstream.git"
    origin = tmp_path / "native-origin.git"
    _git(tmp_path, "init", "--bare", str(upstream))
    _git(tmp_path, "init", "--bare", str(origin))
    repo = tmp_path / "native-candidate"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Unit Test")
    _git(repo, "config", "user.email", "unit@test.invalid")
    _git(repo, "remote", "add", "upstream", upstream.resolve().as_uri())
    _git(repo, "remote", "add", "origin", origin.resolve().as_uri())
    _write(repo / "README.md", "official readme\n")
    _write(repo / "eval.py", "official evaluator\n")
    _write(repo / "train.py", "official trainer\n")
    _write(repo / "configs" / "thumos_i3d.yaml", BASE_CONFIG)
    _write(repo / "libs" / "core" / "config.py", CONFIG_LOADER)
    _write(repo / "libs" / "modeling" / "meta_archs.py", "SPARSE = False\n")
    base = _commit(repo, "native base")
    _git(repo, "push", "upstream", f"{base}:{BASE_REF}")

    for name, policy in (
        ("thumos_i3d_sparsehead_k384_uniform.yaml", "stratified_uniform"),
        ("thumos_i3d_sparsehead_k384_hash.yaml", "video_hash_random"),
    ):
        _write(
            repo / "configs" / name,
            NATIVE_SPARSE_CONFIG.replace("stratified_uniform", policy),
        )
    _write(repo / "libs" / "modeling" / "meta_archs.py", "SPARSE = True\n")
    _write(repo / "libs" / "modeling" / "sparse_heads.py", "BUDGET = 384\n")
    _write(repo / "tests" / "test_native_grid_sparse_heads.py", "def test_k(): pass\n")
    _write(repo / "tests" / "test_sparsehead_official_config.py", "def test_cfg(): pass\n")
    candidate = _commit(repo, "native sparse head")
    _git(repo, "push", "origin", f"{candidate}:{CANDIDATE_REF}")
    return repo, base, candidate


def _collect(repo, base, candidate):
    return source_diff.collect_attestation(
        repository=repo,
        base_commit=base,
        candidate_commit=candidate,
        base_repository_url=_git(repo, "remote", "get-url", "upstream"),
        candidate_repository_url=_git(repo, "remote", "get-url", "origin"),
        base_remote="upstream",
        candidate_remote="origin",
        base_remote_ref=BASE_REF,
        candidate_remote_ref=CANDIDATE_REF,
        base_config_path="configs/thumos_i3d.yaml",
        candidate_config_path="configs/thumos_i3d_random_k384.yaml",
        intervention="selection_budget",
    )


def _collect_native_sparse(repo, base, candidate):
    return source_diff.collect_attestation(
        repository=repo,
        base_commit=base,
        candidate_commit=candidate,
        base_repository_url=_git(repo, "remote", "get-url", "upstream"),
        candidate_repository_url=_git(repo, "remote", "get-url", "origin"),
        base_remote="upstream",
        candidate_remote="origin",
        base_remote_ref=BASE_REF,
        candidate_remote_ref=CANDIDATE_REF,
        base_config_path="configs/thumos_i3d.yaml",
        candidate_config_path=(
            "configs/thumos_i3d_sparsehead_k384_uniform.yaml"
        ),
        intervention="native_grid_sparse_head_k384",
    )


def test_live_source_diff_attestation_recomputes_exactly(tmp_path):
    repo, base, candidate = _selection_repo(tmp_path)
    attestation = _collect(repo, base, candidate)

    assert attestation["validation_pass"] is True
    assert attestation["issues"] == []
    assert attestation["base"]["commit"] == base
    assert attestation["candidate"]["commit"] == candidate
    assert (
        attestation["base"]["effective_config_sha256"]
        != attestation["candidate"]["effective_config_sha256"]
    )
    assert (
        attestation["effective_config"]["changed_paths"]
        == sorted(source_diff.EFFECTIVE_CONFIG_ALLOWED_PATHS["selection_budget"])
    )
    assert attestation["effective_config"]["protected_sha256"]
    assert attestation["diff"]["changed_paths"] == [
        "configs/thumos_i3d_random_k384.yaml",
        "libs/datasets/deterministic_selection.py",
        "libs/datasets/thumos14.py",
    ]
    assert source_diff.validate_attestation_live(attestation) == attestation


def test_native_grid_sparse_head_has_one_exact_fail_closed_intervention(tmp_path):
    repo, base, candidate = _native_sparse_repo(tmp_path)
    attestation = _collect_native_sparse(repo, base, candidate)

    assert attestation["validation_pass"] is True
    assert attestation["intervention"] == "native_grid_sparse_head_k384"
    assert attestation["diff"]["changed_paths"] == sorted(
        source_diff.SOURCE_INTERVENTION_ALLOWED_PATHS[
            "native_grid_sparse_head_k384"
        ]
    )
    assert attestation["effective_config"]["changed_paths"] == sorted(
        source_diff.EFFECTIVE_CONFIG_ALLOWED_PATHS[
            "native_grid_sparse_head_k384"
        ]
    )
    assert source_diff.validate_attestation_live(attestation) == attestation


def test_live_sealed_attestation_can_be_revalidated_offline(tmp_path, monkeypatch):
    repo, base, candidate = _native_sparse_repo(tmp_path)
    attestation = _collect_native_sparse(repo, base, candidate)
    monkeypatch.setattr(
        source_diff,
        "_remote_ref_commit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("offline validation attempted a remote lookup")
        ),
    )

    assert source_diff.validate_attestation_snapshot(attestation) == attestation

    tampered = copy.deepcopy(attestation)
    tampered["diff"]["binary_sha256"] = "0" * 64
    with pytest.raises(source_diff.SourceDiffError, match="source diff differs"):
        source_diff.validate_attestation_snapshot(tampered)


def test_native_grid_sparse_head_cannot_hide_optimizer_drift(tmp_path):
    repo, base, _ = _native_sparse_repo(tmp_path)
    path = repo / "configs" / "thumos_i3d_sparsehead_k384_uniform.yaml"
    _write(path, path.read_text(encoding="utf-8") + "\nopt:\n  epochs: 29\n")
    candidate = _commit(repo, "native sparse optimizer drift")
    _publish_candidate(repo, candidate)

    with pytest.raises(
        source_diff.SourceDiffError,
        match="effective config changes protected paths",
    ):
        _collect_native_sparse(repo, base, candidate)


def test_tampered_diff_digest_fails_live_recomputation(tmp_path):
    repo, base, candidate = _selection_repo(tmp_path)
    attestation = _collect(repo, base, candidate)
    attestation["diff"]["binary_sha256"] = "0" * 64

    with pytest.raises(source_diff.SourceDiffError, match="live Git recomputation"):
        source_diff.validate_attestation_live(attestation)


def test_changed_path_outside_intervention_allowlist_is_rejected(tmp_path):
    repo, base, _ = _selection_repo(tmp_path)
    _write(repo / "eval.py", "changed evaluator\n")
    candidate = _commit(repo, "forbidden evaluator change")
    _publish_candidate(repo, candidate)

    with pytest.raises(source_diff.SourceDiffError, match="outside.*allowlist"):
        _collect(repo, base, candidate)


def test_delete_rename_and_copy_semantics_are_rejected(tmp_path):
    repo, base, _ = _selection_repo(tmp_path)
    (repo / "libs" / "datasets" / "thumos14.py").unlink()
    candidate = _commit(repo, "delete selector hook")
    _publish_candidate(repo, candidate)

    with pytest.raises(source_diff.SourceDiffError, match="status is forbidden"):
        _collect(repo, base, candidate)


def test_dirty_candidate_worktree_is_rejected(tmp_path):
    repo, base, candidate = _selection_repo(tmp_path)
    _write(repo / "untracked.txt", "not sealed\n")

    with pytest.raises(source_diff.SourceDiffError, match="not clean"):
        _collect(repo, base, candidate)


def test_remote_url_and_checked_out_candidate_are_live_bound(tmp_path):
    repo, base, candidate = _selection_repo(tmp_path)

    with pytest.raises(source_diff.SourceDiffError, match="candidate repository URL"):
        source_diff.collect_attestation(
            repository=repo,
            base_commit=base,
            candidate_commit=candidate,
            base_repository_url=_git(repo, "remote", "get-url", "upstream"),
            candidate_repository_url="https://github.com/example/wrong",
            base_remote="upstream",
            candidate_remote="origin",
            base_remote_ref=BASE_REF,
            candidate_remote_ref=CANDIDATE_REF,
            base_config_path="configs/thumos_i3d.yaml",
            candidate_config_path="configs/thumos_i3d_random_k384.yaml",
            intervention="selection_budget",
        )

    _write(repo / "configs" / "thumos_i3d_random_k384.yaml", "selection_budget: 383\n")
    later = _commit(repo, "later candidate")
    _publish_candidate(repo, later)
    assert later != candidate
    with pytest.raises(source_diff.SourceDiffError, match="not checked out"):
        _collect(repo, base, candidate)


def test_attestation_policy_payload_cannot_be_self_expanded(tmp_path):
    repo, base, candidate = _selection_repo(tmp_path)
    attestation = _collect(repo, base, candidate)
    expanded = copy.deepcopy(attestation)
    expanded["policy"]["allowed_paths"].append("eval.py")

    with pytest.raises(source_diff.SourceDiffError, match="live Git recomputation"):
        source_diff.validate_attestation_live(expanded)


def test_protected_effective_config_change_is_rejected(tmp_path):
    repo, base, _ = _selection_repo(tmp_path)
    _write(
        repo / "configs" / "thumos_i3d_random_k384.yaml",
        CANDIDATE_CONFIG + "\nopt:\n  learning_rate: 0.001\n",
    )
    candidate = _commit(repo, "hidden optimizer change")
    _publish_candidate(repo, candidate)

    with pytest.raises(
        source_diff.SourceDiffError,
        match="effective config changes protected paths",
    ):
        _collect(repo, base, candidate)


def test_candidate_config_must_be_regular_non_executable_blob(tmp_path):
    repo, base, _ = _selection_repo(tmp_path)
    relative = "configs/thumos_i3d_random_k384.yaml"
    if sys.platform != "win32":
        config_path = repo / relative
        config_path.chmod(config_path.stat().st_mode | 0o111)
    _git(repo, "update-index", "--chmod=+x", relative)
    # Commit the staged index change directly. Calling _commit() would run
    # ``git add -A`` and, on POSIX, restore the non-executable worktree mode.
    _git(repo, "commit", "-m", "executable candidate config")
    candidate = _git(repo, "rev-parse", "HEAD")
    _publish_candidate(repo, candidate)

    with pytest.raises(
        source_diff.SourceDiffError,
        match="regular non-executable blob",
    ):
        _collect(repo, base, candidate)


def test_remote_candidate_ref_must_publish_declared_commit(tmp_path):
    repo, base, candidate = _selection_repo(tmp_path)
    _write(repo / "libs" / "datasets" / "thumos14.py", "SELECTOR = 'later'\n")
    later = _commit(repo, "unpublished later candidate")
    assert later != candidate

    with pytest.raises(
        source_diff.SourceDiffError,
        match="candidate remote ref does not resolve",
    ):
        _collect(repo, base, later)


def test_cli_builds_a_live_remote_bound_attestation(tmp_path):
    repo, base, candidate = _selection_repo(tmp_path)
    output = tmp_path / "SOURCE_DIFF_ATTESTATION.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(BUILDER_PATH),
            "--repository",
            str(repo),
            "--base-commit",
            base,
            "--candidate-commit",
            candidate,
            "--base-repository-url",
            _git(repo, "remote", "get-url", "upstream"),
            "--candidate-repository-url",
            _git(repo, "remote", "get-url", "origin"),
            "--base-remote-ref",
            BASE_REF,
            "--candidate-remote-ref",
            CANDIDATE_REF,
            "--candidate-config-path",
            "configs/thumos_i3d_random_k384.yaml",
            "--intervention",
            "selection_budget",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert json.loads(completed.stdout) == payload
    assert source_diff.validate_attestation_live(payload) == payload
